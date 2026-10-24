"""World-model state layout, canonicalisation, reward and termination.

State S (dim 297) = tracked-body pose/velocity + joint state + previous action:
  pos(14x3) | rot6d(14x6) | lin_vel(14x3) | ang_vel(14x3) | q(29) | qd(29) | a_prev(29)
Body 0 is the pelvis (root and tracking anchor).  Positions are env-local world coordinates.
"""
import torch

from .geom import mat_to_6d, quat_to_mat, rot_angle_sq, rot_z, sixd_to_mat, yaw_of

NB, NJ = 14, 29
O_POS, O_ROT, O_LV, O_AV = 0, 42, 126, 168
O_Q, O_QD, O_AP = 210, 239, 268
SDIM = 297
XI_DIM = 1 + NJ + 3  # friction, encoder bias, torso COM offset
DT = 0.02

# Tracking reward (Table A.2); weights multiplied by the control period.
W_ROOT_POS, W_ROOT_ORI, W_BODY_POS, W_BODY_ORI, W_LV, W_AV = 0.5, 0.5, 1.0, 1.0, 1.0, 1.0
W_ACT, W_JLIM = -0.1, -2.0
R_POS_MAX = W_ROOT_POS + W_ROOT_ORI + W_BODY_POS + W_BODY_ORI + W_LV + W_AV  # = 5
TERM_Z, TERM_TILT = 0.25, 0.8


def pack_state(pos, R, lv, av, q, qd, ap):
  """pos [B,14,3], R [B,14,3,3] -> S [B,297]."""
  B = pos.shape[0]
  return torch.cat([pos.reshape(B, -1), mat_to_6d(R).reshape(B, -1), lv.reshape(B, -1),
                    av.reshape(B, -1), q, qd, ap], -1)


def unpack(S):
  sh = S.shape[:-1]
  return (S[..., O_POS:O_ROT].reshape(*sh, NB, 3), S[..., O_ROT:O_LV].reshape(*sh, NB, 6),
          S[..., O_LV:O_AV].reshape(*sh, NB, 3), S[..., O_AV:O_Q].reshape(*sh, NB, 3),
          S[..., O_Q:O_QD], S[..., O_QD:O_AP], S[..., O_AP:SDIM])


def canon(S, frame=None):
  """Remove root xy translation and heading. Returns (Sc, root_xy, yaw).

  `frame=(root_xy, yaw)` canonicalises w.r.t. another state's frame (used for the fit loss)."""
  pos, r6, lv, av, q, qd, ap = unpack(S)
  if frame is None:
    yaw = torch.atan2(r6[..., 0, 1], r6[..., 0, 0])
    root_xy = pos[..., 0, :2]
  else:
    root_xy, yaw = frame
  c, s = torch.cos(yaw)[..., None, None], torch.sin(yaw)[..., None, None]

  def rot(v):  # rotate vectors by -yaw (about z)
    x, y, z = v[..., 0:1], v[..., 1:2], v[..., 2:3]
    return torch.cat([c * x + s * y, -s * x + c * y, z], -1)

  p = pos.clone()
  p = torch.cat([p[..., :2] - root_xy[..., None, :], p[..., 2:]], -1)
  Sc = torch.cat([rot(p).flatten(-2), torch.cat([rot(r6[..., :3]), rot(r6[..., 3:])], -1).flatten(-2),
                  rot(lv).flatten(-2), rot(av).flatten(-2), q, qd, ap], -1)
  return Sc, root_xy, yaw


def decanon(Sc, root_xy, yaw):
  pos, r6, lv, av, q, qd, ap = unpack(Sc)
  c, s = torch.cos(yaw)[..., None, None], torch.sin(yaw)[..., None, None]

  def rot(v):  # rotate by +yaw
    x, y, z = v[..., 0:1], v[..., 1:2], v[..., 2:3]
    return torch.cat([c * x - s * y, s * x + c * y, z], -1)

  p = rot(pos)
  p = torch.cat([p[..., :2] + root_xy[..., None, :], p[..., 2:]], -1)
  return torch.cat([p.flatten(-2), torch.cat([rot(r6[..., :3]), rot(r6[..., 3:])], -1).flatten(-2),
                    rot(lv).flatten(-2), rot(av).flatten(-2), q, qd, ap], -1)


def state_mats(S):
  return sixd_to_mat(unpack(S)[1])


def reward_terms(S, a_prev, a, ref, qlim):
  """Analytic reward r_{t+1} (x dt) and termination features for state s_{t+1}.

  S: s_{t+1}; a_prev: a_{t-1} (= s_t's previous-action slot); a: a_t.
  ref: dict(pos [...,14,3], R [...,14,3,3], lv, av) at the frame aligned with s_{t+1}.
  """
  pos, r6, lv, av, q, qd, _ = unpack(S)
  R = sixd_to_mat(r6)
  ap, aR = pos[..., 0, :], R[..., 0, :, :]
  rp, rR = ref["pos"][..., 0, :], ref["R"][..., 0, :, :]
  e_rp = ((rp - ap) ** 2).sum(-1)
  e_ro = rot_angle_sq(rR, aR)
  Rz = rot_z(yaw_of(aR @ rR.transpose(-1, -2)))
  dpos = torch.cat([ap[..., :2], rp[..., 2:]], -1)
  bpos = dpos[..., None, :] + (Rz[..., None, :, :] @ (ref["pos"] - rp[..., None, :])[..., None]).squeeze(-1)
  bR = Rz[..., None, :, :] @ ref["R"]
  e_bp = ((bpos - pos) ** 2).sum(-1).mean(-1)
  e_bo = rot_angle_sq(bR, R).mean(-1)
  e_lv = ((ref["lv"] - lv) ** 2).sum(-1).mean(-1)
  e_av = ((ref["av"] - av) ** 2).sum(-1).mean(-1)
  r_track = (W_ROOT_POS * torch.exp(-e_rp / 0.3 ** 2) + W_ROOT_ORI * torch.exp(-e_ro / 0.4 ** 2)
             + W_BODY_POS * torch.exp(-e_bp / 0.3 ** 2) + W_BODY_ORI * torch.exp(-e_bo / 0.4 ** 2)
             + W_LV * torch.exp(-e_lv / 1.0 ** 2) + W_AV * torch.exp(-e_av / 3.14 ** 2))
  pen_act = ((a - a_prev) ** 2).sum(-1)
  pen_lim = (torch.clamp(q - qlim[1], min=0) + torch.clamp(qlim[0] - q, min=0)).sum(-1)
  r = DT * (r_track + W_ACT * pen_act + W_JLIM * pen_lim)
  e_z = (rp[..., 2] - ap[..., 2]).abs()
  e_tilt = (rR[..., 2, 2] - aR[..., 2, 2]).abs()
  return r, e_z, e_tilt
