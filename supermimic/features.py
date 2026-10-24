"""Policy observation (proprioceptive history) and goal (K future reference frames) features."""
import torch

from .geom import mat_to_6d, sixd_to_mat
from .library import K_FUT, Library
from .state import NJ, unpack

OBS_DIM = NJ + NJ + 3 + 3 + NJ  # q-q0, qd, imu ang vel, projected gravity, previous action
HIST = 10
GOAL_DIM = NJ + NJ + 3 + 6  # ref q, ref qd, ref root pos in robot frame, ref root rot in robot frame

# Sensor noise (Table A.1)
OBS_NOISE = torch.cat([torch.full((NJ,), 0.01), torch.full((NJ,), 0.5), torch.full((3,), 0.2),
                       torch.full((3,), 0.05), torch.zeros(NJ)])

# normalisation of the sim context xi = [friction, encoder bias(29), torso COM offset(3)]
XI_MID = torch.cat([torch.tensor([0.75]), torch.zeros(NJ + 3)])
XI_HALF = torch.cat([torch.tensor([0.45]), torch.full((NJ,), 0.01), torch.tensor([0.025, 0.05, 0.05])])


def norm_xi(xi):
  return (xi - XI_MID.to(xi.device)) / XI_HALF.to(xi.device)


def proprio_obs(S, xi, q0):
  """Noise-free observation o-bar_t (93) from the state and sim context."""
  pos, r6, lv, av, q, qd, ap = unpack(S)
  R0 = sixd_to_mat(r6[..., 0, :])
  w_b = (R0.transpose(-1, -2) @ av[..., 0, :, None]).squeeze(-1)
  g_b = -R0[..., 2, :]
  return torch.cat([q + xi[..., 1:1 + NJ] - q0, qd, w_b, g_b, ap], -1)


def add_noise(o):
  w = OBS_NOISE.to(o.device)
  return o + (torch.rand_like(o) * 2 - 1) * w


def goal_feat(S, idx, lib: Library):
  """Goal g_t from the K future reference frames idx+1..idx+K: [B, K, 67]."""
  f = lib.future(idx)
  pos, r6, *_ = unpack(S)
  R0 = sixd_to_mat(r6[..., 0, :])  # [B,3,3]
  p0 = pos[..., 0, :]
  rp = lib.pos[f][..., 0, :]  # [B,K,3]
  rR = lib.R[f][..., 0, :, :]  # [B,K,3,3]
  rel_p = torch.einsum("bji,bkj->bki", R0, rp - p0[:, None])
  rel_R = mat_to_6d(torch.einsum("bji,bkjl->bkil", R0, rR))
  return torch.cat([lib.jp[f], lib.jv[f], rel_p, rel_R], -1)
