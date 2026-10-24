"""Downstream Velocity-Command Interface (Section 4.5, Appendix B, Appendix G, Table A.6).

Trains a downstream velocity encoder (MLP 256, 256 -> 32-dim tanh latent z) sharing the trained
motion decoder and differentiable world model on the walking/running subset of LAFAN1 (Appendix G.1):
  - RL encoder: optimizes the velocity-tracking task reward (Table A.6) through H imagined steps of the
    world model (with online simulator data collection and world-model updates), plus a 0.1 latent-alignment
    MSE on reference-command rollouts.
  - Distillation baseline: optimizes the motion-tracking reward (Table A.2) on reference commands read from
    the reference motion, plus the 0.1 latent-alignment MSE.
Evaluates both on a grid of step velocity commands (vx, vy, wz) from standing (Appendix G.2).
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import state as st
from .env import SimEnv
from .features import HIST, OBS_DIM, add_noise, goal_feat, norm_xi, proprio_obs
from .geom import sixd_to_mat
from .library import Library
from .models import Policy, WorldModel, mlp
from .replay import Replay
from .sampler import GAMMA

CMD_SCALE = torch.tensor([2.0, 1.32, 3.35])  # max command magnitudes (Appendix G.1)


class VelocityEncoder(nn.Module):
  """Table A.4: Velocity encoder enc_{theta,vel} -> MLP(256, 256) -> 32-dim tanh latent z."""

  def __init__(self):
    super().__init__()
    self.net = mlp(3 + HIST * OBS_DIM, [256, 256], 32)

  def forward(self, cmd, hist_norm):
    c = cmd / CMD_SCALE.to(cmd.device)
    return torch.tanh(self.net(torch.cat([c, hist_norm.flatten(-2)], -1)))


class VelocityCritic(nn.Module):
  """Table A.4: Critic V_{psi,vel} -> MLP(256, 128) -> scalar."""

  def __init__(self):
    super().__init__()
    self.net = mlp(st.SDIM + 3 + st.XI_DIM, [256, 128], 1)

  def forward(self, Sc_n, cmd, xi):
    c = cmd / CMD_SCALE.to(cmd.device)
    return self.net(torch.cat([Sc_n, c, norm_xi(xi)], -1)).squeeze(-1)


def ref_velocity_command(lib: Library, idx):
  """Centered 25-frame mean of the reference root velocity in the reference's heading frame (Appendix G.1)."""
  offs = torch.arange(-12, 13, device=idx.device)
  idx_c = idx.clamp(0, lib.F - 1)
  f = (idx_c[..., None] + offs).clamp(0, lib.F - 1)
  last = lib.last_frame[idx_c][..., None]
  first = lib.clip_start[lib.clip_id[idx_c]][..., None]
  f = torch.minimum(torch.maximum(f, first), last)
  lv = lib.lv[f, 0]       # [..., 25, 3] world linear velocity of pelvis
  av = lib.av[f, 0]       # [..., 25, 3] world angular velocity of pelvis
  R = lib.R[f, 0]         # [..., 25, 3, 3]
  yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
  c, s = torch.cos(yaw), torch.sin(yaw)
  vx = c * lv[..., 0] + s * lv[..., 1]
  vy = -s * lv[..., 0] + c * lv[..., 1]
  wz = av[..., 2]
  return torch.stack([vx.mean(-1), vy.mean(-1), wz.mean(-1)], -1)


def velocity_reward_and_fall(S, a_prev, a, cmd, qlim):
  """Table A.6: Velocity-tracking reward and fall termination on state s_{t+1}.

  Body 0 is pelvis (root); body 7 in BODY_NAMES is torso_link.
  """
  pos, r6, lv, av, q, qd, _ = st.unpack(S)
  R = sixd_to_mat(r6)
  R_root = R[..., 0, :, :]
  yaw = torch.atan2(R_root[..., 1, 0], R_root[..., 0, 0])
  c, s = torch.cos(yaw), torch.sin(yaw)
  lv_root = lv[..., 0, :]
  vx = c * lv_root[..., 0] + s * lv_root[..., 1]
  vy = -s * lv_root[..., 0] + c * lv_root[..., 1]
  vxy = torch.stack([vx, vy], -1)
  wz = av[..., 0, 2]

  # Torso (index 7 in BODY_NAMES) in heading frame
  R_torso = R[..., 7, :, :]
  av_torso = av[..., 7, :]
  tz_w = R_torso[..., :, 2]
  sin_side = -s * tz_w[..., 0] + c * tz_w[..., 1]
  w_roll = c * av_torso[..., 0] + s * av_torso[..., 1]
  w_pitch = -s * av_torso[..., 0] + c * av_torso[..., 1]

  r_vxy = 2.0 * torch.exp(-((vxy - cmd[..., :2]) ** 2).sum(-1) / (0.5 ** 2))
  r_wz = 2.0 * torch.exp(-((wz - cmd[..., 2]) ** 2) / 0.5)
  pen_act = ((a - a_prev) ** 2).sum(-1)
  pen_lim = (torch.clamp(q - qlim[1], min=0) + torch.clamp(qlim[0] - q, min=0)).sum(-1)
  pen_lean = sin_side ** 2
  pen_rp = w_roll ** 2 + w_pitch ** 2

  r = st.DT * (r_vxy + r_wz - 0.3 * pen_act - 2.0 * pen_lim - 4.0 * pen_lean - 0.05 * pen_rp)
  h_root = pos[..., 0, 2]
  uz = -R_root[..., 2, 2]  # z-component of unit gravity (0,0,-1) in root frame
  fall = (h_root < 0.35) | (1.0 + uz > 0.7)
  logit = 12.0 * torch.maximum((0.35 - h_root) / 0.35, (1.0 + uz - 0.7) / 0.7)
  d_fall = torch.sigmoid(logit)
  return r, d_fall, fall, torch.stack([vx, vy, wz], -1)


def sample_random_commands(B, device):
  """Appendix G.1: vx in [-1.5, 2.0], vy in [-1.32, 1.32], wz in [-3.35, 3.35], zeroed with prob 0.1."""
  u = torch.rand(B, 3, device=device)
  vx = -1.5 + 3.5 * u[:, 0]
  vy = -1.32 + 2.64 * u[:, 1]
  wz = -3.35 + 6.70 * u[:, 2]
  cmd = torch.stack([vx, vy, wz], -1)
  zero_mask = torch.rand(B, 1, device=device) < 0.1
  return torch.where(zero_mask, torch.zeros_like(cmd), cmd)


def find_standing_frame(lib: Library):
  """Find the most stationary upright standing frame in the locomotion library."""
  spd = lib.lv[:, 0].norm(dim=-1) + lib.av[:, 0].norm(dim=-1)
  upright = (lib.pos[:, 0, 2] > 0.74) & (lib.R[:, 0, 2, 2] > 0.96)
  score = torch.where(upright, spd, spd + 100.0)
  return int(score.argmin().item())


@torch.no_grad()
def evaluate_step_response(lib, pol, vel_enc, stand_frame, device="cuda:0", n_grid=9, steps=250, n_trials=4):
  """Appendix G.2 step-response test across the 3 command planes (vx-vy, vx-wz, vy-wz) with 4 trials."""
  cmds = []
  vxs = torch.linspace(-1.5, 2.0, n_grid)
  vys = torch.linspace(-1.32, 1.32, n_grid)
  wzs = torch.linspace(-3.35, 3.35, n_grid)
  for x in vxs:
    for y in vys:
      if abs(x) + abs(y) > 1e-4:
        cmds.append([x.item(), y.item(), 0.0])
  for x in vxs:
    for w in wzs:
      if abs(x) + abs(w) > 1e-4:
        cmds.append([x.item(), 0.0, w.item()])
  for y in vys:
    for w in wzs:
      if abs(y) + abs(w) > 1e-4:
        cmds.append([0.0, y.item(), w.item()])
  cmd_single = torch.tensor(cmds, dtype=torch.float32, device=device)
  M = cmd_single.shape[0]
  cmd_t = cmd_single.repeat_interleave(n_trials, dim=0)
  total_envs = M * n_trials

  env = SimEnv(lib, total_envs, device, dr_on=False, reset_noise=True, fixed_friction=1.0)
  env.frames_fixed = torch.full((total_envs,), stand_frame, dtype=torch.long, device=device)
  env.reset_all()
  q0, qlim = env.q0, env.qlim
  S, xi = env.extract(), env.xi()
  o = proprio_obs(S, xi, q0)
  hist = o[:, None].repeat(1, HIST, 1)
  fell = torch.zeros(total_envs, dtype=torch.bool, device=device)
  vel_acc = torch.zeros(total_envs, 3, device=device)
  ss_count = 0
  for t in range(steps):
    hn = pol.obs_norm(hist)
    z = vel_enc(cmd_t, hn)
    a = pol.dec(torch.cat([z, hn.flatten(-2)], -1))
    S, _, _, _, ap = env.step(a)
    _, _, f_step, v_meas = velocity_reward_and_fall(S, ap, a, cmd_t, qlim)
    fell |= f_step
    if t >= steps - 100:  # last 2 seconds (100 steps at 50 Hz)
      vel_acc += v_meas
      ss_count += 1
    hist = torch.cat([hist[:, 1:], proprio_obs(S, xi, q0)[:, None]], 1)
  v_mean = vel_acc / max(ss_count, 1)
  rel_err_all = (v_mean - cmd_t).norm(dim=-1) / cmd_t.norm(dim=-1).clamp(min=1e-4)
  rel_err_all = rel_err_all.view(M, n_trials)
  fell_all = fell.view(M, n_trials)

  # Appendix G.2: per-command value is the median over trials that did not fall; if all fell, cap at 0.9
  rel_err = torch.zeros(M, device=device)
  cmd_fell_all = torch.zeros(M, dtype=torch.bool, device=device)
  for i in range(M):
    ok = ~fell_all[i]
    if ok.any():
      rel_err[i] = rel_err_all[i][ok].median().clamp(max=0.9)
    else:
      rel_err[i] = 0.9
      cmd_fell_all[i] = True

  del env
  torch.cuda.empty_cache()
  return dict(cmds=cmd_single.cpu(), rel_err=rel_err.cpu(), fell=cmd_fell_all.cpu(),
              median_err=rel_err.median().item(), fall_count=int(cmd_fell_all.sum().item()))


def train_velocity_encoders(ckpt_path, lib_path="data/lib/lafan1_g1.npz", out_dir="runs/velocity_interface",
                            iters=800, batch=2048, horizon=16, lr=3e-4):
  dev = "cuda:0"
  # Appendix G.1: walking and running library filtered from LAFAN1
  full_lib = Library(lib_path, dev)
  walk_clips = [i for i, n in enumerate(full_lib.clip_names) if any(k in n for k in ("walk", "run", "sprint"))]
  del full_lib
  lib = Library(lib_path, dev, clips=walk_clips)
  stand_frame = find_standing_frame(lib)
  print(f"[vel] Loaded {lib.n_clips} locomotion clips ({lib.F} frames), stand_frame={stand_frame}", flush=True)

  ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
  wm = WorldModel().to(dev)
  wm.load_state_dict(ck["wm"])
  pol = Policy().to(dev)
  pol.load_state_dict(ck["pol"])
  pol.eval()
  pol.requires_grad_(False)

  enc_rl = VelocityEncoder().to(dev)
  enc_dist = VelocityEncoder().to(dev)
  crit_rl = VelocityCritic().to(dev)
  opt_wm = torch.optim.Adam(wm.parameters(), lr=1e-4)
  opt_rl = torch.optim.Adam(enc_rl.parameters(), lr=lr, betas=(0.9, 0.95))
  opt_v = torch.optim.Adam(crit_rl.parameters(), lr=lr)
  opt_dist = torch.optim.Adam(enc_dist.parameters(), lr=lr, betas=(0.9, 0.95))

  # Online simulator environment (2048 envs: first 1024 rolled out by enc_rl, second 1024 by pol/enc_dist)
  N = 2048
  H_half = N // 2
  env = SimEnv(lib, N, dev)
  q0, qlim = env.q0, env.qlim
  buf = Replay(400, N, dev)
  S = env.extract()
  xi = env.xi()
  ob = add_noise(proprio_obs(S, xi, q0))
  hist = ob[:, None].repeat(1, HIST, 1)

  # Warmup replay buffer with 150 steps of motion policy + latent-alignment pretraining
  with torch.no_grad():
    for _ in range(150):
      idx = env.ref_idx.clone()
      g = goal_feat(S, idx, lib)
      act = pol(hist, g) + 0.05 * torch.randn(N, st. NJ, device=dev)
      S2, _, term, trunc, _ = env.step(act)
      done = term | trunc
      buf.push(S, act, ob, xi, idx, env.ep_step.clone(), term, done)
      ids = done.nonzero().squeeze(-1)
      if len(ids) > 0:
        env.reset_ids(ids)
        S2 = env.extract()
        xi = env.xi()
      S = S2
      ob = add_noise(proprio_obs(S, xi, q0))
      hist = torch.cat([hist[:, 1:], ob[:, None]], 1)
      if len(ids) > 0:
        hist[ids] = ob[ids][:, None, :]

  # Pre-align both velocity encoders to the motion encoder latent on reference commands (200 steps)
  for _ in range(200):
    S0, hist0, _, idx0 = buf.sample_seeds(batch)
    with torch.no_grad():
      g = goal_feat(S0, idx0, lib)
      z_mot = torch.tanh(pol.enc(pol.goal_norm(g).flatten(-2)))
      cmd_ref = ref_velocity_command(lib, idx0)
      hn = pol.obs_norm(hist0)
    loss_pre = F.mse_loss(enc_rl(cmd_ref, hn), z_mot) + F.mse_loss(enc_dist(cmd_ref, hn), z_mot)
    opt_rl.zero_grad(set_to_none=True)
    opt_dist.zero_grad(set_to_none=True)
    loss_pre.backward()
    opt_rl.step()
    opt_dist.step()
  # Copy pre-aligned weights so both start from identical initialization
  enc_dist.load_state_dict(enc_rl.state_dict())

  env_cmd = sample_random_commands(H_half, dev)
  env_use_ref = torch.rand(H_half, device=dev) < 0.5

  os.makedirs(out_dir, exist_ok=True)
  for it in range(1, iters + 1):
    # 1. Online simulator step (Appendix G.1: half velocity interface, half motion/distillation)
    with torch.no_grad():
      idx = env.ref_idx.clone()
      hn = pol.obs_norm(hist)
      cmd_ref_half = ref_velocity_command(lib, idx[:H_half])
      cmd_half = torch.where(env_use_ref[:, None], cmd_ref_half, env_cmd)
      z_rl_sim = enc_rl(cmd_half, hn[:H_half])
      a_rl_sim = pol.dec(torch.cat([z_rl_sim, hn[:H_half].flatten(-2)], -1))
      g_second = goal_feat(S[H_half:], idx[H_half:], lib)
      a_mot_sim = pol(hist[H_half:], g_second)
      act = torch.cat([a_rl_sim, a_mot_sim], 0) + 0.05 * torch.randn(N, st.NJ, device=dev)
      S2, _, term, trunc, ap = env.step(act)
      # For velocity envs, use fall termination instead of motion tracking termination
      _, _, fall_rl, _ = velocity_reward_and_fall(S2[:H_half], ap[:H_half], act[:H_half], cmd_half, qlim)
      term_combined = torch.cat([fall_rl, term[H_half:]], 0)
      done = term_combined | trunc
      buf.push(S, act, ob, xi, idx, env.ep_step.clone(), term_combined, done)
      ids = done.nonzero().squeeze(-1)
      if len(ids) > 0:
        env.reset_ids(ids)
        S2 = env.extract()
        xi = env.xi()
        rl_rst = ids[ids < H_half]
        if len(rl_rst) > 0:
          env_cmd[rl_rst] = sample_random_commands(len(rl_rst), dev)
          env_use_ref[rl_rst] = torch.rand(len(rl_rst), device=dev) < 0.5
      if it % 250 == 0:
        env_cmd = sample_random_commands(H_half, dev)
        env_use_ref = torch.rand(H_half, device=dev) < 0.5
      S = S2
      ob = add_noise(proprio_obs(S, xi, q0))
      hist = torch.cat([hist[:, 1:], ob[:, None]], 1)
      if len(ids) > 0:
        hist[ids] = ob[ids][:, None, :]

    # 2. World model fine-tuning step on recent buffer windows (keeps WM accurate on velocity rollouts)
    wm.requires_grad_(True)
    Sw, Aw, Xw, _, _ = buf.sample_windows(1024, 8)
    Sh = Sw[:, 0]
    wm_loss = 0.0
    for j in range(8):
      Sh = wm.next_state(Sh, Aw[:, j], Xw)
      tgt = Sw[:, j + 1]
      fr = st.canon(tgt)[1:]
      err = ((st.canon(Sh, fr)[0] - st.canon(tgt, fr)[0]) / wm.err_sd)[..., :st.O_AP]
      wm_loss = wm_loss + (err ** 2).mean(-1).mean()
    opt_wm.zero_grad(set_to_none=True)
    wm_loss.backward()
    nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
    opt_wm.step()
    wm.requires_grad_(False)

    # 3. Sample seed states for imagination
    S0, hist0, xi_b, idx0 = buf.sample_seeds(batch)
    B = S0.shape[0]

    # ---- 4. Distillation encoder update (motion-tracking reward + 0.1 latent alignment) ----
    Si, hi, idxi = S0.clone(), hist0.clone(), idx0.clone()
    c = torch.ones(B, device=dev)
    J_dist, align_dist = torch.zeros(B, device=dev), 0.0
    for t in range(horizon):
      if t > 0:
        hi = torch.cat([hi[:, 1:], add_noise(proprio_obs(Si, xi_b, q0))[:, None]], 1)
      g = goal_feat(Si, idxi, lib)
      with torch.no_grad():
        z_mot = torch.tanh(pol.enc(pol.goal_norm(g).flatten(-2)))
      cmd_ref = ref_velocity_command(lib, idxi)
      hn = pol.obs_norm(hi)
      z_vel = enc_dist(cmd_ref, hn)
      align_dist = align_dist + F.mse_loss(z_vel, z_mot)
      a_step = pol.dec(torch.cat([z_vel, hn.flatten(-2)], -1))
      idx1 = lib.clamp(idxi + 1)
      Si, r, d, _, _ = wm.step(Si, a_step, xi_b, lib.ref(idx1), qlim)
      J_dist = J_dist + (GAMMA ** t) * c * r
      c = c * (1 - d)
      idxi = idx1
    loss_dist = -J_dist.mean() + 0.1 * (align_dist / horizon)
    opt_dist.zero_grad(set_to_none=True)
    loss_dist.backward()
    nn.utils.clip_grad_norm_(enc_dist.parameters(), 1.0)
    opt_dist.step()

    # ---- 5. RL encoder update (velocity-tracking reward Table A.6 + 0.1 alignment on ref commands) ----
    Si, hi, idxi = S0.clone(), hist0.clone(), idx0.clone()
    use_ref = torch.rand(B, device=dev) < 0.5
    cmd_rand = sample_random_commands(B, dev)
    c = torch.ones(B, device=dev)
    J_rl, align_rl = torch.zeros(B, device=dev), 0.0
    Ss, cmds, rs, ds = [Si.detach()], [], [], []
    for t in range(horizon):
      if t > 0:
        hi = torch.cat([hi[:, 1:], add_noise(proprio_obs(Si, xi_b, q0))[:, None]], 1)
      cmd_ref = ref_velocity_command(lib, idxi)
      cmd = torch.where(use_ref[:, None], cmd_ref, cmd_rand)
      hn = pol.obs_norm(hi)
      z_vel = enc_rl(cmd, hn)
      if use_ref.any():
        with torch.no_grad():
          g = goal_feat(Si[use_ref], idxi[use_ref], lib)
          z_mot = torch.tanh(pol.enc(pol.goal_norm(g).flatten(-2)))
        align_rl = align_rl + F.mse_loss(z_vel[use_ref], z_mot)
      a_step = pol.dec(torch.cat([z_vel, hn.flatten(-2)], -1))
      a_prev = Si[..., st.O_AP:]
      S2 = wm.next_state(Si, a_step, xi_b)
      r_vel, d_fall, _, _ = velocity_reward_and_fall(S2, a_prev, a_step, cmd, qlim)
      J_rl = J_rl + (GAMMA ** t) * c * r_vel
      c = c * (1 - d_fall)
      cmds.append(cmd.detach())
      rs.append(r_vel.detach())
      ds.append(d_fall.detach())
      Si = S2
      idxi = lib.clamp(idxi + 1)
      Ss.append(Si.detach())
    crit_rl.requires_grad_(False)
    Sc_H = (st.canon(Si)[0] - wm.in_mu) / wm.in_sd
    VH = crit_rl(Sc_H, cmds[-1], xi_b).clamp(-2.0, 8.0)
    J_rl_total = J_rl + (GAMMA ** horizon) * c * VH
    loss_rl = -J_rl_total.mean() + 0.2 * (align_rl / horizon)
    opt_rl.zero_grad(set_to_none=True)
    loss_rl.backward()
    nn.utils.clip_grad_norm_(enc_rl.parameters(), 1.0)
    opt_rl.step()
    crit_rl.requires_grad_(True)

    # Critic fit (5 gradient updates per iteration)
    with torch.no_grad():
      Ss_t = torch.stack(Ss, 1)
      Sn = (st.canon(Ss_t)[0] - wm.in_mu) / wm.in_sd
      cmds_t = torch.stack(cmds + [cmds[-1]], 1)
      xi_e = xi_b[:, None].expand(-1, horizon + 1, -1)
      V = crit_rl(Sn, cmds_t, xi_e).clamp(-2.0, 8.0)
      R = torch.zeros(B, horizon + 1, device=dev)
      R[:, horizon] = V[:, horizon]
      rs_t, ds_t = torch.stack(rs, 1), torch.stack(ds, 1)
      for t in range(horizon - 1, -1, -1):
        R[:, t] = rs_t[:, t] + GAMMA * (1 - ds_t[:, t]) * (0.05 * V[:, t + 1] + 0.95 * R[:, t + 1])
    flat_Sn = Sn[:, :horizon].reshape(-1, st.SDIM)
    flat_cmd = cmds_t[:, :horizon].reshape(-1, 3)
    flat_xi = xi_e[:, :horizon].reshape(-1, st.XI_DIM)
    flat_R = R[:, :horizon].reshape(-1)
    for _ in range(5):
      v_loss = F.mse_loss(crit_rl(flat_Sn, flat_cmd, flat_xi), flat_R)
      opt_v.zero_grad(set_to_none=True)
      v_loss.backward()
      nn.utils.clip_grad_norm_(crit_rl.parameters(), 1.0)
      opt_v.step()

    if it % 100 == 0:
      print(f"[vel] it={it}/{iters} r_rl={J_rl.mean().item()/horizon:.4f} r_dist={J_dist.mean().item()/horizon:.4f} "
            f"wm_loss={wm_loss.item():.4f} v_loss={v_loss.item():.4f}", flush=True)

  del env
  torch.cuda.empty_cache()

  res_rl = evaluate_step_response(lib, pol, enc_rl, stand_frame, dev)
  res_dist = evaluate_step_response(lib, pol, enc_dist, stand_frame, dev)
  rl_wins = int((res_rl["rel_err"] < res_dist["rel_err"]).sum().item())
  dist_wins = int((res_dist["rel_err"] < res_rl["rel_err"]).sum().item())
  total = len(res_rl["rel_err"])
  print(f"[vel step test] total={total} | RL better: {rl_wins} | Distill better: {dist_wins} | "
        f"RL median err={res_rl['median_err']:.3f} (falls={res_rl['fall_count']}) | "
        f"Distill median err={res_dist['median_err']:.3f} (falls={res_dist['fall_count']})", flush=True)
  torch.save(dict(enc_rl=enc_rl.state_dict(), enc_dist=enc_dist.state_dict(),
                  res_rl=res_rl, res_dist=res_dist, rl_wins=rl_wins, dist_wins=dist_wins, total=total),
             os.path.join(out_dir, "velocity_results.pt"))


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--ckpt", default="runs/supermimic_lafan_1gpu/ckpt.pt")
  p.add_argument("--out", default="runs/velocity_interface")
  p.add_argument("--iters", type=int, default=800)
  args = p.parse_args()
  train_velocity_encoders(args.ckpt, out_dir=args.out, iters=args.iters)
