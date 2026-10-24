"""SuperMimic training loop (Algorithm A.1): collect -> fit world model -> improve policy by BPTT through the
frozen world model, critic bootstrapping beyond H imagined steps."""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import state as st
from .env import SimEnv
from .evaluate import Evaluator
from .features import HIST, add_noise, goal_feat, proprio_obs
from .library import Library
from .models import Critic, Policy, WorldModel
from .replay import Replay
from .sampler import GAMMA, ResetSampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_args():
  p = argparse.ArgumentParser()
  p.add_argument("--lib", default="data/lib/lafan1_g1.npz")
  p.add_argument("--out", default="runs/supermimic_lafan")
  p.add_argument("--envs", type=int, default=4096)
  p.add_argument("--buffer", type=int, default=1000)
  p.add_argument("--horizon", type=int, default=32)
  p.add_argument("--window", type=int, default=8)
  p.add_argument("--lr", type=float, default=2e-4)
  p.add_argument("--lr_final", type=float, default=0.03)
  p.add_argument("--sched_iters", type=int, default=250_000)
  p.add_argument("--iters", type=int, default=320_000)
  p.add_argument("--wm_batches", type=int, default=10)
  p.add_argument("--wm_batch", type=int, default=2048)
  p.add_argument("--pol_batch", type=int, default=4096)
  p.add_argument("--utd", type=float, default=0.4)
  p.add_argument("--action_noise", type=float, default=0.1)
  p.add_argument("--no_critic", action="store_true")
  p.add_argument("--warmup_steps", type=int, default=64)
  p.add_argument("--wm_warmup_iters", type=int, default=300)
  p.add_argument("--time_budget_h", type=float, default=1e9, help="training wall-clock budget (excl. eval)")
  p.add_argument("--eval_every_min", type=float, default=15.0)
  p.add_argument("--eval_steps", type=int, default=3000, help="intermediate eval length (full clips at end)")
  p.add_argument("--final_eval_steps", type=int, default=0, help="0 = full clips")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--clips", type=str, default="", help="comma-separated clip indices (default all)")
  return p.parse_args()


class Trainer:
  def __init__(self, a):
    self.a = a
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    self.dev = "cuda:0"
    clips = [int(c) for c in a.clips.split(",")] if a.clips else None
    self.lib = Library(a.lib, self.dev, clips)
    self.env = SimEnv(self.lib, a.envs, self.dev)
    self.sampler = ResetSampler(self.lib, a.envs)
    self.env.start_fn = self.sampler
    self.wm = WorldModel().to(self.dev)
    self.pol = Policy().to(self.dev)
    self.critic = Critic().to(self.dev)
    self.buf = Replay(a.buffer, a.envs, self.dev)
    self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=a.lr)
    self.opt_pi = torch.optim.Adam(self.pol.parameters(), lr=a.lr, betas=(0.9, 0.95))
    self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=a.lr)
    self.q0, self.qlim = self.env.q0, self.env.qlim
    self.init_wm_stats()
    os.makedirs(a.out, exist_ok=True)
    self.log_f = open(os.path.join(a.out, "log.csv"), "w", newline="")
    self.log = csv.writer(self.log_f)
    self.log.writerow(["iter", "train_h", "env_steps", "wm_err8", "wm_bce", "J_imag", "v_loss",
                       "train_rew", "term_rate", "mean_d_hat", "lr", "t_collect", "t_wm", "t_pi"])
    self.eval_f = open(os.path.join(a.out, "eval.csv"), "w", newline="")
    self.evalw = csv.writer(self.eval_f)
    self.evalw.writerow(["iter", "train_h", "norm_return", "n_failed", "eval_steps"])
    self.evaluator = None
    # collection state
    self.sampler_reset_all()
    self.pending = torch.zeros(a.envs, dtype=torch.bool, device=self.dev)
    self.pending_term = torch.zeros_like(self.pending)
    self.new_ep = torch.ones_like(self.pending)
    self.hist = torch.zeros(a.envs, HIST, 93, device=self.dev)
    self.S = self.env.extract()
    self.collect_acc = 0.0
    self.stat_rew, self.stat_n, self.stat_term, self.stat_eps = 0.0, 0, 0, 0

  def sampler_reset_all(self):
    self.env.reset_all()

  # ---------------------------------------------------------------------------------------------
  @torch.no_grad()
  def init_wm_stats(self):
    """Normalisers from the reference library (canonical states and consecutive-frame deltas)."""
    lib = self.lib
    idx = torch.randint(0, lib.F - 1, (100_000,), device=self.dev)
    idx = torch.minimum(idx, lib.last_frame[idx] - 1)

    def lib_state(i):
      B = i.shape[0]
      return st.pack_state(lib.pos[i, :], lib.R[i], lib.lv[i], lib.av[i], lib.jp[i], lib.jv[i],
                           torch.zeros(B, st.NJ, device=self.dev))

    S0, S1 = lib_state(idx), lib_state(idx + 1)
    Sc0, rxy, yaw = st.canon(S0)
    Sc1 = st.canon(S1, (rxy, yaw))[0]
    sd = Sc0.std(0).clamp(min=0.05)
    sd[st.O_AP:] = 1.0
    self.wm.in_mu.copy_(Sc0.mean(0))
    self.wm.in_mu[st.O_AP:] = 0.0
    self.wm.in_sd.copy_(sd)
    d = (Sc1 - Sc0).std(0).clamp(min=2e-3)
    self.wm.d_sd.copy_(d)
    self.wm.err_sd.copy_(sd)

  def cn(self, S):
    """Canonical + normalised state for the critic."""
    return (st.canon(S)[0] - self.wm.in_mu) / self.wm.in_sd

  # ---------------------------------------------------------------------------------------------
  @torch.no_grad()
  def collect_step(self):
    env, lib, a = self.env, self.lib, self.a
    S = self.S
    xi = env.xi()
    ob = add_noise(proprio_obs(S, xi, self.q0))
    new = self.new_ep
    self.hist = torch.cat([self.hist[:, 1:], ob[:, None]], 1)
    if new.any():
      self.hist[new] = ob[new][:, None, :]
    idx = env.ref_idx.clone()
    g = goal_feat(S, idx, lib)
    act = self.pol(self.hist, g)
    a_exec = act + torch.randn_like(act) * a.action_noise
    a_exec[self.pending] = 0.0
    self.pol.obs_norm.update(ob)
    self.pol.goal_norm.update(g[:, 0])
    self.buf.push(S, a_exec, ob, xi, idx, env.ep_step.clone(), self.pending & self.pending_term, self.pending)
    S2, r, term, trunc, _ = env.step(a_exec)
    live = ~self.pending
    self.sampler.observe(torch.where(live, r, torch.zeros_like(r)), term & live, trunc & live)
    self.stat_rew += (r[live].sum().item()) / (st.DT * st.R_POS_MAX)
    self.stat_n += int(live.sum().item())
    self.stat_term += int((term & live).sum().item())
    self.stat_eps += int(((term | trunc) & live).sum().item())
    ids = self.pending.nonzero().squeeze(-1)
    if len(ids) > 0:
      env.reset_ids(ids)
      S2 = env.extract()
    self.new_ep = self.pending.clone()
    self.pending_term = term & live
    self.pending = (term | trunc) & live
    self.S = S2

  def collect(self, n):
    for _ in range(n):
      self.collect_step()

  # ---------------------------------------------------------------------------------------------
  def wm_update(self):
    a, wm, lib = self.a, self.wm, self.lib
    k = a.window
    e8, bce_tot = 0.0, 0.0
    for _ in range(a.wm_batches):
      S, act, xi, idx, term = self.buf.sample_windows(a.wm_batch, k)
      Sh = S[:, 0]
      loss_s = 0.0
      loss_d = 0.0
      for j in range(k):
        Sh = wm.next_state(Sh, act[:, j], xi)
        tgt = S[:, j + 1]
        fr = st.canon(tgt)[1:]
        err = ((st.canon(Sh, fr)[0] - st.canon(tgt, fr)[0]) / wm.err_sd)[..., :st.O_AP]
        e = (err ** 2).mean(-1).mean()
        loss_s = loss_s + e
        _, e_z, e_t = st.reward_terms(Sh, S[:, j, st.O_AP:], act[:, j], lib.ref(lib.clamp(idx[:, j + 1])), self.qlim)
        loss_d = loss_d + F.binary_cross_entropy_with_logits(wm.term_prob_logit(e_z, e_t), term[:, j].float())
        if j == k - 1:
          e8 += e.item()
      loss = loss_s + loss_d
      self.opt_wm.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
      self.opt_wm.step()
      bce_tot += loss_d.item() / k
    return e8 / a.wm_batches, bce_tot / a.wm_batches

  def policy_update(self):
    a, wm, lib, pol, critic = self.a, self.wm, self.lib, self.pol, self.critic
    H = a.horizon
    B = a.pol_batch
    S, hist, xi, idx = self.buf.sample_seeds(B)
    B = S.shape[0]
    wm.requires_grad_(False)
    critic.requires_grad_(False)
    c = torch.ones(B, device=self.dev)
    J = torch.zeros(B, device=self.dev)
    Ss, gs, rs, ds = [S.detach()], [], [], []
    for t in range(H):
      if t > 0:
        hist = torch.cat([hist[:, 1:], add_noise(proprio_obs(S, xi, self.q0))[:, None]], 1)
      g = goal_feat(S, idx, lib)
      act = pol(hist, g)
      idx1 = lib.clamp(idx + 1)
      S2, r, d, _, _ = wm.step(S, act, xi, lib.ref(idx1), self.qlim)
      J = J + (GAMMA ** t) * c * r
      c = c * (1 - d)
      gs.append(g.detach())
      rs.append(r.detach())
      ds.append(d.detach())
      S, idx = S2, idx1
      Ss.append(S.detach())
    gH = goal_feat(S, idx, lib)
    gs.append(gH.detach())
    if not a.no_critic:
      VH = critic(self.cn(S), pol.goal_norm(gH), xi)
      J = J + (GAMMA ** H) * c * VH
    loss = -J.mean()
    self.opt_pi.zero_grad(set_to_none=True)
    loss.backward(inputs=list(pol.parameters()))
    torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
    for p_ in pol.parameters():
      if p_.grad is not None:
        p_.grad.clamp_(-0.1, 0.1)
    self.opt_pi.step()
    wm.requires_grad_(True)
    critic.requires_grad_(True)
    v_loss = 0.0
    if not a.no_critic:
      v_loss = self.critic_update(torch.stack(Ss, 1), torch.stack(gs, 1), torch.stack(rs, 1),
                                  torch.stack(ds, 1), xi)
    return J.mean().item() / H, v_loss, torch.stack(ds, 1).mean().item()

  def critic_update(self, Ss, gs, rs, ds, xi):
    a, critic, pol = self.a, self.critic, self.pol
    B, H1 = Ss.shape[:2]
    H = H1 - 1
    with torch.no_grad():
      Sn = self.cn(Ss)
      gn = pol.goal_norm(gs)
      xi_e = xi[:, None].expand(-1, H1, -1)
      V = critic(Sn, gn, xi_e)
      R = torch.zeros(B, H1, device=self.dev)
      R[:, H] = V[:, H]
      lam = 0.95
      for t in range(H - 1, -1, -1):
        R[:, t] = rs[:, t] + GAMMA * (1 - ds[:, t]) * ((1 - lam) * V[:, t + 1] + lam * R[:, t + 1])
    X = (Sn[:, :H].reshape(-1, Sn.shape[-1]), gn[:, :H].reshape(-1, *gn.shape[2:]),
         xi_e[:, :H].reshape(-1, xi.shape[-1]), R[:, :H].reshape(-1))
    n = X[0].shape[0]
    tot = 0.0
    for _ in range(5):
      perm = torch.randperm(n, device=self.dev)
      for mb in perm.chunk(4):
        loss = F.mse_loss(critic(X[0][mb], X[1][mb], X[2][mb]), X[3][mb])
        self.opt_v.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        self.opt_v.step()
        tot += loss.item()
    return tot / 20

  # ---------------------------------------------------------------------------------------------
  def set_lr(self, it):
    a = self.a
    f = min(it / a.sched_iters, 1.0)
    lr = a.lr * (a.lr_final + (1 - a.lr_final) * 0.5 * (1 + math.cos(math.pi * f)))
    for o in (self.opt_wm, self.opt_pi, self.opt_v):
      for g in o.param_groups:
        g["lr"] = lr
    return lr

  def save(self, name="ckpt.pt"):
    torch.save(dict(pol=self.pol.state_dict(), wm=self.wm.state_dict(), critic=self.critic.state_dict(),
                    args=vars(self.a)), os.path.join(self.a.out, name))

  def evaluate(self, it, train_h, full=False):
    if self.evaluator is None:
      self.evaluator = Evaluator(self.lib, self.dev)
    steps = (self.a.final_eval_steps or None) if full else self.a.eval_steps
    res = self.evaluator.run(self.pol, steps)
    self.evalw.writerow([it, f"{train_h:.4f}", f"{res['norm_return']:.4f}", res["n_failed"],
                         steps or "full"])
    self.eval_f.flush()
    print(f"[eval] it={it} train_h={train_h:.3f} norm_return={res['norm_return']:.4f} "
          f"failed={res['n_failed']}/{self.lib.n_clips}", flush=True)
    return res

  def run(self):
    a = self.a
    print(f"collecting {a.warmup_steps} warm-up steps", flush=True)
    self.collect(a.warmup_steps)
    train_t = 0.0
    last_eval = 0.0
    sync = torch.cuda.synchronize
    for i in range(a.wm_warmup_iters):
      self.wm_update()
      if i % 50 == 0:
        print(f"wm warmup {i}", flush=True)
    t_start = time.time()
    it = 0
    env_steps = a.warmup_steps * a.envs
    while it < a.iters and train_t < a.time_budget_h * 3600:
      t0 = time.time()
      lr = self.set_lr(it)
      self.collect_acc += 1.0 / a.utd
      n = int(self.collect_acc)
      self.collect_acc -= n
      self.collect(n)
      env_steps += n * a.envs
      sync(); t1 = time.time()
      e8, bce = self.wm_update()
      sync(); t2 = time.time()
      J, vl, dh = self.policy_update()
      sync(); t3 = time.time()
      train_t += t3 - t0
      it += 1
      if it % 25 == 0:
        rew = self.stat_rew / max(self.stat_n, 1)
        trm = self.stat_term / max(self.stat_eps, 1)
        self.log.writerow([it, f"{train_t/3600:.4f}", env_steps, f"{e8:.5f}", f"{bce:.5f}", f"{J:.5f}", f"{vl:.5f}",
                           f"{rew:.4f}", f"{trm:.4f}", f"{dh:.5f}", f"{lr:.2e}", f"{t1-t0:.3f}", f"{t2-t1:.3f}", f"{t3-t2:.3f}"])
        self.log_f.flush()
        self.stat_rew, self.stat_n, self.stat_term, self.stat_eps = 0.0, 0, 0, 0
        if it % 100 == 0:
          print(f"it {it} t={train_t/60:.1f}min wm_err8={e8:.4f} bce={bce:.4f} J={J:.4f} vl={vl:.4f} "
                f"rew={rew:.3f} term={trm:.3f} | collect {t1-t0:.2f}s wm {t2-t1:.2f}s pi {t3-t2:.2f}s", flush=True)
      if (train_t - last_eval) / 60 >= a.eval_every_min:
        self.save()
        self.evaluate(it, train_t / 3600)
        last_eval = train_t
    self.save()
    self.save("ckpt_final.pt")
    print("training done; full-clip evaluation", flush=True)
    res = self.evaluate(it, train_t / 3600, full=True)
    torch.save(res, os.path.join(a.out, "final_eval.pt"))


if __name__ == "__main__":
  Trainer(get_args()).run()
