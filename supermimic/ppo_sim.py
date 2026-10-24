"""PPO-Sim baseline (Section 4.1, Figure 4a): model-free PPO trained directly on simulator rollouts.

Shares SuperMimic's motion library, observations, goal representation, reward, termination, domain
randomization, actor/critic network architectures, and evaluation protocol.
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import state as st
from .env import SimEnv
from .evaluate import Evaluator
from .features import HIST, add_noise, goal_feat, proprio_obs
from .library import Library
from .models import Critic, Policy
from .sampler import GAMMA, ResetSampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_args():
  p = argparse.ArgumentParser()
  p.add_argument("--lib", default="data/lib/lafan1_g1.npz")
  p.add_argument("--out", default="runs/ppo_sim_lafan_1gpu")
  p.add_argument("--envs", type=int, default=4096)
  p.add_argument("--rollout_steps", type=int, default=24)
  p.add_argument("--epochs", type=int, default=5)
  p.add_argument("--minibatches", type=int, default=4)
  p.add_argument("--lr", type=float, default=3e-4)
  p.add_argument("--clip_param", type=float, default=0.2)
  p.add_argument("--entropy_coef", type=float, default=0.005)
  p.add_argument("--init_noise_std", type=float, default=0.3)
  p.add_argument("--iters", type=int, default=20000)
  p.add_argument("--time_budget_h", type=float, default=1.0)
  p.add_argument("--eval_every_min", type=float, default=10.0)
  p.add_argument("--eval_steps", type=int, default=3000)
  p.add_argument("--seed", type=int, default=0)
  return p.parse_args()


class PPOTrajectoryRunner:
  def __init__(self, a):
    self.a = a
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    self.dev = "cuda:0"
    self.lib = Library(a.lib, self.dev)
    self.env = SimEnv(self.lib, a.envs, self.dev)
    self.sampler = ResetSampler(self.lib, a.envs)
    self.env.start_fn = self.sampler
    self.pol = Policy().to(self.dev)
    self.log_std = nn.Parameter(torch.full((st.NJ,), math.log(a.init_noise_std), device=self.dev))
    self.critic = Critic().to(self.dev)
    self.opt = torch.optim.Adam(
      list(self.pol.parameters()) + [self.log_std] + list(self.critic.parameters()), lr=a.lr
    )
    self.q0, self.qlim = self.env.q0, self.env.qlim
    self.init_state_norm()
    os.makedirs(a.out, exist_ok=True)
    self.log_f = open(os.path.join(a.out, "log.csv"), "w", newline="")
    self.log = csv.writer(self.log_f)
    self.log.writerow(["iter", "train_h", "env_steps", "policy_loss", "value_loss", "train_rew", "term_rate", "std"])
    self.eval_f = open(os.path.join(a.out, "eval.csv"), "w", newline="")
    self.evalw = csv.writer(self.eval_f)
    self.evalw.writerow(["iter", "train_h", "norm_return", "n_failed", "eval_steps"])
    self.evaluator = None

    self.env.reset_all()
    self.S = self.env.extract()
    self.xi = self.env.xi()
    ob = add_noise(proprio_obs(self.S, self.xi, self.q0))
    self.hist = ob[:, None].repeat(1, HIST, 1)

  @torch.no_grad()
  def init_state_norm(self):
    lib = self.lib
    idx = torch.randint(0, lib.F - 1, (50_000,), device=self.dev)
    B = idx.shape[0]
    S0 = st.pack_state(lib.pos[idx], lib.R[idx], lib.lv[idx], lib.av[idx], lib.jp[idx], lib.jv[idx],
                       torch.zeros(B, st.NJ, device=self.dev))
    Sc0 = st.canon(S0)[0]
    self.in_mu = Sc0.mean(0)
    self.in_mu[st.O_AP:] = 0.0
    self.in_sd = Sc0.std(0).clamp(min=0.05)
    self.in_sd[st.O_AP:] = 1.0

  def cn(self, S):
    return (st.canon(S)[0] - self.in_mu) / self.in_sd

  def evaluate(self, it, train_h, full=False):
    if self.evaluator is None:
      self.evaluator = Evaluator(self.lib, self.dev)
    res = self.evaluator.run(self.pol, None if full else self.a.eval_steps)
    self.evalw.writerow([it, f"{train_h:.4f}", f"{res['norm_return']:.4f}", res["n_failed"],
                         "full" if full else self.a.eval_steps])
    self.eval_f.flush()
    print(f"[ppo eval] it={it} train_h={train_h:.3f} norm_return={res['norm_return']:.4f} "
          f"failed={res['n_failed']}/{self.lib.n_clips}", flush=True)
    return res

  def run(self):
    a = self.a
    T, N = a.rollout_steps, a.envs
    train_t = 0.0
    last_eval = 0.0
    env_steps = 0
    for it in range(1, a.iters + 1):
      if train_t >= a.time_budget_h * 3600:
        break
      t0 = time.time()
      b_hist = torch.zeros(T, N, HIST, 93, device=self.dev)
      b_g = torch.zeros(T, N, 10, 67, device=self.dev)
      b_Sc = torch.zeros(T, N, st.SDIM, device=self.dev)
      b_xi = torch.zeros(T, N, st.XI_DIM, device=self.dev)
      b_act = torch.zeros(T, N, st.NJ, device=self.dev)
      b_logp = torch.zeros(T, N, device=self.dev)
      b_rew = torch.zeros(T, N, device=self.dev)
      b_done = torch.zeros(T, N, device=self.dev)
      b_term = torch.zeros(T, N, device=self.dev)
      b_val = torch.zeros(T, N, device=self.dev)

      with torch.no_grad():
        std = self.log_std.clamp(-2.5, 0.5).exp()
        for t in range(T):
          idx = self.env.ref_idx.clone()
          g = goal_feat(self.S, idx, self.lib)
          self.pol.obs_norm.update(self.hist[:, -1])
          self.pol.goal_norm.update(g[:, 0])
          Sc_n = self.cn(self.S)
          mu = self.pol(self.hist, g)
          dist = torch.distributions.Normal(mu, std)
          act = dist.sample()
          logp = dist.log_prob(act).sum(-1)
          val = self.critic(Sc_n, self.pol.goal_norm(g), self.xi)

          S2, r, term, trunc, _ = self.env.step(act)
          done = term | trunc
          self.sampler.observe(r, term, trunc)

          b_hist[t], b_g[t], b_Sc[t], b_xi[t] = self.hist, g, Sc_n, self.xi
          b_act[t], b_logp[t], b_rew[t], b_val[t] = act, logp, r, val
          b_done[t], b_term[t] = done.float(), term.float()

          ids = done.nonzero().squeeze(-1)
          if len(ids) > 0:
            self.env.reset_ids(ids)
            S2 = self.env.extract()
            self.xi = self.env.xi()
          ob2 = add_noise(proprio_obs(S2, self.xi, self.q0))
          self.hist = torch.cat([self.hist[:, 1:], ob2[:, None]], 1)
          if len(ids) > 0:
            self.hist[ids] = ob2[ids][:, None, :]
          self.S = S2

        g_last = goal_feat(self.S, self.env.ref_idx, self.lib)
        next_val = self.critic(self.cn(self.S), self.pol.goal_norm(g_last), self.xi)
        adv = torch.zeros(T, N, device=self.dev)
        gae = torch.zeros(N, device=self.dev)
        for t in range(T - 1, -1, -1):
          nv = next_val if t == T - 1 else b_val[t + 1]
          nonterm = 1.0 - b_term[t]
          nondone = 1.0 - b_done[t]
          delta = b_rew[t] + GAMMA * nonterm * nv - b_val[t]
          gae = delta + GAMMA * 0.95 * nondone * gae
          adv[t] = gae
        ret = adv + b_val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

      # Flatten
      F_N = T * N
      f_hist = b_hist.reshape(F_N, HIST, 93)
      f_g = b_g.reshape(F_N, 10, 67)
      f_Sc = b_Sc.reshape(F_N, st.SDIM)
      f_xi = b_xi.reshape(F_N, st.XI_DIM)
      f_act = b_act.reshape(F_N, st.NJ)
      f_logp = b_logp.reshape(F_N)
      f_adv = adv.reshape(F_N)
      f_ret = ret.reshape(F_N)

      pl_tot, vl_tot = 0.0, 0.0
      for _ in range(a.epochs):
        perm = torch.randperm(F_N, device=self.dev)
        for mb in perm.chunk(a.minibatches):
          mu = self.pol(f_hist[mb], f_g[mb])
          std = self.log_std.clamp(-2.5, 0.5).exp()
          dist = torch.distributions.Normal(mu, std)
          logp = dist.log_prob(f_act[mb]).sum(-1)
          ratio = torch.exp(logp - f_logp[mb])
          s1 = ratio * f_adv[mb]
          s2 = torch.clamp(ratio, 1.0 - a.clip_param, 1.0 + a.clip_param) * f_adv[mb]
          p_loss = -torch.min(s1, s2).mean() - a.entropy_coef * dist.entropy().sum(-1).mean()
          v_pred = self.critic(f_Sc[mb], self.pol.goal_norm(f_g[mb]), f_xi[mb])
          v_loss = F.mse_loss(v_pred, f_ret[mb])
          loss = p_loss + 0.5 * v_loss
          self.opt.zero_grad(set_to_none=True)
          loss.backward()
          nn.utils.clip_grad_norm_(list(self.pol.parameters()) + [self.log_std] + list(self.critic.parameters()), 1.0)
          self.opt.step()
          pl_tot += p_loss.item()
          vl_tot += v_loss.item()

      torch.cuda.synchronize()
      train_t += time.time() - t0
      env_steps += F_N
      if it % 10 == 0:
        rew_n = (b_rew.mean().item()) / (st.DT * st.R_POS_MAX)
        trm = b_term.sum().item() / max(b_done.sum().item(), 1.0)
        n_mb = a.epochs * a.minibatches
        self.log.writerow([it, f"{train_t/3600:.4f}", env_steps, f"{pl_tot/n_mb:.5f}", f"{vl_tot/n_mb:.5f}",
                           f"{rew_n:.4f}", f"{trm:.4f}", f"{self.log_std.exp().mean().item():.3f}"])
        self.log_f.flush()
      if (train_t - last_eval) / 60 >= a.eval_every_min:
        self.evaluate(it, train_t / 3600)
        last_eval = train_t

    res = self.evaluate(it, train_t / 3600, full=True)
    torch.save(dict(pol=self.pol.state_dict(), eval=res), os.path.join(a.out, "ckpt_final.pt"))


if __name__ == "__main__":
  PPOTrajectoryRunner(get_args()).run()
