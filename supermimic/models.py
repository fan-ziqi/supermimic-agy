"""Networks: differentiable world model, goal-conditioned policy and critic."""
import math

import torch
import torch.nn as nn

from . import state as st
from .features import GOAL_DIM, HIST, K_FUT, OBS_DIM, norm_xi
from .geom import yaw_of


class RunningNorm(nn.Module):
  def __init__(self, dim, clip=10.0, max_count=1e6):
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.tensor(1e-4))
    self.clip, self.max_count = clip, max_count

  @torch.no_grad()
  def update(self, x):
    x = x.reshape(-1, x.shape[-1]).float()
    n = x.shape[0]
    bm, bv = x.mean(0), x.var(0, unbiased=False)
    tot = self.count + n
    d = bm - self.mean
    self.mean += d * n / tot
    self.var.copy_((self.var * self.count + bv * n + d * d * self.count * n / tot) / tot)
    self.count.copy_(torch.clamp(tot, max=self.max_count))

  def forward(self, x):
    return ((x - self.mean) / torch.sqrt(self.var + 1e-4)).clamp(-self.clip, self.clip)


def mlp(i, hidden, o, act=nn.ELU, last_scale=None):
  layers, d = [], i
  for h in hidden:
    layers += [nn.Linear(d, h), act()]
    d = h
  layers.append(nn.Linear(d, o))
  net = nn.Sequential(*layers)
  if last_scale is not None:
    with torch.no_grad():
      net[-1].weight.mul_(last_scale)
      net[-1].bias.zero_()
  return net


class WorldModel(nn.Module):
  """s_hat' = f(s_hat, a, xi) (learned, in a heading-canonical frame); analytic reward head; learned
  termination head d_hat = sigmoid(MLP(e_z, e_tilt))."""

  def __init__(self, hidden=2048):
    super().__init__()
    self.dyn = mlp(st.SDIM + st.NJ + st.XI_DIM, [hidden] * 3, st.SDIM, nn.SiLU)
    self.term = mlp(2, [64, 64], 1, nn.SiLU)
    with torch.no_grad():  # start with d_hat ~ 0 below the thresholds, via logit = k*(max(e/thr)-1)
      self.term[-1].bias.fill_(-4.0)
    self.register_buffer("in_mu", torch.zeros(st.SDIM))
    self.register_buffer("in_sd", torch.ones(st.SDIM))
    self.register_buffer("d_sd", torch.ones(st.SDIM))
    self.register_buffer("err_sd", torch.ones(st.SDIM))  # scale used in the fit loss

  def term_prob_logit(self, e_z, e_tilt):
    f = torch.stack([e_z / st.TERM_Z, e_tilt / st.TERM_TILT], -1)
    return self.term(f).squeeze(-1) + 8.0 * (torch.maximum(f[..., 0], f[..., 1]) - 1.0)

  def next_state(self, S, a, xi):
    Sc, rxy, yaw = st.canon(S)
    x = torch.cat([(Sc - self.in_mu) / self.in_sd, a, norm_xi(xi)], -1)
    Sc2 = Sc + self.dyn(x) * self.d_sd
    S2 = st.decanon(Sc2, rxy, yaw)
    return torch.cat([S2[..., :st.O_AP], a], -1)  # previous-action slot is exact

  def step(self, S, a, xi, ref_next, qlim):
    """One imagined step -> (S', reward, d_hat) using the analytic reward and learned termination."""
    S2 = self.next_state(S, a, xi)
    r, e_z, e_tilt = st.reward_terms(S2, S[..., st.O_AP:], a, ref_next, qlim)
    return S2, r, torch.sigmoid(self.term_prob_logit(e_z, e_tilt)), e_z, e_tilt


class Policy(nn.Module):
  def __init__(self):
    super().__init__()
    self.enc = mlp(K_FUT * GOAL_DIM, [2048, 1024, 512, 256], 32)
    self.dec = mlp(32 + HIST * OBS_DIM, [2048, 1024, 512, 256], st.NJ, last_scale=0.01)
    self.obs_norm = RunningNorm(OBS_DIM)
    self.goal_norm = RunningNorm(GOAL_DIM)

  def forward(self, hist, g):
    z = torch.tanh(self.enc(self.goal_norm(g).flatten(-2)))
    return self.dec(torch.cat([z, self.obs_norm(hist).flatten(-2)], -1))


class Critic(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = mlp(st.SDIM + K_FUT * GOAL_DIM + st.XI_DIM, [256, 128], 1)

  def forward(self, Sc_n, g_n, xi):
    return self.net(torch.cat([Sc_n, g_n.flatten(-2), norm_xi(xi)], -1)).squeeze(-1)
