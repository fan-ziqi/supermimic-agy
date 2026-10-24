"""Return-based reset sampling (Appendix D.1)."""
import torch

from . import state as st
from .library import Library

GAMMA = 0.99


class ResetSampler:
  """Resets pick 1 s bins of the motion library: 50% uniform, 40% softmax(-25 G_b/Gmax), 10% proportional to
  each bin's past (slowly averaged) softmax weight.  G_b is a running estimate of the discounted return from
  bin b, measured on the policy's own rollouts over the first `cap` steps after each reset (normalised by the
  maximum attainable value over the same steps, so G_b/Gmax lies in [0, 1])."""

  def __init__(self, lib: Library, n_envs, cap=64, alpha=0.1, temp=25.0, past_rate=1e-3):
    self.lib, self.cap, self.alpha, self.temp, self.past_rate = lib, cap, alpha, temp, past_rate
    dev = lib.device
    self.G = torch.full((lib.n_bins,), 0.5, device=dev)
    self.past_w = torch.full((lib.n_bins,), 1.0 / lib.n_bins, device=dev)
    self.m_bin = torch.full((n_envs,), -1, dtype=torch.long, device=dev)
    self.m_G = torch.zeros(n_envs, device=dev)
    self.m_k = torch.zeros(n_envs, dtype=torch.long, device=dev)
    self.m_n = torch.ones(n_envs, dtype=torch.long, device=dev)
    disc = GAMMA ** torch.arange(cap, device=dev)
    self.denom = st.R_POS_MAX * st.DT * torch.cumsum(disc, 0)  # attainable return over first n steps
    self.disc = disc

  def __call__(self, env_ids):
    lib, n = self.lib, len(env_ids)
    w = torch.softmax(-self.temp * self.G, 0)
    self.past_w.mul_(1 - self.past_rate).add_(self.past_rate * w)
    u = torch.rand(n, device=lib.device)
    b_uni = torch.randint(0, lib.n_bins, (n,), device=lib.device)
    b_soft = torch.multinomial(w, n, replacement=True)
    b_past = torch.multinomial(self.past_w / self.past_w.sum(), n, replacement=True)
    b = torch.where(u < 0.5, b_uni, torch.where(u < 0.9, b_soft, b_past))
    f = lib.bin_first[b] + (torch.rand(n, device=lib.device) * lib.bin_nframes[b]).long()
    f = torch.minimum(f, lib.last_frame[f] - 1).clamp(min=0)
    self.m_bin[env_ids] = b
    self.m_G[env_ids] = 0.0
    self.m_k[env_ids] = 0
    self.m_n[env_ids] = torch.clamp(lib.last_frame[f] - f, min=1, max=self.cap)
    return f

  @torch.no_grad()
  def observe(self, r, term, trunc):
    """Accumulate per-step reward for the measurement window of every env."""
    act = self.m_bin >= 0
    kk = self.m_k.clamp(max=self.cap - 1)
    self.m_G += torch.where(act, self.disc[kk] * r, torch.zeros_like(r))
    self.m_k += act.long()
    fin = act & (term | trunc | (self.m_k >= self.m_n))
    if fin.any():
      ids = fin.nonzero().squeeze(-1)
      n_eff = torch.where(term[ids], torch.full_like(self.m_n[ids], self.cap), self.m_n[ids])
      gn = (self.m_G[ids] / self.denom[(n_eff - 1).clamp(0, self.cap - 1)]).clamp(0, 1)
      b = self.m_bin[ids]
      sums = torch.zeros_like(self.G).index_add_(0, b, gn)
      cnt = torch.zeros_like(self.G).index_add_(0, b, torch.ones_like(gn))
      m = cnt > 0
      self.G[m] += self.alpha * (sums[m] / cnt[m] - self.G[m])
      self.m_bin[ids] = -1
