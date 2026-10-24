"""Fixed-size replay buffer of simulator transitions (new rows replace the oldest).

Row t, env n holds the state s_t reached *before* acting, the executed (noisy) action a_t, the noisy
observation o_t, the sim context xi, the aligned reference frame, the in-episode step counter, whether the
state is terminal (`term`) and whether it is the last row of its episode (`final`, no action follows).
"""
import torch

from . import state as st
from .features import HIST, OBS_DIM


class Replay:
  def __init__(self, T, N, device):
    self.T, self.N, self.dev = T, N, device
    self.S = torch.zeros(T, N, st.SDIM, device=device)
    self.a = torch.zeros(T, N, st.NJ, device=device, dtype=torch.float16)
    self.o = torch.zeros(T, N, OBS_DIM, device=device, dtype=torch.float16)
    self.xi = torch.zeros(T, N, st.XI_DIM, device=device, dtype=torch.float16)
    self.idx = torch.zeros(T, N, device=device, dtype=torch.int32)
    self.ep = torch.zeros(T, N, device=device, dtype=torch.int32)
    self.term = torch.zeros(T, N, device=device, dtype=torch.bool)
    self.final = torch.ones(T, N, device=device, dtype=torch.bool)
    self.ptr, self.filled = 0, 0

  def push(self, S, a, o, xi, idx, ep, term, final):
    p = self.ptr
    self.S[p], self.a[p], self.o[p], self.xi[p] = S, a.half(), o.half(), xi.half()
    self.idx[p], self.ep[p], self.term[p], self.final[p] = idx.int(), ep.int(), term, final
    self.ptr = (p + 1) % self.T
    self.filled = min(self.filled + 1, self.T)

  def _row(self, off):
    start = self.ptr if self.filled == self.T else 0
    return (start + off) % self.T

  def sample_windows(self, B, k):
    """B valid k-step windows: S [B,k+1,297], a [B,k,29], xi [B,33], idx [B,k+1], term [B,k]."""
    dev = self.dev
    out = None
    got = []
    need = B
    for _ in range(6):
      m = int(need * 2.0) + 16
      off = torch.randint(0, self.filled - k, (m,), device=dev)
      n = torch.randint(0, self.N, (m,), device=dev)
      rows = self._row(off[:, None] + torch.arange(k + 1, device=dev))  # [m,k+1]
      ep = self.ep[rows, n[:, None]]
      fin = self.final[rows[:, :k], n[:, None]]
      ok = (~fin).all(1) & ((ep[:, 1:] - ep[:, :-1]) == 1).all(1)
      got.append((rows[ok], n[ok]))
      need = B - sum(g[0].shape[0] for g in got)
      if need <= 0:
        break
    rows = torch.cat([g[0] for g in got])[:B]
    n = torch.cat([g[1] for g in got])[:B]
    nn_ = n[:, None]
    return (self.S[rows, nn_], self.a[rows[:, :k], nn_].float(), self.xi[rows[:, 0], n].float(),
            self.idx[rows, nn_].long(), self.term[rows[:, 1:], nn_])

  def sample_seeds(self, B):
    """Valid start rows with their previous HIST-1 noisy observations."""
    dev = self.dev
    m = B * 4
    off = torch.randint(0, self.filled, (m,), device=dev)
    n = torch.randint(0, self.N, (m,), device=dev)
    rows = self._row(off)
    ok = ~self.final[rows, n]
    off, rows, n = off[ok][:B], rows[ok][:B], n[ok][:B]
    ep = self.ep[rows, n].long()
    k = torch.arange(HIST - 1, -1, -1, device=dev)  # oldest -> newest
    back = torch.minimum(torch.minimum(k[None, :], ep[:, None]), off[:, None])
    hrows = (rows[:, None] - back) % self.T
    hist = self.o[hrows, n[:, None]].float()
    return self.S[rows, n], hist, self.xi[rows, n].float(), self.idx[rows, n].long()
