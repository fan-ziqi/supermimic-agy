"""Reproduce Figure 5(a): Rollout throughput (steps per second) of the learned World Model vs MuJoCo Warp
across batch sizes (parallel environments).
"""
import csv
import os
import time
import torch

from supermimic.env import SimEnv
from supermimic.library import Library
from supermimic.models import WorldModel
from supermimic import state as st

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@torch.no_grad()
def bench_wm(wm, lib, B, steps=100, device="cuda:0"):
  S = torch.randn(B, st.SDIM, device=device)
  a = torch.randn(B, st.NJ, device=device)
  xi = torch.randn(B, st.XI_DIM, device=device)
  idx = torch.randint(0, lib.F - 100, (B,), device=device)
  qlim = torch.stack([-torch.ones(st.NJ, device=device), torch.ones(st.NJ, device=device)])
  for _ in range(10):
    S, _, _, _, _ = wm.step(S, a, xi, lib.ref(idx), qlim)
  torch.cuda.synchronize()
  t0 = time.time()
  for _ in range(steps):
    idx = (idx + 1).clamp(max=lib.F - 1)
    S, _, _, _, _ = wm.step(S, a, xi, lib.ref(idx), qlim)
  torch.cuda.synchronize()
  dt = time.time() - t0
  return (B * steps) / dt


@torch.no_grad()
def bench_sim(lib, B, steps=50, device="cuda:0"):
  env = SimEnv(lib, B, device, dr_on=False, reset_noise=False)
  a = torch.zeros(B, st.NJ, device=device)
  for _ in range(5):
    env.step(a)
  torch.cuda.synchronize()
  t0 = time.time()
  for _ in range(steps):
    env.step(a)
  torch.cuda.synchronize()
  dt = time.time() - t0
  del env
  torch.cuda.empty_cache()
  return (B * steps) / dt


def main():
  dev = "cuda:0"
  lib = Library("data/lib/lafan1_g1.npz", dev)
  wm = WorldModel().to(dev).eval()
  os.makedirs("runs/benchmarks", exist_ok=True)
  out_path = "runs/benchmarks/throughput_fig5a.csv"
  with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["batch_size", "wm_sps", "sim_sps", "speedup"])
    for B in [512, 1024, 2048, 4096, 8192, 16384]:
      wm_sps = bench_wm(wm, lib, B)
      sim_sps = bench_sim(lib, B)
      ratio = wm_sps / sim_sps
      print(f"B={B:6d} | World Model: {wm_sps:10.1f} step/s | MuJoCo Warp: {sim_sps:9.1f} step/s | Speedup: {ratio:5.1f}x", flush=True)
      w.writerow([B, f"{wm_sps:.1f}", f"{sim_sps:.1f}", f"{ratio:.2f}"])


if __name__ == "__main__":
  main()
