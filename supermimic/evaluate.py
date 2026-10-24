"""Evaluation protocol: every clip plays once from its first frame, foot friction 1.0, no domain
randomisation / observation noise / action noise, deterministic policy. A motion fails when the rollout
triggers the training termination. Normalised return = undiscounted reward sum / its maximum."""
import torch

from . import state as st
from .env import SimEnv
from .features import HIST, OBS_DIM, goal_feat, proprio_obs
from .library import Library


class Evaluator:
  def __init__(self, lib: Library, device="cuda:0"):
    self.lib, self.dev = lib, device
    self.env = SimEnv(lib, lib.n_clips, device, dr_on=False, reset_noise=False, fixed_friction=1.0)
    self.env.frames_fixed = lib.clip_start.clone()

  @torch.no_grad()
  def run(self, policy, max_steps=None):
    lib, env = self.lib, self.env
    N = lib.n_clips
    env.frames_fixed = lib.clip_start.clone()
    env.reset_all()
    q0 = env.q0
    S, xi = env.extract(), env.xi()
    o = proprio_obs(S, xi, q0)
    hist = o[:, None].repeat(1, HIST, 1)
    ret = torch.zeros(N, device=self.dev)
    steps = torch.zeros(N, device=self.dev)
    failed = torch.zeros(N, dtype=torch.bool, device=self.dev)
    done = torch.zeros(N, dtype=torch.bool, device=self.dev)
    T = int(lib.clip_len.max().item()) - 1 if max_steps is None else max_steps
    T_clip = (lib.clip_len - 1).float() if max_steps is None else torch.minimum(
        lib.clip_len - 1, torch.full_like(lib.clip_len, max_steps)).float()
    for _ in range(T):
      g = goal_feat(S, env.ref_idx, lib)
      a = policy(hist, g)
      S, r, term, trunc, _ = env.step(a)
      alive = ~done
      ret += torch.where(alive, r, torch.zeros_like(r))
      steps += alive.float()
      failed |= alive & term
      done |= term | trunc
      if done.all():
        break
      hist = torch.cat([hist[:, 1:], proprio_obs(S, xi, q0)[:, None]], 1)
    step_norm = ret / (steps.clamp(min=1) * st.DT * st.R_POS_MAX)
    ep_norm = ret / (T_clip.clamp(min=1) * st.DT * st.R_POS_MAX)
    return dict(norm_return=ep_norm.mean().item(), step_return=step_norm.mean().item(),
                n_failed=int(failed.sum().item()),
                per_clip_return=ep_norm.cpu(), per_clip_step_return=step_norm.cpu(),
                per_clip_failed=failed.cpu(), steps=steps.cpu())


if __name__ == "__main__":
  import argparse
  import json
  from .models import Policy

  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", default="runs/supermimic_lafan_1gpu/ckpt.pt")
  ap.add_argument("--lib", default="data/lib/lafan1_g1.npz")
  ap.add_argument("--max_steps", type=int, default=None)
  ap.add_argument("--out", default="runs/supermimic_lafan_1gpu/full_eval.json")
  args = ap.parse_args()

  dev = "cuda:0"
  lib = Library(args.lib, dev)
  ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
  pol = Policy().to(dev)
  pol.load_state_dict(ck["pol"])
  pol.eval()

  ev = Evaluator(lib, dev)
  res = ev.run(pol, max_steps=args.max_steps)
  clips = []
  for i, name in enumerate(lib.clip_names):
    clips.append({
        "clip": str(name),
        "len_frames": int(lib.clip_len[i].item()),
        "survived_steps": int(res["steps"][i].item()),
        "failed": bool(res["per_clip_failed"][i].item()),
        "norm_return": float(res["per_clip_return"][i].item()),
    })
  out_data = {
      "iter": int(ck.get("it", -1)),
      "max_steps": args.max_steps,
      "norm_return": res["norm_return"],
      "n_failed": res["n_failed"],
      "n_clips": len(clips),
      "success_rate": 1.0 - res["n_failed"] / len(clips),
      "clips": clips,
  }
  with open(args.out, "w") as f:
    json.dump(out_data, f, indent=2)
  print(f"[FullEval] iter={out_data['iter']} norm_return={out_data['norm_return']:.4f} "
        f"failed={out_data['n_failed']}/{out_data['n_clips']} "
        f"success_rate={out_data['success_rate']*100:.1f}%")

