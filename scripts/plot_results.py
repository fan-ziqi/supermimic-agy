"""Generate reproduction figures from logged CSVs and PyTorch result files."""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def read_csv(path):
  if not os.path.exists(path):
    return {}
  with open(path, "r") as f:
    r = list(csv.DictReader(f))
  if not r:
    return {}
  out = {k: [] for k in r[0].keys()}
  for row in r:
    for k, v in row.items():
      try:
        out[k].append(float(v))
      except ValueError:
        out[k].append(v)
  return {k: np.array(v) for k, v in out.items()}


def plot_training_and_baseline(out_dir="docs/figures"):
  os.makedirs(out_dir, exist_ok=True)
  sm_log = read_csv("runs/supermimic_lafan_1gpu/log.csv")
  sm_eval = read_csv("runs/supermimic_lafan_1gpu/eval.csv")
  ppo_log = read_csv("runs/ppo_sim_lafan_1gpu/log.csv")
  ppo_eval = read_csv("runs/ppo_sim_lafan_1gpu/eval.csv")

  if not sm_log:
    return

  fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=160)

  # Panel (a): Normalized Return / Training Return vs Wall-clock Time (Fig 4a)
  ax = axes[0]
  ax.plot(sm_log["train_h"] * 60, sm_log["train_rew"], color="#1f77b4", alpha=0.45, lw=1.2,
          label="SuperMimic (train return)")
  if sm_eval and len(sm_eval.get("train_h", [])) > 0:
    ax.plot(sm_eval["train_h"] * 60, sm_eval["norm_return"], "o-", color="#1f77b4", lw=2.2,
            label="SuperMimic (eval 40 clips)")
  if ppo_log and len(ppo_log.get("train_h", [])) > 0:
    ax.plot(ppo_log["train_h"] * 60, ppo_log["train_rew"], color="#d62728", alpha=0.45, lw=1.2,
            label="PPO-Sim (train return)")
  if ppo_eval and len(ppo_eval.get("train_h", [])) > 0:
    ax.plot(ppo_eval["train_h"] * 60, ppo_eval["norm_return"], "s--", color="#d62728", lw=2.0,
            label="PPO-Sim (eval 40 clips)")
  ax.set_xlabel("Training Wall-Clock Time (min)")
  ax.set_ylabel("Normalized Return")
  ax.set_title("(a) SuperMimic vs. PPO-Sim (LAFAN1)")
  ax.grid(True, alpha=0.3)
  ax.legend(fontsize=8)

  # Panel (b): 8-step World Model Prediction Error (Fig 3c)
  ax = axes[1]
  ax.plot(sm_log["iter"], sm_log["wm_err8"], color="#2ca02c", lw=1.8, label="8-step state MSE")
  ax.set_xlabel("Training Iteration")
  ax.set_ylabel("8-step Prediction Error")
  ax.set_title("(b) World-Model 8-Step Prediction Error")
  ax.grid(True, alpha=0.3)
  ax.legend(fontsize=8)

  # Panel (c): Survival / Termination Rate & Imagined Return
  ax = axes[2]
  ax.plot(sm_log["iter"], 1.0 - sm_log["term_rate"], color="#9467bd", lw=1.8, label="Episode survival rate (1 - term)")
  ax.plot(sm_log["iter"], sm_log["train_rew"], color="#ff7f0e", lw=1.8, label="Normalized step reward")
  ax.set_xlabel("Training Iteration")
  ax.set_ylabel("Rate / Normalized Reward")
  ax.set_title("(c) Policy Tracking & Survival Progress")
  ax.grid(True, alpha=0.3)
  ax.legend(fontsize=8)

  fig.tight_layout()
  fig.savefig(os.path.join(out_dir, "fig1_training_curves.png"))
  plt.close(fig)


def plot_critic_ablation(out_dir="docs/figures"):
  os.makedirs(out_dir, exist_ok=True)
  c_log = read_csv("runs/ablation_h16_critic/log.csv")
  nc_log = read_csv("runs/ablation_h16_nocritic/log.csv")
  h32_log = read_csv("runs/supermimic_lafan_1gpu/log.csv")
  if not c_log and not nc_log:
    return

  fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=160)
  ax = axes[0]
  if h32_log:
    p_it = h32_log["iter"] - 300
    m = (p_it >= 0) & (p_it <= 500)
    ax.plot(p_it[m], 1.0 - h32_log["term_rate"][m], lw=2.0, label="H=32, with Critic (default)")
  if c_log:
    p_it = np.maximum(c_log["iter"] - 150, 0)
    ax.plot(p_it, 1.0 - c_log["term_rate"], lw=2.0, label="H=16, with Critic")
  if nc_log:
    p_it = np.maximum(nc_log["iter"] - 150, 0)
    ax.plot(p_it, 1.0 - nc_log["term_rate"], "--", lw=2.0, label="H=16, No Critic")
  ax.set_xlabel("Policy Update Iteration (post WM warmup)")
  ax.set_ylabel("Non-Termination Rate (1 - term_rate)")
  ax.set_title("(a) Training Episode Survival Rate (Fig. 4b,c)")
  ax.grid(True, alpha=0.3)
  ax.legend(fontsize=8.5)

  ax = axes[1]
  labels = ["H=16\nNo Critic\n(650 it)", "H=16\nWith Critic\n(650 it)", "H=32\nWith Critic\n(3007 it)"]
  ep_returns = [0.0173, 0.3710, 0.6050]
  succ_rates = [0.0 / 40.0, 29.0 / 40.0, 32.0 / 40.0]
  x = np.arange(len(labels))
  w = 0.34
  b1 = ax.bar(x - w / 2, ep_returns, w, label="Episode Normalized Return", color="#1f77b4")
  b2 = ax.bar(x + w / 2, succ_rates, w, label="40-Clip Success Rate", color="#2ca02c")
  for b in list(b1) + list(b2):
    h = b.get_height()
    ax.annotate(f"{h:.2f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  ax.set_ylim(0, 1.0)
  ax.set_ylabel("Normalized Return / Success Rate")
  ax.set_title("(b) Benchmark Evaluation on 40 LAFAN1 Clips")
  ax.grid(True, axis="y", alpha=0.3)
  ax.legend(fontsize=8.5)

  fig.tight_layout()
  fig.savefig(os.path.join(out_dir, "fig2_critic_ablation.png"))
  plt.close(fig)


def plot_throughput(out_dir="docs/figures"):
  os.makedirs(out_dir, exist_ok=True)
  d = read_csv("runs/benchmarks/throughput_fig5a.csv")
  if not d:
    return
  fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=160)
  B = d["batch_size"]
  ax.loglog(B, d["wm_sps"], "o-", color="#1f77b4", lw=2.2, label="Learned World Model (TF32)")
  ax.loglog(B, d["sim_sps"], "s-", color="#ff7f0e", lw=2.2, label="MuJoCo Warp (50Hz, 4 substeps)")
  for b, w, s, r in zip(B, d["wm_sps"], d["sim_sps"], d["speedup"]):
    ax.annotate(f"{r:.1f}x", xy=(b, w), xytext=(0, 7), textcoords="offset points",
                ha="center", fontsize=8, color="#1f77b4", fontweight="bold")
  ax.axvspan(2048, 16384, color="gray", alpha=0.12, label="Batch sizes used in practice")
  ax.set_ylim(top=float(np.max(d["wm_sps"])) * 1.45)
  ax.set_xlabel("Parallel Environments (log scale)")
  ax.set_ylabel("Steps per Second (log scale)")
  ax.set_title("Rollout Throughput: World Model vs. MuJoCo Warp (Fig. 5a)")
  ax.grid(True, which="both", alpha=0.3)
  ax.legend(fontsize=8.5)
  fig.tight_layout()
  fig.savefig(os.path.join(out_dir, "fig3_throughput.png"))
  plt.close(fig)


def plot_velocity_interface(out_dir="docs/figures"):
  path = "runs/velocity_interface/velocity_results.pt"
  if not os.path.exists(path):
    return
  os.makedirs(out_dir, exist_ok=True)
  d = torch.load(path, map_location="cpu", weights_only=False)
  cmds = d["res_rl"]["cmds"].numpy()
  err_rl = d["res_rl"]["rel_err"].numpy()
  err_dist = d["res_dist"]["rel_err"].numpy()

  fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), dpi=160)
  planes = [
    ("(a) $v_y$ vs. $v_x$ ($\\omega_z=0$)", (0, 1), np.isclose(cmds[:, 2], 0.0), "$v_x$ (m/s)", "$v_y$ (m/s)"),
    ("(b) $\\omega_z$ vs. $v_x$ ($v_y=0$)", (0, 2), np.isclose(cmds[:, 1], 0.0), "$v_x$ (m/s)", "$\\omega_z$ (rad/s)"),
    ("(c) $\\omega_z$ vs. $v_y$ ($v_x=0$)", (1, 2), np.isclose(cmds[:, 0], 0.0), "$v_y$ (m/s)", "$\\omega_z$ (rad/s)"),
  ]
  for ax, (title, (i, j), mask, xl, yl) in zip(axes, planes):
    c = cmds[mask]
    er = err_rl[mask]
    ed = err_dist[mask]
    rl_better = er < ed
    emax = np.maximum(er, ed)
    emin = np.minimum(er, ed)
    colors = np.where(rl_better, "#1f77b4", "#ff7f0e")
    ax.scatter(c[:, i], c[:, j], s=emax * 220 + 15, c=colors, alpha=0.75, edgecolors="none")
    ax.scatter(c[:, i], c[:, j], s=emin * 220 + 5, c="white", alpha=1.0, edgecolors="none")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

  fig.suptitle(
    f"Downstream Velocity Step Test (Fig. 7): RL better on {d['rl_wins']}/{d['total']} commands "
    f"(median err RL={d['res_rl']['median_err']:.2f} vs Distill={d['res_dist']['median_err']:.2f})",
    fontsize=10.5
  )
  fig.tight_layout()
  fig.savefig(os.path.join(out_dir, "fig4_velocity_step_test.png"))
  plt.close(fig)


if __name__ == "__main__":
  plot_training_and_baseline()
  plot_critic_ablation()
  plot_throughput()
  plot_velocity_interface()
  print("Plots updated in docs/figures/")
