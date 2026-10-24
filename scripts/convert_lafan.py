"""Convert LAFAN1 G1-retargeted CSV clips into one 50 Hz motion library (npz).

Each clip is replayed kinematically (frame by frame, all frames as parallel worlds) in the
mjlab G1 scene so that body poses/velocities come from MuJoCo forward kinematics.
Output arrays are concatenated across clips; `clip_start`/`clip_end` index the frames.
"""
import glob
import os
import sys

import numpy as np
import torch

from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader  # CSV->50Hz interpolation + velocities
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

JOINTS = [
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
  "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
  "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]


def replay(csv, device="cuda:0"):
  m = MotionLoader(csv, 30.0, 50.0, device)
  n = m.output_frames
  scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  cfg = SimulationCfg()
  cfg.mujoco.timestep = 1.0 / 50.0
  sim = Simulation(num_envs=n, cfg=cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  robot = scene["robot"]
  idx = robot.find_joints(JOINTS, preserve_order=True)[0]
  root = robot.data.default_root_state.clone()
  root[:, :3] = m.motion_base_poss
  root[:, 3:7] = m.motion_base_rots
  root[:, 7:10] = m.motion_base_lin_vels
  root[:, 10:] = m.motion_base_ang_vels
  robot.write_root_state_to_sim(root)
  jp = robot.data.default_joint_pos.clone()
  jv = robot.data.default_joint_vel.clone()
  jp[:, idx] = m.motion_dof_poss
  jv[:, idx] = m.motion_dof_vels
  robot.write_joint_state_to_sim(jp, jv)
  sim.forward()
  scene.update(cfg.mujoco.timestep)
  d = robot.data
  out = dict(
    joint_pos=d.joint_pos[:, idx], joint_vel=d.joint_vel[:, idx],
    body_pos_w=d.body_link_pos_w, body_quat_w=d.body_link_quat_w,
    body_lin_vel_w=d.body_link_lin_vel_w, body_ang_vel_w=d.body_link_ang_vel_w,
  )
  return {k: v.cpu().numpy() for k, v in out.items()}, list(robot.body_names)


def main(csv_dir, out):
  files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
  lib, starts, ends, names, t = {}, [], [], [], 0
  for f in files:
    d, body_names = replay(f)
    n = d["joint_pos"].shape[0]
    for k, v in d.items():
      lib.setdefault(k, []).append(v)
    starts.append(t); ends.append(t + n); t += n
    names.append(os.path.basename(f)[:-4])
    print(f"{names[-1]:30s} frames={n:6d} ({n/50:.1f}s)", flush=True)
  lib = {k: np.concatenate(v, 0).astype(np.float32) for k, v in lib.items()}
  np.savez(out, clip_start=np.array(starts), clip_end=np.array(ends),
           clip_names=np.array(names), body_names=np.array(body_names), **lib)
  print(f"saved {out}: {len(files)} clips, {t} frames = {t/50/3600:.2f} h")


if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2])
