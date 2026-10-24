"""mjlab-backed G1 simulator used for data collection / evaluation.

The simulator is a standard (non-differentiable) MuJoCo-Warp environment built from mjlab's G1 tracking
task. mjlab's own reward / termination / observation logic is stripped; rewards and terminations are the
analytic functions in `state.py` evaluated on the logged simulator state, so that simulator and world model
use *identical* reward code.
"""
from dataclasses import dataclass

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from .geom import quat_to_mat
from .library import BODY_NAMES, Library
from . import state as st

DUMMY = "/tmp/supermimic_dummy_motion.npz"


def _write_dummy():
  n = 30
  np.savez(DUMMY, joint_pos=np.zeros((2, 29)), joint_vel=np.zeros((2, 29)),
           body_pos_w=np.zeros((2, n, 3)), body_quat_w=np.tile([1, 0, 0, 0.], (2, n, 1)),
           body_lin_vel_w=np.zeros((2, n, 3)), body_ang_vel_w=np.zeros((2, n, 3)))


class LibCommand(MotionCommand):
  """Motion command over a multi-clip library: resets and reference stepping are driven by the wrapper."""

  def _resample_command(self, env_ids):
    w = self.wrapper
    lib = w.lib
    f = w.pick_start(env_ids)
    self.time_steps[env_ids] = f
    c = self.cfg
    n = len(env_ids)
    dev = self.device
    root_pos = lib.pos[f, 0] + self._env.scene.env_origins[env_ids]
    root_ori = lib.quat_root[f].clone()
    root_lv, root_av = lib.lv[f, 0].clone(), lib.av[f, 0].clone()
    if w.reset_noise:
      pr = torch.tensor([c.pose_range.get(k, (0., 0.)) for k in ("x", "y", "z", "roll", "pitch", "yaw")], device=dev)
      s = sample_uniform(pr[:, 0], pr[:, 1], (n, 6), device=dev)
      root_pos += s[:, :3]
      root_ori = quat_mul(quat_from_euler_xyz(s[:, 3], s[:, 4], s[:, 5]), root_ori)
      vr = torch.tensor([c.velocity_range.get(k, (0., 0.)) for k in ("x", "y", "z", "roll", "pitch", "yaw")], device=dev)
      s = sample_uniform(vr[:, 0], vr[:, 1], (n, 6), device=dev)
      root_lv += s[:, :3]
      root_av += s[:, 3:]
      jp = lib.jp[f] + sample_uniform(c.joint_position_range[0], c.joint_position_range[1], (n, 29), device=dev)
    else:
      jp = lib.jp[f].clone()
    self._write_reference_state_to_sim(env_ids, root_pos, root_ori, root_lv, root_av, jp, lib.jv[f])

  def _update_command(self):
    if self.wrapper.advance:
      lib = self.wrapper.lib
      last = lib.last_frame[self.time_steps.clamp(0, lib.F - 1)]
      self.time_steps = torch.minimum(self.time_steps + 1, last)

  def _update_metrics(self):
    pass


@dataclass(kw_only=True)
class LibCommandCfg(MotionCommandCfg):
  def build(self, env):
    return LibCommand(self, env)


class SimEnv:
  """Vectorised G1 simulator exposing the SuperMimic state vector."""

  def __init__(self, lib: Library, num_envs: int, device="cuda:0", dr_on=True, reset_noise=True,
               start_mode="library", fixed_friction=None):
    _write_dummy()
    self.lib, self.N, self.device = lib, num_envs, device
    self.reset_noise, self.advance, self.start_mode = reset_noise, True, start_mode
    cfg = unitree_g1_flat_tracking_env_cfg()
    cfg.scene.num_envs = num_envs
    cfg.auto_reset = False
    cfg.episode_length_s = 1e9
    old = cfg.commands["motion"]
    cfg.commands["motion"] = LibCommandCfg(
      entity_name=old.entity_name, resampling_time_range=(1e9, 1e9), debug_vis=False,
      pose_range=old.pose_range, velocity_range=old.velocity_range,
      joint_position_range=old.joint_position_range, motion_file=DUMMY,
      anchor_body_name="pelvis", body_names=old.body_names)
    # Strip mjlab's own rewards / terminations / observations.
    cfg.rewards = {}
    cfg.terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}
    cfg.observations = {"actor": ObservationGroupCfg(
      terms={"joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel)}, concatenate_terms=True)}
    cfg.scene.sensors = ()
    # Domain randomisation (Table A.3): per-episode (reset-mode); no root-velocity pushes.
    ev = {}
    if dr_on:
      ev = {
        "base_com": EventTermCfg(mode="reset", func=dr.body_com_offset, params={
          "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)), "operation": "add",
          "ranges": {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)}}),
        "encoder_bias": EventTermCfg(mode="reset", func=dr.encoder_bias, params={
          "asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)}),
        "foot_friction": EventTermCfg(mode="reset", func=dr.geom_friction, params={
          "asset_cfg": SceneEntityCfg("robot", geom_names=r"^(left|right)_foot[1-7]_collision$"),
          "operation": "abs", "ranges": (0.3, 1.2), "shared_random": True}),
      }
    elif fixed_friction is not None:
      ev = {"foot_friction": EventTermCfg(mode="reset", func=dr.geom_friction, params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=r"^(left|right)_foot[1-7]_collision$"),
        "operation": "abs", "ranges": (fixed_friction, fixed_friction), "shared_random": True})}
    cfg.events = ev
    self.cfg = cfg
    self.env = ManagerBasedRlEnv(cfg, device=device)
    self.cmd: LibCommand = self.env.command_manager.get_term("motion")
    self.cmd.wrapper = self
    self.robot = self.env.scene["robot"]
    self.bidx = torch.tensor(self.robot.find_bodies(list(BODY_NAMES), preserve_order=True)[0],
                             dtype=torch.long, device=device)
    assert tuple(np.array(self.robot.body_names)[self.bidx.cpu().numpy()]) == BODY_NAMES
    self.q0 = self.robot.data.default_joint_pos[0].clone()
    self.qlim = self.robot.data.soft_joint_pos_limits[0].T.contiguous()  # [2,29]
    self.origins = self.env.scene.env_origins
    self.torso_id = self.robot.body_names.index("torso_link")
    self.foot_geom = None
    self.a_prev = torch.zeros(num_envs, 29, device=device)
    self.ep_step = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.start_fn = self._uniform_start  # replaced by trainer (return-based sampler)
    self.frames_fixed = None
    self.reset_all()

  # ---- resets -------------------------------------------------------------------------------
  def pick_start(self, env_ids):
    if self.frames_fixed is not None:
      return self.frames_fixed[env_ids]
    return self.start_fn(env_ids)

  def _uniform_start(self, env_ids):
    f = torch.randint(0, self.lib.F, (len(env_ids),), device=self.device)
    return torch.minimum(f, self.lib.last_frame[f] - 1).clamp(min=0)

  def reset_ids(self, env_ids):
    self.advance = False
    self.env.reset(env_ids=env_ids)
    self.advance = True
    self.a_prev[env_ids] = 0.0
    self.ep_step[env_ids] = 0

  def reset_all(self):
    self.reset_ids(torch.arange(self.N, device=self.device))

  # ---- state --------------------------------------------------------------------------------
  @property
  def ref_idx(self):
    return self.cmd.time_steps

  def xi(self):
    d = self.robot.data
    m = self.env.sim.model
    N = self.N
    if self.foot_geom is None:
      gid = self.robot.indexing.geom_ids if hasattr(self.robot.indexing, "geom_ids") else None
      self.foot_geom = self._find_foot_geom()
    fr = m.geom_friction[:, self.foot_geom, 0] if m.geom_friction.shape[0] == N else \
      m.geom_friction[0:1, self.foot_geom, 0].expand(N)
    com = m.body_ipos[:, self.torso_id] if m.body_ipos.shape[0] == N else m.body_ipos[0:1, self.torso_id].expand(N, 3)
    com = com - self.com0
    return torch.cat([fr.reshape(N, 1), d.encoder_bias, com], -1)

  def _find_foot_geom(self):
    import mujoco
    mj = self.env.sim.mj_model
    for g in range(mj.ngeom):
      nm = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, g)
      if nm and "left_foot1_collision" in nm:
        self.com0 = torch.tensor(mj.body_ipos[self.torso_id], dtype=torch.float32, device=self.device)
        return g
    raise RuntimeError("foot geom not found")

  def extract(self):
    d = self.robot.data
    b = self.bidx
    pos = d.body_link_pos_w[:, b] - self.origins[:, None, :]
    R = quat_to_mat(d.body_link_quat_w[:, b])
    return st.pack_state(pos, R, d.body_link_lin_vel_w[:, b], d.body_link_ang_vel_w[:, b],
                         d.joint_pos, d.joint_vel, self.a_prev)

  def step(self, a_exec):
    """Execute a_exec; returns (S', reward, terminated, truncated)."""
    self.env.step(a_exec)
    a_prev = self.a_prev.clone()
    self.a_prev[:] = a_exec
    self.ep_step += 1
    S = self.extract()
    idx = self.ref_idx.clamp(0, self.lib.F - 1)
    r, e_z, e_tilt = st.reward_terms(S, a_prev, a_exec, self.lib.ref(self.lib.clamp(idx)), self.qlim)
    term = (e_z > st.TERM_Z) | (e_tilt > st.TERM_TILT)
    trunc = (idx >= self.lib.last_frame[idx]) & ~term
    return S, r, term, trunc, a_prev
