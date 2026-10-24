"""Reference motion library (tracked bodies only) with clip bounds and 1-second bins."""
import numpy as np
import torch

from .geom import quat_to_mat

BODY_NAMES = (
  "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
  "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
  "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
  "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)
K_FUT = 10
BIN_FRAMES = 50  # 1 s at 50 Hz


class Library:
  def __init__(self, path, device="cuda:0", clips=None):
    d = np.load(path)
    names = list(d["body_names"])
    sel = [names.index(n) for n in BODY_NAMES]
    cs, ce = d["clip_start"], d["clip_end"]
    keep = np.arange(len(cs)) if clips is None else np.asarray(clips)
    fr = np.concatenate([np.arange(cs[i], ce[i]) for i in keep])
    lens = np.array([ce[i] - cs[i] for i in keep])
    self.clip_names = [str(d["clip_names"][i]) for i in keep]
    t = lambda x: torch.tensor(x[fr], dtype=torch.float32, device=device)
    self.jp, self.jv = t(d["joint_pos"]), t(d["joint_vel"])
    self.pos = t(d["body_pos_w"][:, sel])
    self.R = quat_to_mat(t(d["body_quat_w"][:, sel]))
    self.lv, self.av = t(d["body_lin_vel_w"][:, sel]), t(d["body_ang_vel_w"][:, sel])
    self.quat_root = t(d["body_quat_w"][:, sel[0]])
    self.F = len(fr)
    self.n_clips = len(keep)
    self.clip_start = torch.tensor(np.concatenate([[0], np.cumsum(lens)[:-1]]), device=device)
    self.clip_len = torch.tensor(lens, device=device)
    self.clip_end = self.clip_start + self.clip_len  # exclusive
    cid = np.repeat(np.arange(len(keep)), lens)
    self.clip_id = torch.tensor(cid, device=device)
    self.last_frame = self.clip_end[self.clip_id] - 1  # per frame: last valid frame of its clip
    # 1 s bins within each clip
    nb = (lens + BIN_FRAMES - 1) // BIN_FRAMES
    self.bin_clip = torch.tensor(np.repeat(np.arange(len(keep)), nb), device=device)
    self.bin_off = torch.tensor(np.concatenate([np.arange(n) * BIN_FRAMES for n in nb]), device=device)
    self.n_bins = int(nb.sum())
    self.bin_first = self.clip_start[self.bin_clip] + self.bin_off
    self.bin_nframes = torch.minimum(torch.full_like(self.bin_off, BIN_FRAMES),
                                     self.clip_len[self.bin_clip] - self.bin_off)
    self.frame_bin = torch.zeros(self.F, dtype=torch.long, device=device)
    fb = np.concatenate([np.arange(n).repeat(BIN_FRAMES)[:l] for n, l in zip(nb, lens)])
    bin_base = np.concatenate([[0], np.cumsum(nb)[:-1]])
    self.frame_bin = torch.tensor(fb + np.repeat(bin_base, lens), device=device)
    self.device = device

  def ref(self, idx):
    """Reference tensors for frame indices idx [...]."""
    idx = idx.clamp(0, self.F - 1)
    return dict(pos=self.pos[idx], R=self.R[idx], lv=self.lv[idx], av=self.av[idx])

  def clamp(self, idx):
    idx = idx.clamp(0, self.F - 1)
    return torch.minimum(idx, self.last_frame[idx])

  def future(self, idx, K=K_FUT):
    """Frame indices idx+1..idx+K (clamped to clip end): [..., K]."""
    off = torch.arange(1, K + 1, device=idx.device)
    idx = idx.clamp(0, self.F - 1)
    f = idx[..., None] + off
    return torch.minimum(f, self.last_frame[idx][..., None]).clamp(0, self.F - 1)
