"""Small differentiable rotation utilities (quaternions are wxyz)."""
import torch


def quat_to_mat(q):
  w, x, y, z = q.unbind(-1)
  return torch.stack([
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ], -1).reshape(*q.shape[:-1], 3, 3)


def mat_to_6d(R):
  """First two columns, concatenated: [..., 6]."""
  return torch.cat([R[..., :, 0], R[..., :, 1]], -1)


def sixd_to_mat(d):
  a1, a2 = d[..., :3], d[..., 3:6]
  b1 = torch.nn.functional.normalize(a1, dim=-1)
  a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
  b2 = torch.nn.functional.normalize(a2, dim=-1)
  b3 = torch.cross(b1, b2, dim=-1)
  return torch.stack([b1, b2, b3], -1)


def rot_angle_sq(Ra, Rb):
  """Squared geodesic angle between rotation matrices (smooth at 0)."""
  M = Ra.transpose(-1, -2) @ Rb
  v = torch.stack([M[..., 2, 1] - M[..., 1, 2], M[..., 0, 2] - M[..., 2, 0],
                   M[..., 1, 0] - M[..., 0, 1]], -1) * 0.5
  s = torch.sqrt((v * v).sum(-1) + 1e-8)
  c = (M[..., 0, 0] + M[..., 1, 1] + M[..., 2, 2] - 1.0) * 0.5
  return torch.atan2(s, c) ** 2


def yaw_of(R):
  return torch.atan2(R[..., 1, 0], R[..., 0, 0])


def rot_z(yaw):
  c, s = torch.cos(yaw), torch.sin(yaw)
  z, o = torch.zeros_like(c), torch.ones_like(c)
  return torch.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(*yaw.shape, 3, 3)
