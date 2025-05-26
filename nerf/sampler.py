import torch

def sample_points_batch(ray_origins, ray_directions, num_samples, near, far, perturb=True):
    """
    Sample 3D points along a batch of rays.

    Args:
        ray_origins: (B, 3)
        ray_directions: (B, 3)
        num_samples: int
        near: float
        far: float

    Returns:
        pts: (B, N, 3)
        t_vals: (B, N)
    """
    B = ray_origins.shape[0]
    device = ray_origins.device

    t_vals = torch.linspace(near, far, steps=num_samples, device=device)
    t_vals = t_vals.expand(B, num_samples)

    if perturb:
        mids = 0.5 * (t_vals[:, :-1] + t_vals[:, 1:])
        upper = torch.cat([mids, t_vals[:, -1:]], dim=-1)
        lower = torch.cat([t_vals[:, :1], mids], dim=-1)
        t_rand = torch.rand(t_vals.shape, device=device)
        t_vals = lower + (upper - lower) * t_rand

    pts = ray_origins[:, None, :] + t_vals[:, :, None] * ray_directions[:, None, :]
    return pts, t_vals

