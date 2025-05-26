import torch

def volume_render_batch(rgb, sigma, t_vals):
    """
    Vectorized volume render for a batch of rays.

    Args:
        rgb: (B, N, 3)
        sigma: (B, N, 1)
        t_vals: (B, N)

    Returns:
        final_rgb: (B, 3)
    """
    delta = t_vals[:, 1:] - t_vals[:, :-1]
    last_delta = torch.full((t_vals.shape[0], 1), 1e-2, device=t_vals.device)
    delta = torch.cat([delta, last_delta], dim=-1)  # (B, N)

    sigma = sigma.squeeze(-1).clamp(min=0.0, max=10.0)  # (B, N)

    alpha = 1.0 - torch.exp(-sigma * delta)  # (B, N)
    trans = torch.cumprod(torch.cat([
        torch.ones((sigma.shape[0], 1), device=sigma.device), 
        1.0 - alpha + 1e-10
    ], dim=1), dim=1)[:, :-1]

    weights = alpha * trans  # (B, N)
    rgb_map = torch.sum(weights[..., None] * rgb, dim=1)  # (B, 3)

    return rgb_map
