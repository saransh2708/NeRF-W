import torch

def get_rays(H, W, K, c2w):
    """
    Generate rays for all pixels in an image.

    Args:
        H (int): Image height
        W (int): Image width
        K (torch.Tensor): Intrinsic matrix (3x3)
        c2w (torch.Tensor): Camera-to-world matrix (4x4)

    Returns:
        rays_o: (H*W, 3) Ray origins in world space
        rays_d: (H*W, 3) Ray directions in world space
    """
    # Create pixel grid
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing='ij'
    )

    # Normalize pixel coordinates
    dirs = torch.stack([
        (i - K[0, 2]) / K[0, 0],
        (j - K[1, 2]) / K[1, 1],
        torch.ones_like(i)
    ], dim=-1)  # shape: [H, W, 3]

    # Rotate ray directions from camera to world
    rays_d = (dirs @ c2w[:3, :3].T).reshape(-1, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # All rays originate from the same camera position
    rays_o = c2w[:3, 3].expand(rays_d.shape)

    return rays_o, rays_d
