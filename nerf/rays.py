import torch

def get_random_rays(image, c2w, intrinsics, num_rays):
    """
    Sample random rays from an image.

    Args:
        image: (H, W, 3)
        c2w: (3, 4)
        intrinsics: (3, 3)
        num_rays: int

    Returns:
        ray_origins: (num_rays, 3)
        ray_directions: (num_rays, 3)
        target_rgb: (num_rays, 3)
    """
    H, W, _ = image.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Random (x, y) pixel indices
    i = torch.randint(0, W, (num_rays,))
    j = torch.randint(0, H, (num_rays,))

    # Convert pixel indices to camera space directions
    dirs = torch.stack([
        (i - cx) / fx,
        -(j - cy) / fy,
        -torch.ones_like(i)
    ], dim=-1)  # (num_rays, 3)

    # Rotate to world space
    R = c2w[:3, :3]  # (3, 3)
    rays_d = (dirs @ R.T).float()  # (num_rays, 3)
    rays_o = c2w[:3, 3].expand_as(rays_d)  # same origin for all rays

    # Get RGB targets
    rgb = image[j, i]  # (num_rays, 3)

    return rays_o, rays_d, rgb


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
    device = K.device

    # Create pixel grid
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing='ij'
    )

    # Normalize pixel coordinates
    dirs = torch.stack([
        (i - K[0, 2]) / K[0, 0],
        -(j - K[1, 2]) / K[1, 1],
        -torch.ones_like(i)
    ], dim=-1)

    # Rotate ray directions from camera to world
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)

    # All rays originate from the same camera position
    rays_o = c2w[:3, 3].expand(rays_d.shape)

    return rays_o, rays_d
