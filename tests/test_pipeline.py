import torch
from nerf.rays import get_rays
from nerf.sampler import sample_points_along_ray
from nerf.model import NeRFStatic
from nerf.render import volume_render

def test_full_pipeline():
    # --- Setup fake camera
    H, W = 4, 4
    fx = fy = 100.0
    cx = W / 2
    cy = H / 2

    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])
    c2w = torch.eye(4)

    # --- Generate rays
    rays_o, rays_d = get_rays(H, W, K, c2w)

    # Pick the first ray
    ray_origin = rays_o[0]
    ray_dir = rays_d[0]

    # --- Sample points along the ray
    pts, t_vals = sample_points_along_ray(
        ray_origin, ray_dir,
        num_samples=32,
        near=1.0,
        far=4.0
    )

    # --- Initialize model
    model = NeRFStatic()
    
    # --- Predict RGB + density
    dirs = ray_dir.expand(pts.shape)  # repeat same direction for all points
    rgb, sigma = model(pts, dirs)

    # --- Volume render
    final_rgb, acc_alpha, depth = volume_render(rgb, sigma, t_vals, ray_dir)

    # --- Output
    print("Final rendered RGB:", final_rgb)
    print("Accumulated alpha:", acc_alpha)
    print("Depth estimate:", depth)
    print("✅ Full pipeline test passed!")

if __name__ == "__main__":
    test_full_pipeline()
