import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
import os

from nerf.model import NeRFW
from nerf.rays import get_rays
from nerf.sampler import sample_points_batch
from nerf.render import volume_render_batch

# --- Device setup (MPS for Mac GPU, fallback to CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Rendering on device: {device}")


@torch.no_grad()
def render_image(model, H, W, K, c2w, image_index, num_samples=64, near=2.0, far=6.0):
    model.eval()
    model = model.to(device)

    # Move to device
    K = K.to(device)
    c2w = c2w.to(device)

    # Generate rays
    rays_o, rays_d = get_rays(H, W, K, c2w)
    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)
    B = rays_o.shape[0]

    print("rays_o:", rays_o.shape)
    print("rays_d:", rays_d.shape)
    print("num_samples:", num_samples)

    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    B = rays_o.shape[0]

     # Sample points
    pts, t_vals = sample_points_batch(rays_o, rays_d, num_samples, near, far)
    dirs = rays_d[:, None, :].expand_as(pts)

    # Flatten
    pts_flat = pts.reshape(-1, 3)
    dirs_flat = dirs.reshape(-1, 3)
    idx_flat = torch.full((B * num_samples,), image_index, dtype=torch.long, device=device)

    # Model forward
    static_rgb, static_sigma, trans_rgb, trans_sigma = model(pts_flat, dirs_flat, idx_flat)
    rgb = (static_rgb + trans_rgb) / 2.0
    sigma = static_sigma + trans_sigma

    rgb = rgb.view(B, num_samples, 3)
    sigma = sigma.view(B, num_samples, 1)

    # Volume render
    rendered_rgb = volume_render_batch(rgb, sigma, t_vals.to(device))

    # Reshape for image
    img = rendered_rgb.view(H, W, 3).cpu().numpy()
    return np.clip(img, 0, 1)


def create_rotated_camera(angle_degrees):
    theta = math.radians(angle_degrees)
    R = torch.tensor([
        [math.cos(theta), 0, -math.sin(theta)],
        [0, 1, 0],
        [math.sin(theta), 0, math.cos(theta)],
    ])
    T = torch.tensor([0.0, 0.0, 0.0])
    c2w = torch.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = T
    return c2w


def make_gif(model, H=126, W=94, image_index=0, output_path="rotation.gif"):
    focal = 3260.5
    cx, cy = W / 2, H / 2
    K = torch.tensor([
        [focal, 0, cx],
        [0, focal, cy],
        [0, 0, 1]
    ])

    os.makedirs("frames", exist_ok=True)
    frames = []

    for i, angle in enumerate(tqdm(range(0, 360, 10))):
        c2w = create_rotated_camera(angle)
        img = render_image(model, H, W, K, c2w, image_index)

        frame = (img * 255).astype(np.uint8)
        frames.append(frame)

        imageio.imwrite(f"frames/frame_{i:03d}.png", frame)

    imageio.mimsave(output_path, frames, fps=10)
    print(f"🎥 Saved GIF to {output_path}")
    print(f"🖼️ Saved individual frames to ./frames/")


if __name__ == "__main__":
    model = NeRFW(num_images=3)  # or however many images your LLFF dataset has
    model.load_state_dict(torch.load("checkpoints/nerfw.pth", map_location=device))

    make_gif(model, H=252, W=188, image_index=0, output_path="rotation.gif")
