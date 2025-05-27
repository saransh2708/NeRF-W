import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

from nerf.model import NeRFW
from nerf.rays import get_rays
from nerf.sampler import sample_points_batch
from nerf.render import volume_render_batch

# Setup device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Rendering on device: {device}")

@torch.no_grad()
def render_decomposed(model, H, W, K, c2w, image_index, num_samples=64, near=2.0, far=6.0):
    model.eval().to(device)

    K = K.to(device)
    c2w = c2w.to(device)
    rays_o, rays_d = get_rays(H, W, K, c2w)
    rays_o = rays_o.reshape(-1, 3).to(device)
    rays_d = rays_d.reshape(-1, 3).to(device)
    B = rays_o.shape[0]

    pts, t_vals = sample_points_batch(rays_o, rays_d, num_samples, near, far)
    dirs = rays_d[:, None, :].expand_as(pts)

    pts_flat = pts.reshape(-1, 3)
    dirs_flat = dirs.reshape(-1, 3)
    idx_flat = torch.full((B * num_samples,), image_index, dtype=torch.long, device=device)

    static_rgb, static_sigma, trans_rgb, trans_sigma, trans_conf = model(pts_flat, dirs_flat, idx_flat)

    # Reshape
    static_rgb = static_rgb.view(B, num_samples, 3)
    trans_rgb = trans_rgb.view(B, num_samples, 3)
    sigma = (static_sigma + trans_sigma).view(B, num_samples, 1)
    trans_conf = trans_conf.view(B, num_samples, 1)

    # Render separately
    rgb_static = volume_render_batch(static_rgb, sigma, t_vals.to(device))
    rgb_transient = volume_render_batch(trans_rgb, sigma, t_vals.to(device))

    # Blended render using transient confidence
    w = trans_conf
    rgb_blend = static_rgb * (1 - w) + trans_rgb * w
    rgb_blend = volume_render_batch(rgb_blend, sigma, t_vals.to(device))

    # Confidence heatmap (avg along depth)
    conf_map = trans_conf.mean(dim=1).view(H, W).cpu().numpy()

    return (
        rgb_static.view(H, W, 3).cpu().numpy(),
        rgb_transient.view(H, W, 3).cpu().numpy(),
        rgb_blend.view(H, W, 3).cpu().numpy(),
        conf_map
    )


def save_image(path, img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    imageio.imwrite(path, img)


def save_confidence_heatmap(path, conf_map):
    plt.figure()
    plt.imshow(conf_map, cmap='hot')
    plt.colorbar(label="Transient Confidence")
    plt.title("Transient Confidence Map")
    plt.axis("off")
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    # Set camera intrinsics (from LLFF 'fern' default)
    H, W = 126, 94
    focal = 326.05
    cx, cy = W / 2, H / 2
    K = torch.tensor([
        [focal, 0, cx],
        [0, focal, cy],
        [0, 0, 1]
    ])

    # Identity camera pose (just for test)
    c2w = torch.eye(4)

    # Load model
    model = NeRFW(num_images=20)
    model.load_state_dict(torch.load("checkpoints/nerfw.pth", map_location=device), strict=False)

    # Render decomposed components
    static, transient, blended, conf_map = render_decomposed(model, H, W, K, c2w, image_index=0)

    os.makedirs("render_outputs", exist_ok=True)
    save_image("render_outputs/static_rgb.png", static)
    save_image("render_outputs/transient_rgb.png", transient)
    save_image("render_outputs/blended_rgb.png", blended)
    save_confidence_heatmap("render_outputs/trans_confidence.png", conf_map)

    print("✅ Saved rendered outputs to render_outputs/")
