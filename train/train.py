import torch
from nerf.dataloader_llff import LLFFDataset
from nerf.model import NeRFW
from nerf.rays import get_random_rays
from nerf.sampler import sample_points_batch
from nerf.render import volume_render_batch

# --- Detect device (MPS for Apple Silicon, fallback to CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- Config
scene = "fern"
num_epochs = 100
num_rays = 512
lr = 5e-4
near, far = 2.0, 6.0
num_samples = 64

# --- Dataset
dataset = LLFFDataset(f"data/{scene}", downscale=4, apply_transients=True)

# --- Model + Optimizer
model = NeRFW(num_images=len(dataset)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# --- Training loop
for epoch in range(num_epochs):
    total_loss = 0.0

    for img_idx in range(len(dataset)):
        sample = dataset[img_idx]
        rays_o, rays_d, target_rgb = get_random_rays(
            sample["image"], sample["c2w"], sample["intrinsics"], num_rays
        )

        # Move data to device
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)
        target_rgb = target_rgb.to(device)

        # Sample points
        pts, t_vals = sample_points_batch(rays_o, rays_d, num_samples, near, far)
        dirs = rays_d[:, None, :].expand_as(pts)

        # Flatten for model
        B, N = pts.shape[:2]
        pts_flat = pts.reshape(-1, 3).to(device)
        dirs_flat = dirs.reshape(-1, 3).to(device)
        idx_flat = torch.full((B * N,), img_idx, dtype=torch.long).to(device)

        # --- Model forward
        static_rgb, static_sigma, trans_rgb, trans_sigma, trans_conf = model(
            pts_flat, dirs_flat, idx_flat
        )

        # --- Blend RGB using confidence
        # conf ∈ (0, 1), shape: (B*N, 1)
        rgb_blend = static_rgb * (1 - trans_conf) + trans_rgb * trans_conf
        rgb_blend = rgb_blend.view(B, N, 3)
        sigma = (static_sigma + trans_sigma).view(B, N, 1)

        # --- Volume rendering
        rendered_rgb = volume_render_batch(rgb_blend, sigma, t_vals.to(device))

        # --- Loss
        recon_loss = torch.mean((rendered_rgb - target_rgb) ** 2)
        reg_loss = trans_sigma.mean()  # optional: add KL on conf later
        loss = recon_loss + 0.01 * reg_loss

        # --- Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "checkpoints/nerfw.pth")
