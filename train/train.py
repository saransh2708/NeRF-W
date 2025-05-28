import torch
from nerf.dataloader_llff import LLFFDataset
from nerf.model import NeRFW
from nerf.rays import get_random_rays
from nerf.sampler import sample_points_batch
from nerf.render import volume_render_batch
import os

# --- Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- Config
scene = "fern"
num_epochs = 300
num_rays = 1024
lr = 5e-4
near, far = 2.0, 6.0
num_samples = 64

# --- Dataset
dataset = LLFFDataset("data/fern", downscale=2, apply_transients=True)

# --- Model + Optimizer
model = NeRFW(num_images=len(dataset)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# --- Make sure checkpoints folder exists
os.makedirs("checkpoints", exist_ok=True)

# --- Training loop
for epoch in range(num_epochs):
    total_loss = 0.0

    for img_idx in range(len(dataset)):
        sample = dataset[img_idx]
        rays_o, rays_d, target_rgb = get_random_rays(
            sample["image"], sample["c2w"], sample["intrinsics"], num_rays
        )

        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)
        target_rgb = target_rgb.to(device)

        pts, t_vals = sample_points_batch(rays_o, rays_d, num_samples, near, far)
        dirs = rays_d[:, None, :].expand_as(pts)

        B, N = pts.shape[:2]
        pts_flat = pts.reshape(-1, 3).to(device)
        dirs_flat = dirs.reshape(-1, 3).to(device)
        idx_flat = torch.full((B * N,), img_idx, dtype=torch.long).to(device)

        # Model forward
        static_rgb, static_sigma, trans_rgb, trans_sigma, confidence = model(pts_flat, dirs_flat, idx_flat)
        rgb_static = static_rgb.view(B, N, 3)
        rgb_trans = trans_rgb.view(B, N, 3)
        sigma = (static_sigma + trans_sigma).view(B, N, 1)
        confidence = confidence.view(B, N, 1)

        # Blend the RGBs using learned confidence
        blended_rgb = (1 - confidence) * rgb_static + confidence * rgb_trans

        # Volume render
        rendered_rgb = volume_render_batch(blended_rgb, sigma, t_vals.to(device))

        # === Losses ===
        recon_loss = torch.mean((rendered_rgb - target_rgb) ** 2)
        sigma_reg = 0.01 * trans_sigma.mean()
        conf_reg = 0.003 * confidence.mean()  # << was 0.1 before — now much softer
        loss = recon_loss + sigma_reg + conf_reg 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f}")

    # Save checkpoint every 50 epochs
    if (epoch + 1) % 50 == 0:
        ckpt_path = f"checkpoints/nerfw_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"✅ Saved checkpoint: {ckpt_path}")
