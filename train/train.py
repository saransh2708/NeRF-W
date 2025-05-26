import torch
import os
from torch.utils.data import DataLoader
from nerf.dataloader import MiniDataset
from nerf.model import NeRFW
from nerf.sampler import sample_points_batch  # new vectorized sampler
from nerf.render import volume_render_batch   # new vectorized renderer

# --- Training config
num_epochs = 100
batch_size = 4
num_samples = 24
near, far = 1.0, 4.0
lr = 1e-3

# --- Load dataset
dataset = MiniDataset(num_images=3)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- Model + optimizer
model = NeRFW(num_images=3)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# --- Training loop
for epoch in range(num_epochs):
    total_loss = 0.0

    for batch in loader:
        ray_o = batch["ray_o"]        # (B, 3)
        ray_d = batch["ray_d"]        # (B, 3)
        target_rgb = batch["rgb"]     # (B, 3)
        image_idx = batch["image_index"]  # (B,)

        B = ray_o.shape[0]

        # --- Vectorized point sampling
        pts, t_vals = sample_points_batch(ray_o, ray_d, num_samples, near, far)  # (B, N, 3), (B, N)

        dirs = ray_d[:, None, :].expand_as(pts)  # (B, N, 3)

        # --- Flatten for model
        pts_flat = pts.reshape(-1, 3)           # (B*N, 3)
        dirs_flat = dirs.reshape(-1, 3)         # (B*N, 3)
        idx_flat = image_idx.unsqueeze(1).expand(B, num_samples).reshape(-1)  # (B*N,)

        # --- Model forward
        static_rgb, static_sigma, trans_rgb, trans_sigma = model(pts_flat, dirs_flat, idx_flat)

        # --- Combine outputs
        rgb = (static_rgb + trans_rgb) / 2.0
        sigma = static_sigma + trans_sigma
        sigma = sigma.clamp(min=0.0, max=10.0)

        # --- Reshape for rendering
        rgb = rgb.view(B, num_samples, 3)
        sigma = sigma.view(B, num_samples, 1)

        # --- Volume render
        predicted_rgb = volume_render_batch(rgb, sigma, t_vals)  # (B, 3)

        # --- Loss
        loss = torch.mean((predicted_rgb - target_rgb) ** 2)
        reg = trans_sigma.mean()
        loss = loss + 0.01 * reg

        if torch.isnan(loss):
            print("NaN in loss!")
            continue

        # --- Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/nerfw.pth") 
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.6f}")
