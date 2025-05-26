import torch
import numpy as np
from nerf.rays import get_rays

class MiniDataset(torch.utils.data.Dataset):
    def __init__(self, num_images=3, H=16, W=16, focal=100.0):
        super().__init__()
        self.H, self.W = H, W
        self.focal = focal
        self.num_images = num_images

        self.images = []
        self.poses = []

        for i in range(num_images):
            img = torch.rand(H, W, 3) * 0.7 + 0.3 # random RGB image
            pose = torch.eye(4)       # simple fixed camera pose
            pose[:3, 3] = torch.tensor([0.0, 0.0, -i])  # move camera back per image
            self.images.append(img)
            self.poses.append(pose)

        # Intrinsics matrix
        cx, cy = W / 2, H / 2
        self.K = torch.tensor([
            [focal,    0, cx],
            [   0, focal, cy],
            [   0,     0,  1]
        ])

    def __len__(self):
        return self.num_images * self.H * self.W

    def __getitem__(self, idx):
        # Choose image
        img_idx = idx // (self.H * self.W)
        pixel_idx = idx % (self.H * self.W)
        x = pixel_idx % self.W
        y = pixel_idx // self.W

        # Get data
        image = self.images[img_idx]
        pose = self.poses[img_idx]
        rgb = image[y, x]
        rays_o, rays_d = get_rays(self.H, self.W, self.K, pose)

        ray_o = rays_o[y * self.W + x]
        ray_d = rays_d[y * self.W + x]

        return {
            "ray_o": ray_o,
            "ray_d": ray_d,
            "rgb": rgb,
            "image_index": torch.tensor(img_idx)
        }
