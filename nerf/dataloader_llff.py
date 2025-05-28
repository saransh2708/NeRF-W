import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import random
import torchvision.transforms.functional as TF
from PIL import Image

import torchvision.transforms.functional as TF
from PIL import ImageDraw

def apply_transient_effect(img, image_index):
    """
    Add fake clutter to simulate transient objects. Used for training NeRF-W.
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)

    # Option 1: Add random colored rectangles (simulate cars, signs, clutter)
    if image_index % 3 == 0:
        for _ in range(3):
            x0 = random.randint(0, img.width - 30)
            y0 = random.randint(0, img.height - 30)
            x1 = x0 + random.randint(10, 50)
            y1 = y0 + random.randint(10, 50)
            color = tuple([random.randint(0, 255) for _ in range(3)])
            draw.rectangle([x0, y0, x1, y1], fill=color)

    # Option 2: Brightness tint (simulate lighting variation)
    elif image_index % 3 == 1:
        img = TF.adjust_brightness(img, random.uniform(0.6, 1.4))

    # Option 3: Add haze (simulate fog/smoke)
    else:
        overlay = Image.new("RGB", img.size, (200, 200, 200))
        img = Image.blend(img, overlay, alpha=0.3)

    return TF.to_tensor(img)


def load_poses_bounds(path):
    poses_bounds = np.load(os.path.join(path, "poses_bounds.npy"))
    poses = poses_bounds[:, :-2].reshape([-1, 3, 5])
    bounds = poses_bounds[:, -2:]

    poses = poses.transpose([1, 2, 0])  # (3, 5, N)

    # Use default values for fern dataset (can change if needed)
    H, W, focal = 504, 378, poses[2, 4, 0]
    K = np.array([
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1]
    ])
    return poses, bounds, K, H, W

class LLFFDataset(Dataset):
    def __init__(self, root_dir, downscale=1, apply_transients=True):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.image_paths = sorted([
            os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)
            if f.lower().endswith((".jpg", ".png"))
        ])

        self.poses, self.bounds, self.K, self.H, self.W = load_poses_bounds(root_dir)

        self.K = torch.from_numpy(self.K).float()
        self.H = int(self.H // downscale)
        self.W = int(self.W // downscale)
        self.apply_transients = apply_transients

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        c2w = self.poses[:, :4, idx]  # (3, 4)
        image_path = self.image_paths[idx]

        img = Image.open(image_path).convert("RGB")

        if self.apply_transients:
            img = apply_transient_effect(img, idx)  # already returns tensor
        else:
            img = TF.to_tensor(img)  # convert PIL → tensor only if not augmented

        # Resize to match H, W
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(self.H, self.W), mode='area'
        ).squeeze(0)  # (3, H, W)

        img = img.permute(1, 2, 0)  # → (H, W, 3)

        return {
            "image": img,
            "c2w": torch.from_numpy(c2w).float(),
            "intrinsics": self.K
        }

