import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import random
import torchvision.transforms.functional as TF
from PIL import Image

def apply_transient_effect(pil_img, image_index):
    """Apply fake transient artifacts to simulate moving objects, lighting, etc."""
    img = TF.to_tensor(pil_img)

    if isinstance(image_index, torch.Tensor):
        image_index = image_index.item()

    if image_index % 3 == 0:
        # Add a random colored rectangle
        # Add a random colored rectangle (safely)
        H, W = img.shape[1], img.shape[2]
        w = random.randint(10, min(30, W - 1))
        h = random.randint(10, min(30, H - 1))
        x = random.randint(0, W - w)
        y = random.randint(0, H - h)

        img[:, y:y+h, x:x+w] = torch.rand(3, h, w)

    elif image_index % 3 == 1:
        # Light red tint
        tint = torch.tensor([1.0, 0.8, 0.8]).view(3, 1, 1)
        img = img * tint

    else:
        # Brightness variation
        img = img * random.uniform(0.7, 1.3)

    return img.clamp(0, 1)

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

