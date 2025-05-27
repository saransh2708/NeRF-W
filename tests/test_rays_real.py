from nerf.dataloader_llff import LLFFDataset
from nerf.rays import get_random_rays

dataset = LLFFDataset("data/fern", downscale=4)
sample = dataset[0]

rays_o, rays_d, rgb = get_random_rays(
    image=sample["image"],
    c2w=sample["c2w"],
    intrinsics=sample["intrinsics"],
    num_rays=8
)

print("Ray origins:", rays_o.shape)
print("Ray directions:", rays_d.shape)
print("RGB samples:", rgb.shape)
