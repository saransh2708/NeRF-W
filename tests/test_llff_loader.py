from nerf.dataloader_llff import LLFFDataset

dataset = LLFFDataset("data/fern", downscale=4)
print(f"Loaded {len(dataset)} images")

sample = dataset[0]
print("Image shape:", sample["image"].shape)
print("Pose shape:", sample["c2w"].shape)
print("Intrinsics:\n", sample["intrinsics"])
