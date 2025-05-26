from nerf.dataloader import MiniDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def test_minidataset():
    dataset = MiniDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in loader:
        print("Ray origins:", batch["ray_o"].shape)
        print("Ray directions:", batch["ray_d"].shape)
        print("RGB:", batch["rgb"].shape)
        print("Image indices:", batch["image_index"])
        break

    print("✅ MiniDataset test passed!")

def show_training_images():
    dataset = MiniDataset(num_images=3, H=4, W=4)
    for i, img in enumerate(dataset.images):
        img_np = img.numpy()
        plt.imshow(img_np)
        plt.title(f"Training Image {i}")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    show_training_images()