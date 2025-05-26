import torch
from nerf.model import NeRFW

def test_nerfw_forward():
    num_images = 5  # simulate 5 training images
    num_points = 16

    # Instantiate the model
    model = NeRFW(num_images=num_images)

    # Random 3D points and view directions
    x = torch.rand(num_points, 3) * 2 - 1  # positions in [-1, 1]
    d = torch.randn(num_points, 3)
    d = d / torch.norm(d, dim=-1, keepdim=True)  # normalize directions

    # Simulate a batch for image #3
    image_index = torch.tensor(3)

    # Forward pass
    static_rgb, static_sigma, transient_rgb, transient_sigma = model(x, d, image_index)

    # Check shapes
    print("Static RGB shape:", static_rgb.shape)
    print("Static Sigma shape:", static_sigma.shape)
    print("Transient RGB shape:", transient_rgb.shape)
    print("Transient Sigma shape:", transient_sigma.shape)

    assert static_rgb.shape == (num_points, 3)
    assert static_sigma.shape == (num_points, 1)
    assert transient_rgb.shape == (num_points, 3)
    assert transient_sigma.shape == (num_points, 1)

    print("✅ NeRFW forward pass test passed!")

if __name__ == "__main__":
    test_nerfw_forward()
