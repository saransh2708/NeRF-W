import torch
from nerf.model import NeRFStatic

def test_static_nerf():
    model = NeRFStatic()

    N = 5  # number of points to test
    positions = torch.rand(N, 3) * 2 - 1  # random 3D points in [-1, 1]
    directions = torch.randn(N, 3)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)  # normalize

    rgb, sigma = model(positions, directions)

    print("Input positions:", positions.shape)
    print("Input directions:", directions.shape)
    print("Output RGB:", rgb.shape)
    print("Output Sigma (density):", sigma.shape)

    assert rgb.shape == (N, 3), "RGB output shape is wrong"
    assert sigma.shape == (N, 1), "Sigma output shape is wrong"
    print("✅ Static NeRF forward pass works!")

if __name__ == "__main__":
    test_static_nerf()
