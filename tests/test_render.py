import torch
from nerf.render import volume_render

def test_volume_render():
    N = 10  # number of sampled points

    # Fake RGB values (like predicted by your MLP)
    rgb = torch.rand(N, 3)  # random colors

    # Fake densities (some random structure)
    sigma = torch.linspace(0.1, 2.0, steps=N).unsqueeze(-1)  # increasing density

    # Depths sampled between near=1.0 and far=4.0
    t_vals = torch.linspace(1.0, 4.0, steps=N)

    # Dummy ray direction (unit vector, unused in this function but realistic to pass)
    ray_dir = torch.tensor([0.0, 0.0, 1.0])

    final_rgb, acc_alpha, depth = volume_render(rgb, sigma, t_vals, ray_dir)

    print("Rendered color:", final_rgb)
    print("Accumulated transmittance:", acc_alpha)
    print("Depth:", depth)

    assert final_rgb.shape == (3,)
    assert isinstance(acc_alpha.item(), float)
    assert isinstance(depth.item(), float)
    print("✅ Volume render test passed!")

if __name__ == "__main__":
    test_volume_render()
