import torch
from nerf.rays import get_rays
from nerf.sampler import sample_points_along_ray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nerf.utils import PositionalEncoding

def test_sample():
    # Fake image and camera parameters
    H, W = 4, 4  # Small test image
    fx = fy = 100.0
    cx = W / 2
    cy = H / 2

    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])

    # Identity pose: camera at origin, looking down +z
    c2w = torch.eye(4)

    # Generate rays for all pixels
    rays_o, rays_d = get_rays(H, W, K, c2w)

    print(f"Total rays: {rays_o.shape[0]}")
    print(f"First ray origin: {rays_o[0]}")
    print(f"First ray direction: {rays_d[0]}")

    # Sample points along the first ray
    num_samples = 10
    near, far = 1.0, 4.0

    pts, t_vals = sample_points_along_ray(rays_o[0], rays_d[0], num_samples, near, far)

    print("\nSampled depths:")
    print(t_vals)

    print("\nSampled 3D points:")
    print(pts)
    plot_ray_and_samples(rays_o[0], rays_d[0], pts)

def plot_ray_and_samples(ray_origin, ray_direction, pts):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the ray direction (as a line)
    ray_line = ray_origin[None, :] + torch.linspace(0, 5, steps=100)[:, None] * ray_direction[None, :]
    ax.plot(ray_line[:, 0], ray_line[:, 1], ray_line[:, 2], label='Ray', color='gray')

    # Plot the sampled 3D points
    pts_np = pts.detach().cpu().numpy()
    ax.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2], color='red', label='Sampled Points')

    # Plot the camera origin
    origin_np = ray_origin.detach().cpu().numpy()
    ax.scatter(*origin_np, color='blue', label='Camera Origin', s=50)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Ray and Sampled 3D Points")
    plt.show()

if __name__ == "__main__":
    # test_sample()
    encoder = PositionalEncoding(num_freqs=6)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    encoded = encoder(x)

    print("Input shape:", x.shape)
    print("Encoded shape:", encoded.shape)

    
