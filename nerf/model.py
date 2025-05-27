import torch
import torch.nn as nn
from nerf.utils import PositionalEncoding

class NeRFStatic(nn.Module):
    def __init__(self, 
                 pos_dim=3, 
                 dir_dim=3, 
                 num_pos_freqs=10, 
                 num_dir_freqs=4, 
                 hidden_dim=256, 
                 skips=[4]):
        super().__init__()

        self.pos_encoder = PositionalEncoding(num_pos_freqs)
        self.dir_encoder = PositionalEncoding(num_dir_freqs)

        input_pos_dim = pos_dim * (2 * num_pos_freqs + 1)
        input_dir_dim = dir_dim * (2 * num_dir_freqs + 1)

        self.skips = skips

        self.pts_linears = nn.ModuleList()
        for i in range(8):
            if i == 0:
                in_dim = input_pos_dim
            elif i in skips:
                in_dim = hidden_dim + input_pos_dim
            else:
                in_dim = hidden_dim
            self.pts_linears.append(nn.Linear(in_dim, hidden_dim))

        self.density_layer = nn.Linear(hidden_dim, 1)
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)

        self.dir_layer = nn.Sequential(
            nn.Linear(hidden_dim + input_dir_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )

    def forward(self, x, d):
        x_encoded = self.pos_encoder(x)
        d_encoded = self.dir_encoder(d)

        h = x_encoded
        for i, layer in enumerate(self.pts_linears):
            if i in self.skips:
                h = torch.cat([x_encoded, h], dim=-1)
            h = layer(h)
            h = torch.relu(h)

        sigma = self.density_layer(h)
        features = self.feature_layer(h)
        h_dir = torch.cat([features, d_encoded], dim=-1)
        rgb = self.dir_layer(h_dir)

        return rgb, sigma


class NeRFW(nn.Module):
    def __init__(self, 
                 num_images, 
                 pos_dim=3, 
                 dir_dim=3, 
                 num_pos_freqs=10, 
                 num_dir_freqs=4, 
                 hidden_dim=256, 
                 latent_dim=16, 
                 skips=[4]):
        super().__init__()

        self.static = NeRFStatic(
            pos_dim=pos_dim,
            dir_dim=dir_dim,
            num_pos_freqs=num_pos_freqs,
            num_dir_freqs=num_dir_freqs,
            hidden_dim=hidden_dim,
            skips=skips
        )

        # Image-specific embeddings
        self.embedding_a = nn.Embedding(num_images, latent_dim)
        self.embedding_b = nn.Embedding(num_images, latent_dim)

        self.transient_head = nn.Sequential(
            nn.Linear(latent_dim + pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5)  # RGB (3) + density (1) + confidence(1)
        )

    def forward(self, x, d, image_index):
        static_rgb, static_sigma = self.static(x, d)

        # Handle image embeddings safely (works for int or tensor index)
        embed_a = self.embedding_a(image_index)
        embed_b = self.embedding_b(image_index)

        if embed_a.dim() == 1:
            embed_a = embed_a.unsqueeze(0).repeat(x.shape[0], 1)
            embed_b = embed_b.unsqueeze(0).repeat(x.shape[0], 1)
        elif embed_a.shape[0] != x.shape[0]:
            # Happens if image_index is broadcasted wrong
            embed_a = embed_a.repeat_interleave(x.shape[0] // embed_a.shape[0], dim=0)
            embed_b = embed_b.repeat_interleave(x.shape[0] // embed_b.shape[0], dim=0)

        transient_input = torch.cat([x, embed_b], dim=-1)
        out = self.transient_head(transient_input)
        transient_rgb = torch.sigmoid(out[:, :3])      # RGB
        transient_sigma = torch.relu(out[:, 3:4])      # Density
        transient_conf = torch.sigmoid(out[:, 4:5])    # Confidence (0–1)

        return static_rgb, static_sigma, transient_rgb, transient_sigma, transient_conf

