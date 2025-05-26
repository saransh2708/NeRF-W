import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs, include_input=True):
        """
        Args:
            num_freqs: number of frequency bands
            include_input: whether to include the raw input
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input

        self.freq_bands = 2.0 ** torch.arange(num_freqs)

    def forward(self, x):
        """
        Args:
            x: (..., dim) tensor of positions or directions

        Returns:
            (..., dim * num_freqs * 2 [+ dim]) encoded features
        """
        out = []
        if self.include_input:
            out.append(x)

        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))

        return torch.cat(out, dim=-1)
