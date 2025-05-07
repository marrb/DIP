# Autor: Martin Bublavý [xbubla02]

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralNormalizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.utils.spectral_norm(nn.Linear(in_features, out_features))

    def forward(self, x):
        return self.linear(x)

class TemporalSelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=True)  # QKV Projection
        self.scale = dim ** -0.5  # Scaling factor
        self.out_proj = SpectralNormalizedLinear(dim, dim)  # Apply spectral norm

    def forward(self, x):
        """
        x: (batch, frames, channels) - Temporal sequence input
        """

        B, T, C = x.shape  # (Batch, Time, Channels)
        x = x - x.mean(dim=1, keepdim=True)  # Instance Centering (IC)

        # Compute Query, Key, Value
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Compute Scaled Dot-Product Attention
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to value
        out = attn_weights @ v
        out = self.out_proj(out)  # Apply spectral-normalized linear transformation

        return out

class STAM(nn.Module):
    """
    Shift-Restricted Temporal Attention Module (STAM)
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.temporal_attention = TemporalSelfAttention(feature_dim)  
        self.linear = SpectralNormalizedLinear(feature_dim, feature_dim)  
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        """
        x: (batch, channels, frames, height, width) --> (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape

        # Reshape for temporal attention: (B, T, C)
        x_temporal = x.mean(dim=[3, 4])  # Mean pool over spatial dimensions (H, W)
        x_temporal = x_temporal.transpose(1, 2)  # (B, C, T) → (B, T, C)

        # Apply attention and linear transformation
        attn_out = self.temporal_attention(x_temporal)  # (B, T, C)
        attn_out = self.linear(attn_out)  # Linear transformation

        # Apply LayerNorm before residual connection
        x_temporal = self.norm(x_temporal + attn_out)  

        # Expand back to original shape (B, C, T, H, W)
        x_temporal = x_temporal.transpose(1, 2)  # Convert back (B, T, C) → (B, C, T)
        x_expanded = x_temporal.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W) 

        return x + x_expanded  # Residual connection
