import torch
import torch.nn as nn

class FFAM(nn.Module):
    """
    Fine-Coarse Frame Attention Module
    - Improves Spatial consistency between frames
    - Uses a multi-head attention mechanism
    """

    """
    Init method for FFAM
    """
    def __init__(self, dim):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads = 8) #Standart multi-head attention mechanism
    
    """
    Perform forward pass of FFAM
    """
    def forward(self, x):
        # Extract batch size, channels, time steps, height and width
        b, c, t, h, w = x.shape
        
        # Flatten spatial dimensions
        x = x.view(b, c, t, -1)
        
        # Apply self-attention mechanism
        x, _ = self.spatial_attn(x, x, x)
        
        # Reshape back to original shape
        return x.view(b, c, t, h, w)
    