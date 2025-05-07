# Autor: Martin Bublavý [xbubla02]

import torch
import torch.nn as nn
import torch.nn.functional as F

class FFAM(nn.Module):
    """
    Fine-Coarse Frame Attention Module (FFAM)
    - Uses a temporal attention mechanism for better frame consistency
    - Attends to the previous frame and a downsampled version of the current frame
    """

    def __init__(self, dim, num_heads=8, reduction_factor=2):
        """
        Args:
            dim (int): Number of feature channels (C)
            num_heads (int): Number of attention heads
            reduction_factor (int): Factor by which the current frame is downsampled
        """
        super().__init__()
        self.num_heads = num_heads
        self.reduction_factor = reduction_factor

        # Linear projection to ensure correct embedding size
        self.proj = nn.Linear(2 * dim, dim)

        # Multi-head attention for fine-coarse frame interaction
        self.spatial_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, T, H, W) - Feature map of video frames
    
        Returns:
            Updated feature tensor of shape (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        
        if T < 2:
            return x  # No previous frame for FFAM to process
    
        # Extract previous and current frames
        past_frame = x[:, :, :-1, :, :].contiguous()  # Shape: (B, C, T-1, H, W)
        current_frame = x[:, :, 1:, :, :].contiguous()  # Shape: (B, C, T-1, H, W)
    
        B, C, T_minus_1, H, W = current_frame.shape
        current_frame_reshaped = current_frame.reshape(B * T_minus_1, C, H, W)
    
        # Apply downsampling
        downsampled_frame = F.interpolate(
            current_frame_reshaped, scale_factor=1 / self.reduction_factor, mode="bilinear", align_corners=False
        )
    
        # Upsample back to original size
        downsampled_frame = F.interpolate(
            downsampled_frame, size=(H, W), mode="bilinear", align_corners=False
        )
    
        # Reshape back to (B, C, T-1, H, W)
        downsampled_frame = downsampled_frame.reshape(B, C, T_minus_1, H, W)
    
        # Concatenate previous frame and downsampled current frame along the channel dimension
        context = torch.cat([past_frame, downsampled_frame], dim=1)  # Shape: (B, 2C, T-1, H, W)
    
        # Reshape correctly for attention
        context = context.view(B * (T - 1), 2 * C, H * W).permute(0, 2, 1)  # (B*(T-1), HW, 2C)
    
        # Fix embedding dimension mismatch
        context = self.proj(context)  # Shape: (B*(T-1), HW, C)
    
        # Apply Multi-head Self-Attention
        attn_out, _ = self.spatial_attn(context, context, context)  # (B*(T-1), HW, C)
    
        attn_out = attn_out.permute(0, 2, 1).contiguous().reshape(B, C, (T - 1), H, W)
    
        # Merge refinements into the current frame
        refined_output = current_frame + attn_out  # (B, C, T-1, H, W)
    
        # Preserve the first frame
        output = torch.cat([x[:, :, :1, :, :], refined_output], dim=2)  # Shape: (B, C, T, H, W)
    
        return output

