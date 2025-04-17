import torch.nn as nn
import torch
import torch.nn.functional as F

class CrossFrameFusion(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, in_channels, (kernel_size, 1, 1), padding=(padding, 0, 0), groups=in_channels)
        
    def forward(self, x):
        # x shape: (B, C, T, H, W)
        out = self.conv(x)
        return x + out
    
class MultiScaleFeatureAlignment(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool1 = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.pool2 = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 4, 4), stride=(1, 4, 4))
        
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv3d(in_channels * 3, in_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(in_channels, in_channels, 1),
        )
        
    def forward(self, x):
        # x shape: (B, C, T, H, W)
        p1 = F.interpolate(self.pool1(x), size=x.shape[2:], mode="nearest")
        p2 = F.interpolate(self.pool2(x), size=x.shape[2:], mode="nearest")
        
        p1 = self.conv1(p1)
        p2 = self.conv2(p2)
        
        fused = torch.cat([x, p1, p2], dim=1)
        return x + self.fuse(fused)
    
class DynamicAttentionMasking(nn.Module):
    def __init__(self, threshold=0.01):
        super().__init__()
        self.threshold = threshold
  
    def forward(self, x):
        with torch.no_grad():
            diffs = torch.mean((x[:, :, 1:] - x[:, :, :-1])**2, dim=[1,3,4], keepdim=True)
            diffs = F.pad(diffs, (0, 0, 0, 0, 1, 0))
            mask = (diffs > self.threshold).float()
            
        return x * mask
    
class LatentDiffusionAlignment(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, in_channels)
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=1),
        )
        
    def forward(self, x):
        # x shape: (B, C, T, H, W)
        return x + self.mlp(self.norm(x))

class TSM(nn.Module):
    def __init__(self, n_div=8):
        super().__init__()
        self.fold_div = n_div

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.size()
        fold = C // self.fold_div
        out = x.clone()

        out[:, :fold] = torch.roll(x[:, :fold], shifts=1, dims=2)   # shift forward
        out[:, fold:2*fold] = torch.roll(x[:, fold:2*fold], shifts=-1, dims=2)  # shift backward

        return out
    