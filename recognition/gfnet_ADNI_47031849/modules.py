"""
From Task Sheet:
"Containing the source code of the components of your model. Each component must be
implementated as a class or a function"
"""

import torch
import torch.nn as nn
import torch.fft
from timm.layers import DropPath, to_2tuple, trunc_normal_
import math

# patch embedding
# layer norm
# global filter layer
# feed forward network
# residual connections
# global average pooling
# linear classidier

class PatchEmbedding(nn.Module):
    """
    Image to Patch embedding
    """
    def __init__(self, image_size=224, patch_size=14, in_chans=3, embed_dim=768):
        super().__init__()
        self.image_size = to_2tuple(image_size) # (image_size, image_size) tuple
        self.patch_size = to_2tuple(patch_size) # ensures that value is a tuple
        self.num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # check size
        B, C, H, W = x.shape
        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input image size ({H}*{W}) does not match model ({self.image_size[0]} * {self.image_size[1]})"
        
        # 
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.cmplx_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32)*0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            H = W = int(math.sqrt(N))
        else:
            H, W = spatial_size

        x = x.view(B, H, W, C)
        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.cmplx_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x