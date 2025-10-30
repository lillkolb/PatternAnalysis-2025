"""
This code is adapted from the original implementation of GFNet
See:
    https://arxiv.org/abs/2107.00645
    https://github.com/raoyongming/GFNet
    https://github.com/raoyongming/GFNet/blob/master/gfnet.py#L266

The code has been adjusted for a binary classification problem
"""

import torch
import torch.nn as nn
import torch.fft
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
from collections import OrderedDict

class PatchEmbedding(nn.Module):
    """
    Image to Patch embedding
    """
    def __init__(self, image_size=224, patch_size=14, in_chans=3, embed_dim=768):
        super().__init__()
        self.image_size = to_2tuple(image_size) # (image_size, image_size) tuple
        self.patch_size = to_2tuple(patch_size) # ensures that value is a tuple
        self.num_patches = (self.image_size[1] // self.patch_size[1]) * (self.image_size[0] // self.patch_size[0])

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # check size
        B, C, H, W = x.shape
        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input image size ({H}*{W}) does not match model ({self.image_size[0]} * {self.image_size[1]})"

        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class GlobalFilter(nn.Module):
    """
    Global mixing in the spatial dimension

    NOTE: This layer replaces the self-attention layer in ViTs
    """
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

        # apply 2D Fourier transform for frequency domain
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        # apply multiplication between learnable global features and freq domain features
        weight = torch.view_as_complex(self.cmplx_weight)
        x = x * weight

        # apply 2D Inverse Fourier transform for time domain
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x

class MultiLayerPerceptron(nn.Module):
    """
    Per-token depth-wise feature transformation

    MLP provides non-linearity to the global filter layer (with GELU) as well as
    channel interaction

    In the GFNet, the MLP consits of linear projection, activation function (GELU) and dropout layer
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.linear_layer_1 = nn.Linear(in_features, hidden_features)   # first linear projection
        self.activation_layer = act_layer()                             # activation layer
        self.linear_layer_2 = nn.Linear(hidden_features, out_features)  # second linear projection
        self.dropout_layer = nn.Dropout(drop)                           # dropout layer

    def forward(self, x):
        x = self.linear_layer_1(x)      # apply first linear projection
        x = self.activation_layer(x)    # apply activation layer
        x = self.dropout_layer(x)       # apply dropout layer
        x = self.linear_layer_2(x)      # apply second linear projection
        x = self.dropout_layer(x)       # apply dropout layer again
        return x

class Block(nn.Module):
    """
    GFNet Block, connects Global Filter layer and MLP layer
    """
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MultiLayerPerceptron(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x

class GFNet(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1, embed_dim=768, depth=12,
                 mlp_ratio=4., representation_size=None, uniform_drop=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=None,
                 dropcls=0):
        """
        num_classes = 1 for binary classification

        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbedding(
                image_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        h = img_size // patch_size
        w = h // 2 + 1

        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            # print('using linear droppath with expect rate', drop_path_rate * 0.5)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, mlp_ratio=mlp_ratio,
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, h=h, w=w)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x).mean(1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.head(x)
        return x