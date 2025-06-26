import torch
import torch.nn as nn
import math
from typing import Tuple

class PixelShuffle2D(nn.Module):
    """
    二维PixelShuffle模块
    """
    def __init__(self, upscale_factor):
        """
        :param upscale_factor: tensor的放大倍数
        """
        super(PixelShuffle2D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, inputs):
        B, C, H, W = inputs.size()
        C //= self.upscale_factor ** 2
        H_new = H * self.upscale_factor
        W_new = W * self.upscale_factor

        input_view = inputs.contiguous().view(
            B, C, 
            self.upscale_factor, self.upscale_factor, 
            H, W)

        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()

        return shuffle_out.view(B, C, H_new, W_new)

class SingleConvDecoder2D(nn.Module):
    """
    Only one conv layer for 2D encoder
    """
    def __init__(self, encoder_embed_dim: int, patch_size: Tuple, in_chans: int):
        super(SingleConvDecoder2D, self).__init__()
        self.in_chans = in_chans
        self.encoder_embed_dim = encoder_embed_dim
        self.patch_size = patch_size
        self.decoder_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder_embed_dim,
                out_channels=self.patch_size ** 2 * self.in_chans, 
                kernel_size=1),
            PixelShuffle2D(self.patch_size)
        )

    def forward(self, raw_x, x):
        B, _, H, W = raw_x.shape
        NHp, NWp = H // self.patch_size, W // self.patch_size
        encoder_x = x.reshape(B, self.encoder_embed_dim, NHp, NWp)
        return self.decoder_layers(encoder_x)