""" Swin Transformer
A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030

Code/weights from https://github.com/microsoft/Swin-Transformer, original copyright/license info below

S3 (AutoFormerV2, https://arxiv.org/abs/2111.14725) Swin weights from
    - https://github.com/microsoft/Cream/tree/main/AutoFormerV2

Modifications and additions for timm hacked together by / Copyright 2021, Ross Wightman
"""
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import logging
import math
from typing import Any, Dict, Callable, List, Optional, Set, Tuple, Union
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from itertools import chain
from timm.layers import PatchEmbed, Mlp, DropPath, ClassifierHead, to_2tuple, to_ntuple, trunc_normal_, \
    use_fused_attn, resize_rel_pos_bias_table, resample_patch_embed, _assert, use_reentrant_ckpt, ndgrid
from timm.models._features import feature_take_indices
from timm.models import checkpoint_seq
from dinov2.layers import NestedTensorBlock
from .swin_decoder import SingleConvDecoder2D

__all__ = ['SwinTransformer'] 

_logger = logging.getLogger(__name__)

_int_or_tuple_2_t = Union[int, Tuple[int, int]]


def window_partition(
        x: torch.Tensor,
        window_size: Tuple[int, int],
) -> torch.Tensor:
    """Partition into non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: Tuple[int, int], H: int, W: int) -> torch.Tensor:
    """Reverse window partition."""
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


def get_relative_position_index(win_h: int, win_w: int) -> torch.Tensor:
    """Get pair-wise relative position index for each token inside the window."""
    # 这里会使用我们自己定义的 ndgrid 函数，所以代码无需更改
    coords = torch.stack(ndgrid(torch.arange(win_h), torch.arange(win_w)))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias."""
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: Optional[int] = None,
            window_size: _int_or_tuple_2_t = 7,
            qkv_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn(experimental=True)

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * win_h - 1) * (2 * win_w - 1), num_heads))
        self.register_buffer("relative_position_index", get_relative_position_index(win_h, win_w), persistent=False)

        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        # self.softmax = nn.Softmax(dim=-1)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.fused_attn:
            attn_mask = self._get_rel_pos_bias()
            if mask is not None:
                num_win = mask.shape[0]
                mask = mask.view(1, num_win, 1, N, N).expand(B_ // num_win, -1, self.num_heads, -1, -1)
                attn_mask = attn_mask + mask.reshape(-1, self.num_heads, N, N)
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn + self._get_rel_pos_bias()
            if mask is not None:
                num_win = mask.shape[0]
                attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
            attn = softmax(attn, dim=-1, dtype=x.dtype)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""
    def __init__(
            self,
            dim: int,
            input_resolution: _int_or_tuple_2_t,
            num_heads: int = 4,
            head_dim: Optional[int] = None,
            window_size: _int_or_tuple_2_t = 7,
            shift_size: int = 0,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = to_2tuple(input_resolution)
        self.window_size = to_2tuple(window_size)
        self.shift_size = to_2tuple(shift_size)
        self.window_area = self.window_size[0] * self.window_size[1]
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, num_heads=num_heads, head_dim=head_dim, window_size=self.window_size,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer, drop=proj_drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        if any(self.shift_size):
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask, persistent=False)
        
    def _attn(self, x):
        B, H, W, C = x.shape
        if any(self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x
        
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_area, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if any(self.shift_size):
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        else:
            x = shifted_x
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x_skip = x
        x = self.norm1(x)
        x = self._attn(x)
        x = x_skip + self.drop_path1(x)
        
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x_skip + self.drop_path2(x)
        
        x = x.reshape(B, H, W, C)
        return x

class PatchMerging(nn.Module):
    """Patch Merging Layer."""
    def __init__(self, dim: int, out_dim: Optional[int] = None, norm_layer: Callable = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        pad_values = (0, 0, 0, W % 2, 0, H % 2)
        x = nn.functional.pad(x, pad_values)
        _, H_pad, W_pad, _ = x.shape
        x = x.reshape(B, H_pad // 2, 2, W_pad // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class SwinTransformerStage(nn.Module):
    """A basic Swin Transformer layer for one stage."""
    def __init__(
            self,
            dim: int,
            out_dim: int,
            input_resolution: Tuple[int, int],
            depth: int,
            downsample: bool = True,
            num_heads: int = 4,
            head_dim: Optional[int] = None,
            window_size: _int_or_tuple_2_t = 7,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: Union[List[float], float] = 0.,
            norm_layer: Callable = nn.LayerNorm,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.output_resolution = tuple(i // 2 for i in input_resolution) if downsample else input_resolution
        self.depth = depth
        self.grad_checkpointing = False
        window_size = to_2tuple(window_size)
        shift_size = tuple([w // 2 for w in window_size])

        if downsample:
            self.downsample = PatchMerging(dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()
        self.blocks = nn.Sequential(*[
            SwinTransformerBlock(
                dim=out_dim, input_resolution=self.output_resolution,
                num_heads=num_heads, head_dim=head_dim, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else shift_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_drop=proj_drop,
                attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, **kwargs
            )
            for i in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class SwinTransformer(nn.Module):
    """Swin Transformer."""
    def __init__(
            self,
            img_size: _int_or_tuple_2_t = 224,
            patch_size: int = 4,
            in_chans: int = 3,
            # num_classes: int = 1000,
            # global_pool: str = 'avg',
            embed_dim: int = 96,
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            head_dim: Optional[int] = None,
            window_size: _int_or_tuple_2_t = 7,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            embed_layer: Callable = PatchEmbed,
            norm_layer: Union[str, Callable] = nn.LayerNorm,
            **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)
        
        if not isinstance(embed_dim, (tuple, list)):
            embed_dim_list = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        else:
            embed_dim_list = embed_dim
        
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim_list[0], norm_layer=norm_layer, output_fmt='NHWC',
        )
        patch_grid = self.patch_embed.grid_size
        
        head_dim = to_ntuple(self.num_layers)(head_dim)
        window_size = to_ntuple(self.num_layers)(window_size)
        mlp_ratio = to_ntuple(self.num_layers)(mlp_ratio)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        
        layers = []
        in_dim = embed_dim_list[0]
        scale = 1
        for i in range(self.num_layers):
            out_dim = embed_dim_list[i]
            layers += [SwinTransformerStage(
                dim=in_dim, out_dim=out_dim,
                input_resolution=(patch_grid[0] // scale, patch_grid[1] // scale),
                depth=depths[i], downsample=i > 0, num_heads=num_heads[i],
                head_dim=head_dim[i], window_size=window_size[i], mlp_ratio=mlp_ratio[i],
                qkv_bias=qkv_bias, proj_drop=proj_drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer, **kwargs,
            )]
            in_dim = out_dim
            if i > 0:
                scale *= 2
        self.layers = nn.Sequential(*layers)
        
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.decoder = SingleConvDecoder2D(encoder_embed_dim=embed_dim // 8, 
        #                                    patch_size=patch_size,
        #                                    in_chans=1
        #                                    )
    
    def prepare_tokens_with_masks(self, x, masks=None):
        x = self.patch_embed(x)
        if masks is not None:
            embed_x = x
            x = x.reshape(x.shape[0], -1, x.shape[-1])
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype), x)
            x = x.reshape(embed_x.shape)
        return x
    
    def get_intermediate_layers(
            self, x: torch.Tensor,
            n: Optional[Union[int, List[int]]] = None,
            norm: bool = True,
            intermediates_only: bool = True,
            output_fmt: str = 'NCHW',
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        assert output_fmt in ('NCHW', 'NHWC'), 'Output shape must be NCHW or NHWC.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.layers), n)
        
        x = self.patch_embed(x)
        
        for i, stage in enumerate(self.layers):
            x = stage(x)
            if i in take_indices:
                x_inter = self.norm(x) if norm and i == len(self.layers) - 1 else x
                if output_fmt == 'NCHW':
                    x_inter = x_inter.permute(0, 3, 1, 2).contiguous()
                intermediates.append(x_inter)
        
        if intermediates_only:
            return intermediates
        
        x = self.norm(x)
        if output_fmt == 'NCHW':
            x = x.permute(0, 3, 1, 2).contiguous()
        return x, intermediates

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x = self.layers(x)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2).flatten(2) # BCHW
            cls_x = self.avgpool(x)
            x = torch.cat([cls_x, x], dim=-1).transpose(1, 2) # B N+1 C

            output.append(
                {
                    "x_norm_clstoken": x[:, 0],
                    "x_norm_patchtokens": x[:, 1:],
                    # "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x: torch.Tensor, masks=None) -> torch.Tensor:
        # for student input
        if isinstance(x, list): 
            return self.forward_features_list(x, masks)
        
        x = self.prepare_tokens_with_masks(x, masks)
        x = self.layers(x)
        x = self.norm(x)
        
        x = x.permute(0, 3, 1, 2).flatten(2) # BCHW
        cls_x = self.avgpool(x)

        x = torch.cat([cls_x, x], dim=-1).transpose(1, 2) # B N+1 C

        return {
            "x_norm_clstoken": x[:, 0],
            "x_norm_patchtokens": x[:, 1:],
            "masks": masks,
        }

    def forward(self, *args, **kwargs) -> Dict:
        ret = self.forward_features(*args, **kwargs)
        return ret

def swin_base(img_size=256, patch_size=4, in_chans=1, init_values=None):
    model = SwinTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=128, 
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        window_size=8,
    )
    return model

def swin_base(img_size=256, patch_size=4, in_chans=1, init_values=None, **kwargs):
    model = SwinTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=128, 
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        window_size=8,
    )
    return model

def swin_large(img_size=256, patch_size=4, in_chans=1, init_values=None, **kwargs):
    model = SwinTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=192, 
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        window_size=8,
    )
    return model


# import torch
# import torch.nn as nn
# from typing import List
# ==============================================================================
#             封装好的、即插即用的 SwinTransformerLarge 类
# ==============================================================================
# class SwinTransformerLarge(nn.Module):
#     """
#     一个封装好的、配置固定的 Swin Transformer Backbone。

#     这个类硬编码了特定的 Swin Transformer 配置，并设计为直接输出
#     最后三个阶段的特征图，专用于作为检测模型（如RT-DETR）的主干网络。

#     输入:
#         x (torch.Tensor): 输入图像张量，形状为 (B, 1, 512, 512)。

#     输出:
#         list[torch.Tensor]: 一个包含3个特征图张量的列表，分别对应
#                             Swin Transformer 的 Stage 1, 2, 3 的输出。
#                             - [ (B, 256, 64, 64),
#                             -   (B, 512, 32, 32),
#                             -   (B, 1024, 16, 16) ]
#     """
#     def __init__(self):
#         super().__init__()

#         # 1. 硬编码所有配置参数
#         model_kwargs = {
#             'embed_dim': 128,
#             'depths': (2, 2, 18, 2),
#             'num_heads': (6, 12, 24, 48),
#             'window_size': 8,
#             'img_size': (512, 512),
#             'patch_size': 4,
#             'in_chans': 1,
#             'num_classes': 0,  # 作为 backbone，不需要分类头
#         }

#         # 2. 实例化底层的 SwinTransformer 并作为成员变量
#         #    这是确保模型可训练的关键！
#         self.backbone = SwinTransformer(**model_kwargs)
        
#         # 3. 定义需要从 backbone 中抽取的特征层索引
#         #    我们需要所有层来获取最后三层
#         self.out_indices = (0, 1, 2, 3)

#     def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
#         """
#         定义前向传播逻辑。
#         """
#         # 调用底层 backbone 的方法来获取所有中间层的特征
#         all_features = self.backbone.forward_intermediates(
#             x,
#             indices=self.out_indices,
#             intermediates_only=True,
#             output_fmt='NCHW'
#         )

#         # 根据需求，只返回最后三层的特征图
#         # all_features 是一个列表，我们对其进行切片
#         return all_features[1:]

''' 
================================================================================
                              模型输入
================================================================================
Input Image Shape: (B, 1, 512, 512)

      |
      |
      V

================================================================================
                        模块 1: Patch Embedding
================================================================================
- 作用: 将 512x512 图像切分成 4x4 的小块 (Patch)，并映射到128维。
- 计算: H/W = 512/4=128, C=128
- 输出 Shape: (B, 128, 128, 128)  <-- 这是进入 Stage 0 的特征

      |
      |
      V

================================================================================
                       Stage 0 (depth=2, heads=6)
================================================================================
- 降采样: 无 (第一个Stage不降采样)
- 核心: 2 x SwinTransformerBlock (W-MSA, SW-MSA)
  - Block 1 输入: (B, 128, 128, 128) -> 输出: (B, 128, 128, 128)
  - Block 2 输入: (B, 128, 128, 128) -> 输出: (B, 128, 128, 128)
- 输出 Shape (特征1): (B, 128, 128, 128)

      |
      |
      V

================================================================================
                       Stage 1 (depth=2, heads=12)
================================================================================
- 降采样 (PatchMerging):
  - 输入: (B, 128, 128, 128)
  - 输出: (B, 64, 64, 256)     <-- H/W减半, C翻倍
- 核心: 2 x SwinTransformerBlock (在新尺寸上计算)
  - Block 1 输入: (B, 64, 64, 256) -> 输出: (B, 64, 64, 256)
  - Block 2 输入: (B, 64, 64, 256) -> 输出: (B, 64, 64, 256)
- 输出 Shape (特征2): (B, 64, 64, 256)

      |
      |
      V

================================================================================
                      Stage 2 (depth=18, heads=24)
================================================================================
- 降采样 (PatchMerging):
  - 输入: (B, 64, 64, 256)
  - 输出: (B, 32, 32, 512)     <-- H/W减半, C翻倍
- 核心: 18 x SwinTransformerBlock (网络最深的部分)
  - 每个Block的输入输出尺寸均为 (B, 32, 32, 512)
- 输出 Shape (特征3): (B, 32, 32, 512)

      |
      |
      V

================================================================================
                       Stage 3 (depth=2, heads=48)
================================================================================
- 降采样 (PatchMerging):
  - 输入: (B, 32, 32, 512)
  - 输出: (B, 16, 16, 1024)    <-- H/W减半, C翻倍
- 核心: 2 x SwinTransformerBlock
  - Block 1 输入: (B, 16, 16, 1024) -> 输出: (B, 16, 16, 1024)
  - Block 2 输入: (B, 16, 16, 1024) -> 输出: (B, 16, 16, 1024)
- 输出 Shape (特征4): (B, 16, 16, 1024)

      |
      |
      V

================================================================================
                         最终多尺度特征输出
================================================================================
- (如果使用 forward_intermediates 输出 NCHW 格式)
- 特征列表 [
    torch.Size([B, 128, 128, 128]),  <-- from Stage 0
    torch.Size([B, 256, 64, 64]),    <-- from Stage 1
    torch.Size([B, 512, 32, 32]),    <-- from Stage 2
    torch.Size([B, 1024, 16, 16])    <-- from Stage 3
  ]
================================================================================
'''