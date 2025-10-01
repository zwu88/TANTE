#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Aug. 2025 Zhikai Wu, New Haven, CT
# All rights reserved.
#
# This work was developed at Yale University.
#
# Licensed under the MIT License. You may obtain a copy of the License at:
# https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
import torch.fft

from torch import einsum
from torch.utils.checkpoint import checkpoint
from torchinfo import summary

import random
from einops import rearrange, repeat

from torch.nn.modules.utils import _pair

import statistics

device = "cuda" if torch.cuda.is_available() else "cpu"

Patch_map = {
    64: (8, 8),
    32: (8, 4),
    16: (4, 4),
    8: (4, 2),
    4: (2, 2),
    2: (2, 1)
}

class RealConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size=1,                 # 1 | (1, 1)
        overlap_ratio: float = 0.0,   # [0, 1). 0 -> (stride=P), 1 -> (stride≈1)
        padding: str = 'same',        # 'valid' | 'same'
        bias: bool = True,
        enforce_patch_grid: bool = True,
    ):
        super().__init__()
        assert 0.0 <= overlap_ratio < 1.0, "overlap_ratio must be in [0, 1)."
        self.P_h, self.P_w = _pair(patch_size)
        assert self.P_h >= 1 and self.P_w >= 1

        # overlap_ratio -> stride
        stride_h = max(1, int(round(self.P_h * (1.0 - overlap_ratio))))
        stride_w = max(1, int(round(self.P_w * (1.0 - overlap_ratio))))
        self.stride = (stride_h, stride_w)

        # padding
        padding = padding.lower()
        if padding not in ('valid', 'same'):
            raise ValueError("padding must be 'valid' or 'same'")
        self.padding_mode = padding
        if padding == 'valid':
            pad = (0, 0)
        else:
            pad = (self._same_padding(self.P_w, self.stride[1]),
                   self._same_padding(self.P_h, self.stride[0]))  # (pw, ph)
            pad = (pad[1], pad[0])  # -> (ph, pw)
        self.padding = pad  # (ph, pw)

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(self.P_h, self.P_w),
            stride=self.stride,
            padding=self.padding,
            bias=bias,
        )

        self.enforce_patch_grid = enforce_patch_grid

    @staticmethod
    def _same_padding(k: int, s: int) -> int:
        return (k - 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert torch.is_floating_point(x), "Input must be a real (floating) tensor."
        B, C_in, H, W = x.shape
        y = self.conv(x)  # (B, C_out, H_conv, W_conv)

        if not self.enforce_patch_grid:
            return y

        assert H % self.P_h == 0 and W % self.P_w == 0, \
            "To enforce (H//P, W//P), input H and W must be divisible by patch_size."
        tgt_h = H // self.P_h
        tgt_w = W // self.P_w
        y = F.adaptive_avg_pool2d(y, output_size=(tgt_h, tgt_w))
        return y

class RealTransConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size=1,                 # 1 | (1, 1)
        overlap_ratio: float = 0.0,   # [0, 1). 0 -> stride=P, 1 -> stride≈1
        padding: str = 'same',        # 'valid' | 'same'
        bias: bool = True,
        enforce_patch_grid: bool = True,
        upsample_mode: str = 'bilinear',
    ):
        super().__init__()
        assert 0.0 <= overlap_ratio < 1.0, "overlap_ratio must be in [0, 1)."
        self.P_h, self.P_w = _pair(patch_size)
        assert self.P_h >= 1 and self.P_w >= 1

        stride_h = max(1, int(round(self.P_h * (1.0 - overlap_ratio))))
        stride_w = max(1, int(round(self.P_w * (1.0 - overlap_ratio))))
        self.stride = (stride_h, stride_w)

        # padding
        padding = padding.lower()
        if padding not in ('valid', 'same'):
            raise ValueError("padding must be 'valid' or 'same'"
            )
        self.padding_mode = padding
        if padding == 'valid':
            pad = (0, 0)
        else:
            pad = (self._same_padding(self.P_w, self.stride[1]),
                   self._same_padding(self.P_h, self.stride[0]))  # (pw, ph)
            pad = (pad[1], pad[0])  # -> (ph, pw)
        self.padding = pad  # (ph, pw)

        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=(self.P_h, self.P_w),
            stride=self.stride,
            padding=self.padding,
            output_padding=0,
            bias=bias,
        )

        self.enforce_patch_grid = enforce_patch_grid
        self.upsample_mode = upsample_mode  # 'nearest' | 'bilinear' | 'bicubic' | ...

    @staticmethod
    def _same_padding(k: int, s: int) -> int:
        return (k - 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert torch.is_floating_point(x), "Input must be a real (floating) tensor."
        B, C_in, H, W = x.shape

        y = self.deconv(x)  # (B, C_out, Ht, Wt)

        if not self.enforce_patch_grid:
            return y

        tgt_h = H * self.P_h
        tgt_w = W * self.P_w

        if y.shape[-2] == tgt_h and y.shape[-1] == tgt_w:
            return y

        y = F.interpolate(
            y, size=(tgt_h, tgt_w),
            mode=self.upsample_mode,
            align_corners=False if self.upsample_mode in ('bilinear', 'bicubic') else None
        )
        return y

class SpectralLayer(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat) *
            (1.0 / (in_channels * out_channels) ** 0.5)
        )
        self.w0 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    @staticmethod
    def complex_mul_low_modes(X_ft: torch.Tensor, weight: torch.Tensor,
                              modes1: int, modes2: int) -> torch.Tensor:
        B, Cin, H, Wf = X_ft.shape
        Cout = weight.shape[1]
        m1 = min(modes1, H)
        m2 = min(modes2, Wf)
        out_ft = torch.zeros(B, Cout, H, Wf, dtype=torch.cfloat, device=X_ft.device)
        if m1 == 0 or m2 == 0:
            return out_ft
        w = weight[:, :, :m1, :m2]
        x_top = X_ft[:, :, :m1, :m2]
        out_ft[:, :, :m1, :m2] = torch.einsum('b c i j, c o i j -> b o i j', x_top, w)
        x_bot = X_ft[:, :, -m1:, :m2]
        out_ft[:, :, -m1:, :m2] = torch.einsum('b c i j, c o i j -> b o i j', x_bot, w)
        return out_ft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4 and x.size(1) == self.in_channels
        B, Cin, H, W = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        y_ft = self.complex_mul_low_modes(x_ft, self.weight, self.modes1, self.modes2)
        y = torch.fft.irfft2(y_ft, s=(H, W), dim=(-2, -1), norm="ortho")
        s = self.w0(x)  # (B, Cout, H, W)
        out = s + y
        return out

class enc_FNO(nn.Module):
    def __init__(
        self, 
        dset_metadata = None,
        embed_dim: int = 256,
        modes: tuple[int, int] = (32, 32),
        patch_scale = 64,
        overlap_ratio = 0.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        modes1, modes2 = modes
        
        patch_size = Patch_map[patch_scale]

        in_channels = dset_metadata.n_fields if dset_metadata else 4
        shape = dset_metadata.spatial_resolution if dset_metadata else (128, 384)
        self.H = shape[0]
        self.W = shape[1]

        self.enc_spectral_1 = SpectralLayer(in_channels, self.embed_dim//8, modes1, modes2)
        self.enc_conv_1 = RealConv2d(self.embed_dim//8, self.embed_dim//4, patch_size=patch_size[0], overlap_ratio=overlap_ratio)
        self.enc_spectral_2 = SpectralLayer(self.embed_dim//4, self.embed_dim//2, modes1//patch_size[0], modes2//patch_size[0])
        self.enc_conv_2 = RealConv2d(self.embed_dim//2, self.embed_dim, patch_size=patch_size[1], overlap_ratio=overlap_ratio)

        self.patch_shape = (
            self.H//(patch_size[0]*patch_size[1]), 
            self.W//(patch_size[0]*patch_size[1])
        )

        self.act = nn.GELU()
    
    def forward(self, x):

        B, T, D, H, W = x.shape
        z = x.view(B*T, D, H, W)  # (B*T, D, H, W)
        z = self.enc_spectral_1(z) # -> (B*T, C/8, H, W)
        z = self.act(z)
        z = self.enc_conv_1(z) # -> (B*T, C/4, H', W')
        z = self.act(z)

        z = self.enc_spectral_2(z) # -> (B*T, C/2, H', W')
        z = self.act(z)
        z = self.enc_conv_2(z) # -> (B*T, C, H_p, W_p)

        z = rearrange(z, '(b t) c h w -> b t h w c', b=B, t=T)
        
        return z


class dec_FNO(nn.Module):
    def __init__(
        self, 
        dset_metadata = None,
        embed_dim: int = 256,
        modes: tuple[int, int] = (32, 32),
        patch_scale = 64,
        overlap_ratio = 0.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        modes1, modes2 = modes
        
        patch_size = Patch_map[patch_scale]

        in_channels = dset_metadata.n_fields if dset_metadata else 4
        shape = dset_metadata.spatial_resolution if dset_metadata else (128, 384)
        self.H = shape[0]
        self.W = shape[1]

        self.dec_conv_1 = RealTransConv2d(self.embed_dim, self.embed_dim//2, patch_size=patch_size[1], overlap_ratio=overlap_ratio)
        self.dec_spectral_1 = SpectralLayer(self.embed_dim//2, self.embed_dim//4, modes1//patch_size[0], modes2//patch_size[0])
        self.dec_conv_2 = RealTransConv2d(self.embed_dim//4, self.embed_dim//8, patch_size=patch_size[0], overlap_ratio=overlap_ratio)
        self.dec_spectral_2 = SpectralLayer(self.embed_dim//8, in_channels, modes1, modes2)

        self.patch_shape = (
            self.H//(patch_size[0]*patch_size[1]), 
            self.W//(patch_size[0]*patch_size[1])
        )

        self.act = nn.GELU()

    
    def forward(self, x):

        B, T, H_p, W_p, C = x.shape
        x = rearrange(x, 'b t h w c -> (b t) c h w') # (B*T, C, H_P, W_P)

        x_rec = self.dec_conv_1(x) # (B*T, C/2, H', W')
        x_rec = self.act(x_rec)
        x_rec = self.dec_spectral_1(x_rec) # (B*T, C/4, H', W')
        x_rec = self.act(x_rec)

        x_rec = self.dec_conv_2(x_rec)        # (B*T, C/8, H, W)
        x_rec = self.act(x_rec)
        x_rec = self.dec_spectral_2(x_rec) # (B*T, D, H, W)

        x_rec = x_rec.view(B, T, -1, self.H, self.W) # (B, T, D, H, W)

        return x_rec



if __name__ == "__main__":
    #torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, T, Cin, H, W = 3, 11, 4, 128, 384
    Cout = 5

    x_in = torch.randn(B, T, Cin, H, W, device=device, requires_grad=True)
    #x_imag = torch.randn(B, T, Cin, H, W, device=device)
    #x = torch.complex(x_real, x_imag).requires_grad_(True)

    layer = enc_FNO(
        embed_dim=256,
        modes=(32, 32),
        patch_scale = 32,
        overlap_ratio = 0.5,
    ).to(device) 

    layer.train()

    total_params = sum(p.numel() for p in layer.parameters())
    trainable_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    nontrainable_params = total_params - trainable_params

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {nontrainable_params:,}")
    print(f"Approx. model size (fp32, trainable only): {trainable_params * 4 / 1024 / 1024:.2f} MB")

    x_latent = layer(x_in)

    print(f"latent shape: {x_latent.shape}, dtype: {x_latent.dtype}")

    print("Forward passed.")

    layer = dec_FNO(
        embed_dim=256,
        modes=(32, 32),
        patch_scale = 32,
        overlap_ratio = 0.5,
    ).to(device) 

    x_out = layer(x_latent)
    print(f"Output shape: {x_out.shape}, dtype: {x_out.dtype}")
    assert x_out.shape == (B, T, Cin, H, W), "输出尺寸不匹配！"

    # ===== 简单损失与反向传播，检查梯度 =====
    # 使用 |out|^2 的均值作为损失，确保对复数可导
    #loss = (out.real**2 + out.imag**2).mean()
    loss = torch.linalg.norm(x_in - x_out) / (torch.linalg.norm(x_in) + 1e-12)
    loss.backward()

    print(f"Loss: {loss.item():.6f}")

    # 检查梯度是否回传
    # 1) 输入梯度
    if x_in.grad is not None:
        print(f"Grad |x|: {x_in.grad.norm().item()}")
    else:
        print("No grad for input x_in (unexpected).")


    print("Backward passed.")