#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Aug. 2025 Zhikai Wu, Beijing
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch.utils.checkpoint import checkpoint
from torchinfo import summary

import numpy as np
import math
import random
from einops import rearrange, repeat
import transformers
import statistics
from typing import Tuple, Union



def causal_mask(L: int, device=None) -> torch.Tensor:
    return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

class TransformerBlock(nn.Module):
    def __init__(
        self, 
        embed_dim: int, 
        n_head: int, 
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_head, batch_first=True, dropout=dropout, bias=True)

        self.ln2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden, bias=True),
            nn.GELU(approximate="tanh"),    # "NewGELU"
            nn.Linear(hidden, embed_dim, bias=True),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor, # (B, L, C)
        key_padding_mask: torch.Tensor | None = None, # (B, L)
        attn_mask: torch.Tensor | None = None, # (L, L)
        causal: bool = False,
    ) -> torch.Tensor:
        L = x.shape[1]

        qkv = self.ln1(x)

        if causal:
            cm = causal_mask(L, device=x.device)               # (L, L) bool; True = block
            attn_mask = cm if attn_mask is None else (attn_mask.bool() | cm)

        y, _ = self.attn(
            qkv, qkv, qkv,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=causal,
        )
        x = x + self.drop(y)
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


# input: (B, T, H, W, C), where L = H * W

class Attn_Backbone(torch.nn.Module):
    def __init__(
        self, 
        tensor_shape: tuple[int, int, int, int] = (10, 8, 4, 256), # excepted to be (T, H, W, C)
        attn_axes: str = "L TT TT TT L",
        expanded_channel: int = 128,
        n_head: int = 8,
        mlp_ratio: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.T, self.H, self.W, self.C = tensor_shape
        self.L = self.H * self.W

        self.expanded_channel = expanded_channel

        if attn_axes == "":
            raise ValueError(f"Invalid block: empty segment.")
        self.attn_axes = attn_axes

        self.blocks = nn.ModuleList()

        self.vertical_propagator = nn.Sequential(
                        nn.Linear(self.H, self.H),
                        nn.GELU(), nn.Linear(self.H, self.H))
        self.horizontal_propagator = nn.Sequential(
                        nn.Linear(self.W, self.W),
                        nn.GELU(), nn.Linear(self.W, self.W))
        self.temporal_propagator = nn.Sequential(
                        nn.Linear(self.T, self.T),
                        nn.GELU(), nn.Linear(self.T, self.T))
        self.channel_blocks = nn.ModuleList()

        
        for axis in self.attn_axes:            
            if axis in 'LTHWAXY':
                embed_dim = self.C
            elif axis == 'C': # TODO: add expansion method for channel dimension.
                embed_dim = self.expanded_channel
                self.channel_blocks.append(nn.Sequential(
                                        nn.Linear(1, embed_dim//4),
                                        nn.GELU(), nn.Linear(embed_dim//4, embed_dim)))
            
            self.blocks.append(TransformerBlock(embed_dim=embed_dim, n_head=n_head, mlp_ratio=mlp_ratio, dropout=dropout))
        
    def forward(self, x) -> torch.Tensor:

        B, T, H, W, C = x.shape

        channel_index = 0

        x = rearrange(x, 'b t h w c -> b t w c h', b=B, t=T, c=C, h=H, w=W)
        x = x + self.vertical_propagator(x)
        x = rearrange(x, 'b t w c h -> b t h c w', b=B, t=T, c=C, h=H, w=W)
        x = x + self.horizontal_propagator(x)
        x = rearrange(x, 'b t h c w -> b (h w c) t', b=B, t=T, c=C, h=H, w=W)
        x = x + self.temporal_propagator(x)
        x = rearrange(x, 'b (h w c) t -> b t h w c', b=B, t=T, c=C, h=H, w=W)

        for i, axis in enumerate(self.attn_axes):
            if axis == 'T':
                x = rearrange(x, 'b t h w c-> (b h w) t c', b=B, t=T, c=C, h=H, w=W) 
                x = self.blocks[i](x, causal=True)
                x = rearrange(x, '(b h w) t c -> b t h w c', b=B, t=T, c=C, h=H, w=W)

            elif axis == 'H':
                x = rearrange(x, 'b t h w c -> (b t w) h c', b=B, t=T, c=C, h=H, w=W) 
                x = self.blocks[i](x, causal=False)
                x = rearrange(x, '(b t w) h c -> b t h w c', b=B, t=T, c=C, h=H, w=W)

            elif axis == 'W':
                x = rearrange(x, 'b t h w c -> (b t h) w c', b=B, t=T, c=C, h=H, w=W) 
                x = self.blocks[i](x, causal=False)
                x = rearrange(x, '(b t h) w c -> b t h w c', b=B, t=T, c=C, h=H, w=W)

            elif axis == 'L':
                x = rearrange(x, 'b t h w c -> (b t) (h w) c', b=B, t=T, c=C, h=H, w=W) 
                x = self.blocks[i](x, causal=False)
                x = rearrange(x, '(b t) (h w) c -> b t h w c', b=B, t=T, c=C, h=H, w=W)

            elif axis == 'Y':
                x = rearrange(x, 'b t h w c -> (b w) (t h) c', b=B, t=T, c=C, h=H, w=W) 
                x = self.blocks[i](x, causal=False)
                x = rearrange(x, '(b w) (t h) c -> b t h w c', b=B, t=T, c=C, h=H, w=W)

            elif axis == 'X':
                x = rearrange(x, 'b t h w c -> (b h) (t w) c', b=B, t=T, c=C, h=H, w=W) 
                x = self.blocks[i](x, causal=False)
                x = rearrange(x, '(b h) (t w) c -> b t h w c', b=B, t=T, c=C, h=H, w=W)

            elif axis == 'A':
                x = rearrange(x, 'b t h w c -> b (t h w) c', b=B, t=T, c=C, h=H, w=W) 
                x = self.blocks[i](x, causal=False)
                x = rearrange(x, 'b (t h w) c -> b t h w c', b=B, t=T, c=C, h=H, w=W)

            elif axis == 'C':
                x = rearrange(x, 'b t h w c -> (b t h w) c 1', b=B, t=T, c=C, h=H, w=W) 
                x = self.channel_blocks[channel_index](x) # (BTHW, C, E)
                channel_index += 1
                x = self.blocks[i](x, causal=False)[..., -1]
                x = rearrange(x, '(b t h w) c -> b t h w c', b=B, t=T, c=C, h=H, w=W)

        return x # [B, T, H, W, C]



if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = np.random.random_sample((5, 3, 48, 32, 256))

    x = torch.tensor(x).float().to(device)

    model = Attn_Backbone(
        (3,48,32,256),
        'LTCCTHWAXY',
        128,
        )

    model = model.to(device)

    x = model(x)

    print(x.shape)