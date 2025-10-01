#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Sep. 2025 Zhikai Wu, Beijing
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
import statistics

from data.dataset import TanteMetadata

from .enc_dec_cnn import enc_CNN, dec_CNN
from .enc_dec_fno import enc_FNO, dec_FNO
from .attn_backbone import Attn_Backbone

class TANTE(torch.nn.Module):
    def __init__(
        self, 
        in_T,
        dset_metadata: TanteMetadata = None, 

        taylor_order: int = 1,
        frame_interval: float = 1.0, # real-world time interval between 2 frames in the dataset.
        output_length = 1,

        attn_axes: str = "THWTHWTHW",
        expanded_channel: int = 128, # if applicable (axes contain 'C')
        n_head: int = 8,
        mlp_ratio: float = 1.0,
        dropout: float = 0.0,

        enc_dec_type: str = 'cnn', # 'cnn' | 'fno'
        embed_dim: int = 256,
        modes1: int = 32, # if applicable
        modes2: int = 32, # if applicable
        patch_scale: int = 32,
        overlap_ratio: float = 0.0,
        deg: bool = True,
    ):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        n_channel = dset_metadata.n_fields if dset_metadata else 4
        self.T = in_T
        shape = dset_metadata.spatial_resolution if dset_metadata else (128, 384)
        self.H_p = shape[0] // patch_scale
        self.W_p = shape[1] // patch_scale
        self.C = embed_dim
        self.taylor_order = taylor_order
        self.frame_interval = frame_interval
        self.output_length = output_length
        self.deg = deg

        self.attn_axes = attn_axes.replace(" ", "")
        if set(self.attn_axes) - {'T', 'H', 'W', 'L', 'A', 'C', 'X,', 'Y', '-'}:
            raise ValueError("There are invalid letters")

        self.blocks_axes = [p.strip() for p in self.attn_axes.split("-")]
        if len(self.blocks_axes) != taylor_order:
            raise ValueError(
                f"Block allocation doesn't match expansion order: expected {taylor_order} parts, got {len(self.blocks_axes)} (input='{self.attn_axes}')."
            )
        
        self.decoders = nn.ModuleList()

        if enc_dec_type == 'cnn':
            self.encoder = enc_CNN(dset_metadata=dset_metadata, embed_dim=embed_dim, 
                                    patch_scale=patch_scale, overlap_ratio=overlap_ratio)
            for i in range(taylor_order):
                self.decoders.append(
                    dec_CNN(dset_metadata=dset_metadata, embed_dim=embed_dim, 
                            patch_scale=patch_scale, overlap_ratio=overlap_ratio)
                )
        elif enc_dec_type == 'fno':
            self.encoder = enc_FNO(dset_metadata=dset_metadata, embed_dim=embed_dim, 
                                    modes=(modes1, modes2), patch_scale=patch_scale, overlap_ratio=overlap_ratio)
            for i in range(taylor_order):
                self.decoders.append(
                    dec_FNO(dset_metadata=dset_metadata, embed_dim=embed_dim, 
                            modes=(modes1, modes2), patch_scale=patch_scale, overlap_ratio=overlap_ratio)
                )
        
        self.blocks = nn.ModuleList()

        for block_axes in self.blocks_axes:
            self.blocks.append(
                Attn_Backbone(tensor_shape=(self.T, self.H_p, self.W_p, self.C),
                                      attn_axes=block_axes,
                                      expanded_channel=expanded_channel,
                                      n_head=n_head,
                                      mlp_ratio=mlp_ratio,
                                      dropout=dropout)
            )

        self.t_emb = nn.Parameter(t_emb_init(self.C, self.T))
        self.s_emb = nn.Parameter(s_emb_init(self.C, (self.H_p, self.W_p), flatten=False))
        self.t_seq = t_series(self.T, frame_interval).to(device)
        self.t_encode = film(self.C, in_dim=1)

        if not self.deg:
            self.interprators = nn.ModuleList([interprator(self.C, self.H_p * self.W_p) for _ in range(self.taylor_order)])
            self.modifiers = nn.ModuleList([film(self.C, in_dim=1) for _ in range(self.taylor_order)])

    def forward(self, input, out_T=1):

        if input.shape[1]!=self.T:
            input = input[:, -self.T:, ...]

        B, T, D, H, W = input.shape

        x = self.encoder(input) # (B, T, H_p, W_p, C)

        _, _, H_p, W_p, C = x.shape

        x = self.t_encode(x, self.t_seq)

        x = x + self.s_emb
        x = rearrange(x, 'b t h w c -> (b h w) t c')
        x = x + self.t_emb
        x = rearrange(x, '(b h w) t c -> b t h w c', b=B, h=H_p, w=W_p)

        derivatives = []
        r_t = []
        for i in range(self.taylor_order):
            x = self.blocks[i](x)
            derivative = x[:, -1:, ...]
            if not self.deg:
                rt = self.interprators[i](rearrange(derivative, 'b 1 h w c -> b (h w) c'), out_T)
                r_t.append(rt)
                derivative = self.modifiers[i](derivative, rt) # (B, L, C)
                derivative = rearrange(derivative, 'b (h w) c -> b 1 h w c', h=H_p, w=W_p)
            derivative = self.decoders[i](derivative)
            derivatives.append(derivative)

        outputs = []

        if not self.deg:
            r_t = torch.stack(r_t, dim=1) # (B, n)
            R_t = torch.mean(r_t, dim=1) # (B,)

        ########## NOTE ############## TEMPORARY ############## TODO ########
        output_length = self.output_length if self.deg else math.floor(R_t[0])
        
        for i in range(1, output_length+1):
            output = 0
            for order in range(1, self.taylor_order+1):
                output += derivatives[order-1] * (i * self.frame_interval)**order / math.factorial(order)
            outputs.append(output + input[:, -1:, ...])
        
        outputs = torch.cat(outputs, dim=1) # -> [B, T', C, H, W]
        
        if not self.deg:
            return outputs, R_t
        else:
            return outputs
        
class interprator(nn.Module):
    def __init__(self, h_dim=768, sp_dim=16, ep=1.001):
        super().__init__()
        self.sp_dim = sp_dim
        self.ep = ep
        self.interprete = nn.Sequential(
            nn.Linear(h_dim, h_dim//2),
            nn.ReLU(),
            nn.Linear(h_dim//2, h_dim//4),
            nn.ReLU(),
            nn.Linear(h_dim//4, 1),
            )

    def forward(self, x, out_T): # [bs, sp_dim, embed_dim]

        # [bs, sp_dim, embed_dim] -> [bs, sp_dim, 1] -> [bs, sp_dim]
        t = self.interprete(x).reshape(-1, self.sp_dim)
        t_detached = t.detach()
        lower_adjust = torch.relu(-t_detached)  # Only positive when t < 0
        upper_adjust = torch.relu(t_detached - (out_T - 1))  # Only positive when t > out_T-1
        t = t + lower_adjust - upper_adjust
        t = torch.mean(t, dim=1) / 1e0  # (B,)
        t += self.ep
        return t # (B,)

class film(nn.Module):
    def __init__(self, h_dim=768, in_dim=1):
        super().__init__()
        self.condition_to_scale = nn.Sequential(
            nn.Linear(in_dim, h_dim//2),
            nn.ReLU(),
            nn.Linear(h_dim//2, h_dim),
            )

        self.condition_to_shift = nn.Sequential(
            nn.Linear(in_dim, h_dim//2),
            nn.ReLU(),
            nn.Linear(h_dim//2, h_dim),
            )

    def forward(self, x, t):
        scale = self.condition_to_scale(t[..., None])  # (..., C)
        shift = self.condition_to_shift(t[..., None])  # (..., C)
        
        if x.dim() == 3:
            scale = rearrange(scale, 'b c -> b 1 c')
            shift = rearrange(shift, 'b c -> b 1 c')
        elif x.dim() == 5:
            scale = rearrange(scale, 't c -> 1 t 1 1 c')
            shift = rearrange(shift, 't c -> 1 t 1 1 c')
        
        y = x * scale + shift
        return x + y

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    pos = pos.view(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)
    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb
def get_1d_sincos_pos_embed(embed_dim, length):
    return torch.unsqueeze(
        get_1d_sincos_pos_embed_from_grid(
            embed_dim, torch.arange(length, dtype=torch.float32)
        ),
        0,
    )

def get_2d_sincos_pos_embed(embed_dim, grid_size, *, flatten: bool = False):
    """
    flatten=True -> (1, H*W, D)
    flatten=False  -> (1, H, W, D)
    """
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) # (H*W, D/2)
        emb = torch.cat([emb_h, emb_w], dim=1) 
        return emb

    H, W = grid_size
    grid_h = torch.arange(grid_size[0], dtype=torch.float32)
    grid_w = torch.arange(grid_size[1], dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij') # here w goes first
    grid = torch.stack([grid_h, grid_w], dim=0).reshape(2, 1, H, W)

    grid = grid.reshape(2, 1, grid_size[0], grid_size[1])

    pos = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if not flatten:
        return pos.view(H, W, embed_dim).unsqueeze(0) # (1, H, W, D)
    else:
        return pos.unsqueeze(0) # (1, H*W, D)
t_emb_init = get_1d_sincos_pos_embed
s_emb_init = get_2d_sincos_pos_embed
def t_series(IP, frame_interval):
    t_seq = [0.0]
    for i in range(IP - 1):
        t_seq.append(-i * frame_interval)
    t_seq.reverse()
    t_seq = torch.tensor(t_seq)
    return t_seq # (in_T,)


if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = np.random.random_sample((1, 4, 4, 128, 384))
    x = torch.tensor(x).float().to(device)

    model = TANTE(
        in_T=4,
        attn_axes="THW  THW  THW",
        taylor_order = 1,
        frame_interval = 0.5,
        patch_scale = 8,
        deg = True,
    )

    model = model.to(device)

    x = model(x, 5)
    #x, t = model(x, 5)

    print(x.shape)
    #print(x.shape, t.shape, t)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nontrainable_params = total_params - trainable_params

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {nontrainable_params:,}")
    print(f"Approx. model size (fp32, trainable only): {trainable_params * 4 / 1024 / 1024:.2f} MB")