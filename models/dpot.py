# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from data.dataset import TanteMetadata

import math
import logging
from torch.nn.modules.container import Sequential

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


ACTIVATION = {'gelu':nn.GELU(),'tanh':nn.Tanh(),'sigmoid':nn.Sigmoid(),'relu':nn.ReLU(),'leaky_relu':nn.LeakyReLU(0.1),'softplus':nn.Softplus(),'ELU':nn.ELU(),'silu':nn.SiLU()}


class AFNO2D(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    """
    def __init__(self, width = 32, num_blocks=8, channel_first = False,sparsity_threshold=0.01, modes = 32,hard_thresholding_fraction=1, hidden_size_factor=1, act='gelu'):
        super().__init__()
        assert width % num_blocks == 0, f"hidden_size {width} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = width
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.channel_first = channel_first
        self.modes = modes
        self.hidden_size_factor = hidden_size_factor
        self.scale = 1 / (self.block_size * self.block_size * self.hidden_size_factor)

        self.act = ACTIVATION[act]

        self.w1 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size))

    ### N, C, X, Y
    def forward(self, x, spatial_size=None):
        if self.channel_first:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)  # -> N, X, Y, C
        else:
            B, H, W, C = x.shape
        x_orig = x

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")

        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        # total_modes = H*W // 2 + 1
        kept_modes = self.modes

        o1_real[:, :kept_modes, :kept_modes] = self.act(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :kept_modes, :kept_modes] = self.act(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :kept_modes, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :kept_modes, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")



        x = x + x_orig
        if self.channel_first:
            x = x.permute(0, 3, 1, 2)     ### N, C, X, Y

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='gelu', drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = ACTIVATION[act]
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(
        self, 
        mixing_type = 'afno', 
        double_skip = True, 
        width = 32, 
        n_blocks = 4, 
        mlp_ratio=1., 
        channel_first = True,
        modes = 32, 
        drop=0., 
        drop_path=0., 
        act='gelu', 
        h=14, 
        w=8,
    ):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(8, width)
        self.width = width
        self.modes = modes
        self.act = ACTIVATION[act]

        if mixing_type == "afno":
            self.filter = AFNO2D(width = width, num_blocks=n_blocks, sparsity_threshold=0.01, channel_first = channel_first, modes = modes,
                                 hard_thresholding_fraction=1, hidden_size_factor=1, act=act)

        self.norm2 = torch.nn.GroupNorm(8, width)

        mlp_hidden_dim = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=mlp_hidden_dim, kernel_size=1, stride=1),
            self.act,
            nn.Conv2d(in_channels=mlp_hidden_dim, out_channels=width, kernel_size=1, stride=1),
        )

        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)

        x = x + residual

        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size=16, in_chans=3, embed_dim=768, out_dim=128,act='gelu'):
        super().__init__()
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.out_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.out_dim = out_dim
        self.act = ACTIVATION[act]

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            self.act,
            nn.Conv2d(embed_dim, out_dim, kernel_size=1, stride=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        return x


class TimeAggregator(nn.Module):
    def __init__(self, n_channels, n_timesteps, out_channels, type='mlp'):
        super(TimeAggregator, self).__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.out_channels = out_channels
        self.type = type
        if self.type == 'mlp':
            self.w = nn.Parameter(1/(n_timesteps * out_channels**0.5) *torch.randn(n_timesteps, out_channels, out_channels),requires_grad=True)   # initialization could be tuned
        elif self.type == 'exp_mlp':
            self.w = nn.Parameter(1/(n_timesteps * out_channels**0.5) *torch.randn(n_timesteps, out_channels, out_channels),requires_grad=True)   # initialization could be tuned
            self.gamma = nn.Parameter(2**torch.linspace(-10,10, out_channels).unsqueeze(0),requires_grad=True)  # 1, C
    ##  B, X, Y, T, C
    def forward(self, x):
        if self.type == 'mlp':
            x = torch.einsum('tij, ...ti->...j', self.w, x)
        elif self.type == 'exp_mlp':
            t = torch.linspace(0, 1, x.shape[-2]).unsqueeze(-1).to(x.device) # T, 1
            t_embed = torch.cos(t @ self.gamma)
            x = torch.einsum('tij,...ti->...j', self.w, x * t_embed)

        return x

class DPOT(nn.Module):
    def __init__(
        self, 
        in_T: int,
        dset_metadata: TanteMetadata, 
        patch_size=16, 
        mixing_type = 'afno',
        out_timesteps = 1, 
        n_blocks = 4, 
        embed_dim = 768, 
        out_layer_dim = 32, 
        depth = 12, 
        modes = 32,
        mlp_ratio=1., 
        n_cls = 12, 
        act='gelu', 
        time_agg='exp_mlp'
    ):
        super(DPOT, self).__init__()
        img_size = dset_metadata.spatial_resolution
        n_channel = dset_metadata.n_fields
        in_timesteps = in_T
        self.in_channels = n_channel
        self.out_channels = n_channel
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.n_blocks = n_blocks
        self.modes = modes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.mlp_ratio = mlp_ratio
        self.act = ACTIVATION[act]
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=self.in_channels + 3, embed_dim=self.out_channels * patch_size + 3, out_dim=embed_dim,act=act)
        self.latent_size = self.patch_embed.out_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.patch_embed.out_size[0], self.patch_embed.out_size[1]))
        self.time_agg = time_agg
        self.n_cls = n_cls

        h, w = img_size
        self.blocks = nn.ModuleList([
            Block(mixing_type=mixing_type,modes=modes,
                width=embed_dim, mlp_ratio=mlp_ratio, channel_first = True, n_blocks=n_blocks,double_skip=False, h=h, w=w,act = act)
            for i in range(depth)])

        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            self.act,
            nn.Linear(embed_dim, embed_dim),
            self.act,
            nn.Linear(embed_dim, n_cls)
        )

        self.time_agg_layer = TimeAggregator(self.in_channels, in_timesteps, embed_dim, time_agg)

        ### attempt load balancing for high resolution
        self.out_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=embed_dim, out_channels=out_layer_dim, kernel_size=patch_size, stride=patch_size),
            self.act,
            nn.Conv2d(in_channels=out_layer_dim, out_channels=out_layer_dim, kernel_size=1, stride=1),
            self.act,
            nn.Conv2d(in_channels=out_layer_dim, out_channels=self.out_channels * self.out_timesteps,kernel_size=1, stride=1)
        )

        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.mixing_type = mixing_type


    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=.002)    # .02
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def get_grid(self, x):
        batchsize, size_x, size_y = x.shape[0], x.shape[1], x.shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).to(x.device)
        return grid


    def get_grid_3d(self, x):
        batchsize, size_x, size_y, size_z = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).to(x.device).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).to(x.device).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).to(x.device).repeat([batchsize, size_x, size_y, 1, 1])

        grid = torch.cat((gridx, gridy, gridz), dim=-1)
        return grid


    ### in/out: B, X, Y, T, C
    def forward(self, x):
        x = rearrange(x, 'b t c x y -> b x y t c')
        B, _, _, T, _ = x.shape

        grid = self.get_grid_3d(x)
        x = torch.cat((x, grid), dim=-1).contiguous() # B, X, Y, T, C+3
        x = rearrange(x, 'b x y t c -> (b t) c x y')
        x = self.patch_embed(x)

        x = x + self.pos_embed

        x = rearrange(x, '(b t) c x y -> b x y t c', b=B, t=T)

        x = self.time_agg_layer(x)

        x = rearrange(x, 'b x y c -> b c x y')

        for blk in self.blocks:
            x = blk(x)

        cls_token = x.mean(dim=(2, 3), keepdim=False)
        cls_pred = self.cls_head(cls_token)

        x = self.out_layer(x).permute(0, 2, 3, 1)
        x = x.reshape(*x.shape[:3], self.out_timesteps, self.out_channels).contiguous()

        x = rearrange(x, 'b x y t c -> b t c x y')
        return x

    def extra_repr(self) -> str:

        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])
        named_modules = list(named_modules)

        string_repr = ''
        for p in self.named_parameters():
            name = p[0].split('.')[0]
            if name not in named_modules:
                string_repr = string_repr + '(' + name + '): ' \
                              + 'tensor(' + str(tuple(p[1].shape)) + ', requires_grad=' + str(
                    p[1].requires_grad) + ')\n'

        return string_repr


if __name__ == "__main__":
    # x = torch.rand(4, 20, 20, 100)
    # net = AFNO2D(in_timesteps=3, out_timesteps=1, n_channels=2, width=100, num_blocks=5)
    x = torch.rand(5, 384, 128, 7)
    net = DPOT(7, 11, None)
    y = net(x)
    print(y.shape)