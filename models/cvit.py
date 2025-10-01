"""
Translated by Zhikai Wu from:

    Wang et al. 2024, CViT: Continuous Vision Transformer for Operator Learning
    Source: https://github.com/PredictiveIntelligenceLab/cvit (Jax framework)

If you use this implementation, please cite original work above.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import einops
from einops import rearrange, repeat
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def get_2d_sincos_pos_embed(embed_dim, grid_size): 
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = torch.cat([emb_h, emb_w], dim=1)
        return emb

    grid_h = torch.arange(grid_size[0], dtype=torch.float32)
    grid_w = torch.arange(grid_size[1], dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')  # here w goes first
    grid = torch.stack([grid_h, grid_w], dim=0)

    grid = grid.reshape(2, 1, grid_size[0], grid_size[1])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    return torch.unsqueeze(pos_embed, 0)
    
class PatchEmbed(nn.Module):
    def __init__(
        self, 
        n_channel,
        patch_size = (1, 16, 16), 
        emb_dim = 768, 
        use_norm = False, 
        kernel_init = False, 
        layer_norm_eps = 1e-5,
    ):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size  # patch_size: (1, 16, 16)
        self.use_norm = use_norm

        self.conv = nn.Conv3d(
            in_channels=n_channel, 
            out_channels=emb_dim,
            kernel_size=(self.patch_size[0], self.patch_size[1], self.patch_size[2]),
            stride=(self.patch_size[0], self.patch_size[1], self.patch_size[2]),
        )

        if self.use_norm:
            self.layer_norm = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
    
    def forward(self, x):
        b, t, c, h, w = x.shape 

        x = rearrange(x, 'b t c h w -> b c t h w')
        x = self.conv(x)

        x = rearrange(x, 'b c t h w -> b t (h w) c')

        if self.use_norm:
            x = self.layer_norm(x)
        
        return x

class MlpBlock(nn.Module):
    def __init__(self, in_dim=256, dim=256, out_dim=256, kernel_init=True):
        super(MlpBlock, self).__init__()

        self.fc1 = nn.Linear(in_dim, dim) 
        self.fc2 = nn.Linear(dim, out_dim) 

        if kernel_init:
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class SelfAttnBlock(nn.Module):
    def __init__(
        self, 
        num_heads, 
        emb_dim, 
        mlp_ratio, 
        layer_norm_eps=1e-5
    ):
        super(SelfAttnBlock, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)

        self.layer_norm1 = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(emb_dim, eps=layer_norm_eps)

        self.mlp = MlpBlock(emb_dim, emb_dim * mlp_ratio, emb_dim)
    
    def forward(self, inputs):
        x = self.layer_norm1(inputs)
        
        x, _ = self.attn(x, x, x)
        x = x + inputs

        y = self.layer_norm2(x)
        
        y = self.mlp(y)

        return x + y 

class CrossAttnBlock(nn.Module):  
    def __init__(
        self, 
        num_heads, 
        emb_dim, 
        mlp_ratio, 
        layer_norm_eps=1e-5
    ):
        super(CrossAttnBlock, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)

        self.layer_norm1 = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(emb_dim, eps=layer_norm_eps)

        self.mlp = MlpBlock(emb_dim, emb_dim * mlp_ratio, emb_dim)
    
    def forward(self, q_inputs, kv_inputs):
        q = self.layer_norm1(q_inputs)
        kv = self.layer_norm2(kv_inputs)

        x, _ = self.attn(q, kv, kv)
        x = x + q_inputs 

        y = self.layer_norm2(x)
        
        y = self.mlp(y) 

        return x + y

class TimeAggregation(nn.Module):
    def __init__(
        self, 
        emb_dim, 
        depth, 
        num_heads=8, 
        num_latents=64, 
        mlp_ratio=1, 
        layer_norm_eps=1e-5
    ):
        super(TimeAggregation, self).__init__()
        
        self.emb_dim = emb_dim
        self.depth = depth 
        self.num_heads = num_heads
        self.num_latents = num_latents
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps
        
        self.latents = nn.Parameter(torch.randn(num_latents, emb_dim))  # (T', D)

        self.CrossAttnBlocks = nn.ModuleList([
            CrossAttnBlock(
                num_heads = num_heads, 
                emb_dim = emb_dim, 
                mlp_ratio = mlp_ratio, 
                layer_norm_eps = layer_norm_eps
            )
            for i in range(self.depth)
        ])
        
    def forward(self, x):  # (B, T, S, D)
        B, T, S, D = x.shape
        latents = repeat(self.latents, 't d -> b t d', b=B*S)
        x = rearrange(x, "b t s d -> (b s) t d")

        for i in range(self.depth):
            latents = self.CrossAttnBlocks[i](latents, x)
        
        latents = rearrange(latents, "(b s) t d -> b t s d", b=B, s=S)
        return latents

class Mlp(nn.Module):
    def __init__(
        self, 
        in_dim,
        num_layers, 
        hidden_dim, 
        out_dim, 
        kernel_init=True, 
        layer_norm_eps=1e-5
    ):
        super(Mlp, self).__init__()
        self.num_layers = num_layers

        self.dense_layers = nn.ModuleList([
            nn.Linear(in_features=hidden_dim if i>0 else in_dim, out_features=hidden_dim)
            for i in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim, eps=layer_norm_eps) for _ in range(num_layers)])

    def forward(self, inputs):
        x = inputs
        for i in range(self.num_layers):
            y = self.dense_layers[i](x)
            y = F.gelu(y)
            x = x + y
            x = self.layer_norms[i](x)

        x = self.output_layer(x)
        return x

t_emb_init = get_1d_sincos_pos_embed
s_emb_init = get_2d_sincos_pos_embed

# Need THW shape
class Encoder(nn.Module):
    def __init__(
        self, 
        n_channel,
        patch_size=(1, 16, 16), 
        emb_dim=256, 
        depth=3, 
        num_heads=8, 
        mlp_ratio=1, 
        out_dim=1, 
        layer_norm_eps=1e-5,
        THW_shape=(4, 128, 384),
    ):
        super(Encoder, self).__init__()
        self.depth = depth

        # Define layers
        self.patch_embed = PatchEmbed(n_channel, patch_size, emb_dim) 
        self.time_agg = TimeAggregation(
            num_latents=1,
            emb_dim=emb_dim,
            depth=2,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            layer_norm_eps=layer_norm_eps,
        )
        self.layer_norm = nn.LayerNorm(emb_dim, eps=layer_norm_eps)

        # Define position embeddings as variables
        t, h, w = THW_shape

        self.t_emb = nn.Parameter(t_emb_init(emb_dim, t // patch_size[0]))
        self.s_emb = nn.Parameter(s_emb_init(emb_dim, (h // patch_size[1], w // patch_size[2])))  # Position embeddings for spatial

        self.SelfAttnBlocks = nn.ModuleList([
            SelfAttnBlock(num_heads, emb_dim, mlp_ratio, layer_norm_eps)
            for i in range(depth)
        ])

    def forward(self, x):
        b, t, c, h, w = x.shape
        
        x = self.patch_embed(x)
        # (b, t, (h*w)/(16*16), emb_dim)

        t_emb = repeat(self.t_emb, '1 t c -> b t s c', b=b, s=x.shape[2])
        s_emb = repeat(self.s_emb, '1 s c -> b t s c', b=b, t=t)
        
        x = x + t_emb + s_emb

        x = self.time_agg(x)  # (b, t', s, emb_dim), where t' = 1
        x = self.layer_norm(x)

        x = rearrange(x, "b t s d -> b (t s) d") #(b, t's, d)

        for i in range(self.depth):
            x = self.SelfAttnBlocks[i](x)

        return x 

class FourierEmbs(nn.Module):
    def __init__(
        self, 
        embed_scale: float, 
        embed_dim: int,
        D: int = 2,
    ):
        super(FourierEmbs, self).__init__()
        self.embed_scale = embed_scale
        self.embed_dim = embed_dim

        self.kernel = nn.Parameter(torch.randn(D, embed_dim // 2)* embed_scale)  # (T', D)

    def forward(self, x):
        
        dot_product = torch.matmul(x, self.kernel)
        
        cos_part = torch.cos(dot_product)
        sin_part = torch.sin(dot_product)
        
        y = torch.cat([cos_part, sin_part], dim=-1)
        
        return y


class CViT(nn.Module):
    def __init__(
        self, 
        in_T,
        dset_metadata=None,
        out_steps=4,
        patch_size: tuple = (1, 16, 16), # (1, 16, 16)
        grid_size: tuple = (128, 128), # (128, 128)
        latent_dim: int = 256, # 512
        emb_dim: int = 256,
        depth: int = 3,
        num_heads: int = 8,
        dec_emb_dim: int = 256, 
        dec_num_heads: int = 8, 
        dec_depth: int = 1,
        num_mlp_layers: int = 1,
        mlp_ratio: int = 1,
        eps: float = 1e5, 
        layer_norm_eps: float = 1e-5,
        embedding_type: str = "grid", #"grid" | "fourier" | "mlp"
    ):
        super(CViT, self).__init__()

        n_channel = dset_metadata.n_fields if dset_metadata else 4
        self.T = in_T
        self.H, self.W = dset_metadata.spatial_resolution if dset_metadata else (128, 384)
        self.embedding_type = embedding_type
        self.eps = eps
        self.dec_depth = dec_depth
        self.out_steps = out_steps

        out_dim = n_channel * out_steps
        
        if embedding_type == "grid":
            n_x, n_y = grid_size
            self.latents = nn.Parameter(torch.randn(n_x * n_y, latent_dim))
            x = np.linspace(0, 1, n_x)
            y = np.linspace(0, 1, n_y)
            xx, yy = np.meshgrid(x, y, indexing="ij")
            grid = torch.tensor(np.hstack([xx.flatten()[:, None], yy.flatten()[:, None]]))
            grid = grid.to(dtype=self.latents.dtype)
            self.grid = nn.Parameter(grid)

            self.embedding = nn.Sequential(
                nn.Linear(latent_dim, dec_emb_dim),
                nn.LayerNorm(dec_emb_dim, eps=layer_norm_eps),
            )
        
        elif embedding_type == "fourier":
            self.embedding = nn.Sequential(
                FourierEmbs(embed_scale=2 * np.pi, embed_dim=dec_emb_dim),
            )

        elif embedding_type == "mlp":
            self.embedding = nn.Sequential(
                MlpBlock(2, dec_emb_dim, dec_emb_dim),
                nn.LayerNorm(dec_emb_dim, eps=layer_norm_eps),
            )

        self.Encoder = Encoder(
            n_channel=n_channel,
            patch_size=patch_size,
            emb_dim=emb_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            layer_norm_eps=layer_norm_eps,
            THW_shape=(self.T, self.H, self.W)
        )
        
        self.E2D = nn.Linear(emb_dim, dec_emb_dim)

        self.CrossAttnBlocks = nn.ModuleList([
            CrossAttnBlock(
                num_heads=dec_num_heads,
                emb_dim=dec_emb_dim,
                mlp_ratio=mlp_ratio,
                layer_norm_eps=layer_norm_eps,
            )
            for i in range(dec_depth)
        ])

        self.mlp = Mlp(
            in_dim=dec_emb_dim,
            num_layers=num_mlp_layers,
            hidden_dim=dec_emb_dim,
            out_dim=out_dim, 
            layer_norm_eps=layer_norm_eps,
        )

        self.norm1 = nn.LayerNorm(emb_dim, eps=layer_norm_eps)

        self.norm2 = nn.LayerNorm(dec_emb_dim, eps=layer_norm_eps)

    def forward(self, x, input_coords=None):
        b, t, c, h, w = x.shape
        if input_coords == None:
            coords = generate_coords(h, w) # (h * w, 2)
        else: coords = input_coords

        if self.embedding_type == "grid":
            d2 = ((coords[:, None, :] - self.grid[None, :, :]) ** 2).sum(dim=2)
            exp_term = torch.exp(-self.eps * d2)
            w = exp_term / exp_term.sum(dim=1, keepdim=True)
            coords = torch.einsum("ic,pi->pc", self.latents, w)  # (B*n, latent_dim)
            coords = self.embedding(coords)

        elif self.embedding_type == "fourier":
            coords = self.embedding(coords)

        elif self.embedding_type == "mlp":
            coords = self.embedding(coords)

        # (B, num_coords, dec_emb_dim)
        coords =repeat(coords, "n d -> b n d", b=b)

        # Encoder
        x = self.Encoder(x)  # (B, (T'*H*W//256), emb_dim)

        x = self.norm1(x)
        x = self.E2D(x)  # (B, (T'*H*W//256), dec_emb_dim)

        for i in range(self.dec_depth):
            x = self.CrossAttnBlocks[i](coords, x)  # (B, N, dec_emb_dim)

        x = self.norm2(x)
        x = self.mlp(x) # layer #(B, N, D)

        if input_coords == None:
            x = rearrange(x, 'b (h w) (t d) -> b t d h w', h=self.H, w=self.W, t=self.out_steps, d=c)
        else: 
            x = rearrange(x, 'b n (t d) -> b t n d', t=self.out_steps, d=c)

        return x


def generate_coords(h, w):
    """
    Returns:
        torch.Tensor: (num_query_points, 2) 
    """
    x_star = torch.linspace(0, 1, h, device=device)
    y_star = torch.linspace(0, 1, w, device=device)
    x_star, y_star = torch.meshgrid(x_star, y_star, indexing='ij')
    batch_coords = torch.stack([x_star.flatten(), y_star.flatten()], dim=-1)  # (h * w, 2)
    
    return batch_coords.to(device) # (h * w, 2) 


if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = np.random.random_sample((5, 4, 4, 128, 384))

    x = torch.tensor(x).float().to(device)

    model = CViT(
        in_T=4,
        out_steps=4,
        embedding_type='grid',
        patch_size=(1, 32, 32), 
        grid_size=(48, 48),
        latent_dim=256,
        emb_dim=512,
        depth=10,
        num_heads=8,
        dec_emb_dim=512,
        dec_num_heads=8,
        dec_depth=1,
        num_mlp_layers=1,
        mlp_ratio=2,
    )

    model = model.to(device)

    x = model(x)

    print(x.shape)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nontrainable_params = total_params - trainable_params

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {nontrainable_params:,}")
    print(f"Approx. model size (fp32, trainable only): {trainable_params * 4 / 1024 / 1024:.2f} MB")