"""
Colleted by Zhikai Wu from:

    McCabe et al. 2024, Multiple Physics Pretraining for Spatiotemporal Surrogate Models
    Source: https://github.com/PolymathicAI/multiple_physics_pretraining

If you use this implementation, please cite original work above.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange
from functools import partial
from timm.layers import DropPath

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ContinuousPositionBias1D(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.num_heads = n_heads
        self.cpb_mlp = nn.Sequential(nn.Linear(1, 512, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(512, n_heads, bias=False))
        
    def forward(self, h, h2, bc=0):
        dtype, device = self.cpb_mlp[0].weight.dtype, self.cpb_mlp[0].weight.device
        if bc == 0: # Edges are actual endpoints
            relative_coords = torch.arange(-(h-1), h, dtype=dtype, device=device) / (h-1)
        elif bc == 1: # Periodic boundary conditions - aka opposite edges touch
            relative_coords = torch.cat([torch.arange(1, h//2+1, dtype=dtype, device=device),
                    torch.arange(-(h//2-1), h//2+1, dtype=dtype, device=device),
                    torch.arange(-(h//2-1), 0, dtype=dtype, device=device)
            ])  / (h-1)

        coords = torch.arange(h, dtype=torch.float32, device=device)
        coords = coords[None, :] - coords[:, None]
        coords = coords + (h-1)

        rel_pos_model = 16 * torch.sigmoid(self.cpb_mlp(relative_coords[:, None]).squeeze())
        biases = rel_pos_model[coords.long()]
        return biases.permute(2, 0, 1).unsqueeze(0).contiguous()


class RelativePositionBias(nn.Module):
    def __init__(self, bidirectional=True, num_buckets=32, max_distance=128, n_heads=2):
        super(RelativePositionBias, self).__init__()
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=32):
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen, bc=0):
        """ Compute binned relative position bias """
        context_position = torch.arange(qlen, dtype=torch.long,
                                        device=self.relative_attention_bias.weight.device)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long,
                                       device=self.relative_attention_bias.weight.device)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        """
                   k
             0   1   2   3
        q   -1   0   1   2
            -2  -1   0   1
            -3  -2  -1   0
        """
        if bc == 1:
            thresh = klen // 2
            relative_position[relative_position < -thresh] = relative_position[relative_position < -thresh] % thresh
            relative_position[relative_position > thresh] = relative_position[relative_position > thresh] % -thresh
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, qlen, klen)
        return values

    def forward(self, qlen, klen, bc=0):
        return self.compute_bias(qlen, klen, bc)  # shape (1, num_heads, qlen, klen)


class MLP(nn.Module):
    def __init__(self, hidden_dim, exp_factor=4.):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim * exp_factor))
        self.fc2 = nn.Linear(int(hidden_dim * exp_factor), hidden_dim)
        self.act = nn.GELU()
        
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

### Space utils
class RMSInstanceNorm2d(nn.Module):
    def __init__(self, dim, affine=True, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim)) # Forgot to remove this so its in the pretrained weights
    
    def forward(self, x):
        std, mean = torch.std_mean(x, dim=(-2, -1), keepdims=True)
        x = (x) / (std + self.eps)
        if self.affine:
            x = x * self.weight[None, :, None, None]  
        return x

class SubsampledLinear(nn.Module):
    """
    Cross between a linear layer and EmbeddingBag - takes in input 
    and list of indices denoting which state variables from the state
    vocab are present and only performs the linear layer on rows/cols relevant
    to those state variables
    
    Assumes (... C) input
    """
    def __init__(self, dim_in, dim_out, subsample_in=True):
        super().__init__()
        self.subsample_in = subsample_in
        self.dim_in = dim_in
        self.dim_out = dim_out
        temp_linear = nn.Linear(dim_in, dim_out)
        self.weight = nn.Parameter(temp_linear.weight).to(device)
        self.bias = nn.Parameter(temp_linear.bias).to(device)
    
    def forward(self, x, labels):
        # Note - really only works if all batches are the same input type
        # labels = labels[0] # Figure out how to handle this for normal batches later
        label_size = len(labels)
        if self.subsample_in:
            scale = (self.dim_in / label_size)**.5 # Equivalent to swapping init to correct for given subsample of input
            x = scale * F.linear(x, self.weight[:, labels], self.bias)
        else:
            x = F.linear(x, self.weight[labels], self.bias[labels])
        return x

class hMLP_stem(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=(16,16), in_chans=3, embed_dim =768):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.in_proj = torch.nn.Sequential(
            *[nn.Conv2d(in_chans, embed_dim//4, kernel_size=4, stride=4, bias=False),
            RMSInstanceNorm2d(embed_dim//4, affine=True),
            nn.GELU(),
            nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2, bias=False),
            RMSInstanceNorm2d(embed_dim//4, affine=True),
            nn.GELU(),
            nn.Conv2d(embed_dim//4, embed_dim, kernel_size=2, stride=2, bias=False),
            RMSInstanceNorm2d(embed_dim, affine=True),
            ]
            )
    
    def forward(self, x):
        x = self.in_proj(x)
        return x
    
class hMLP_output(nn.Module):
    """ Patch to Image De-bedding
    """
    def __init__(self, patch_size=(16,16), out_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.out_proj = torch.nn.Sequential(
            *[nn.ConvTranspose2d(embed_dim, embed_dim//4, kernel_size=2, stride=2, bias=False),
            RMSInstanceNorm2d(embed_dim//4, affine=True),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2, bias=False),
            RMSInstanceNorm2d(embed_dim//4, affine=True),
            nn.GELU(),
            ])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        out_head = nn.ConvTranspose2d(embed_dim//4, out_chans, kernel_size=4, stride=4).to(device)
        
        self.out_kernel = nn.Parameter(out_head.weight).to(device)
        self.out_bias = nn.Parameter(out_head.bias).to(device)
    
    def forward(self, x, state_labels):
        x = self.out_proj(x)#.flatten(2).transpose(1, 2)
        out_kernel = self.out_kernel.to(device)
        out_bias = self.out_bias.to(device)
        x = F.conv_transpose2d(x, out_kernel[:, state_labels], out_bias[state_labels], stride=4)
        return x
    
class AxialAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8,  drop_path=0, layer_scale_init_value=1e-6, bias_type='rel'):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = RMSInstanceNorm2d(hidden_dim, affine=True)
        self.norm2 = RMSInstanceNorm2d(hidden_dim, affine=True)
        self.gamma_att = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)), 
                            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma_mlp = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)), 
                            requires_grad=True) if layer_scale_init_value > 0 else None
        
        self.input_head = nn.Conv2d(hidden_dim, 3*hidden_dim, 1)
        self.output_head = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.qnorm = nn.LayerNorm(hidden_dim//num_heads)
        self.knorm = nn.LayerNorm(hidden_dim//num_heads)
        if bias_type == 'none':
            self.rel_pos_bias = lambda x, y: None
        elif bias_type == 'continuous':
            self.rel_pos_bias = ContinuousPositionBias1D(n_heads=num_heads)
        else:
            self.rel_pos_bias = RelativePositionBias(n_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        self.mlp = MLP(hidden_dim)
        self.mlp_norm = RMSInstanceNorm2d(hidden_dim, affine=True)

    def forward(self, x):
        # input is t x b x c x h x w 
        B, C, H, W = x.shape
        input = x.clone()
        x = self.norm1(x)
        x = self.input_head(x)

        x = rearrange(x, 'b (he c) h w ->  b he h w c', he=self.num_heads)
        q, k, v = x.tensor_split(3, dim=-1)
        q, k = self.qnorm(q), self.knorm(k)

        # Do attention with current q, k, v matrices along each spatial axis then average results
        # X direction attention
        qx, kx, vx = map(lambda x: rearrange(x, 'b he h w c ->  (b h) he w c'), [q,k,v])
        # Functional doesn't return attention mask :(
        xx = F.scaled_dot_product_attention(qx.contiguous(), kx.contiguous(), vx.contiguous())
        xx = rearrange(xx, '(b h) he w c -> b (he c) h w', h=H)
        # Y direction attention 
        qy, ky, vy = map(lambda x: rearrange(x, 'b he h w c ->  (b w) he h c'), [q,k,v])

        xy = F.scaled_dot_product_attention(qy.contiguous(), ky.contiguous(), vy.contiguous())
        xy = rearrange(xy, '(b w) he h c -> b (he c) h w', w=W)
        # Combine
        x = (xx + xy) / 2
        x = self.norm2(x)
        x = self.output_head(x)
        x = self.drop_path(x*self.gamma_att[None, :, None, None]) + input

        # MLP
        input = x.clone()
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.mlp(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.mlp_norm(x)
        output = input + self.drop_path(self.gamma_mlp[None, :, None, None] * x)

        return output

class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8, drop_path=0, layer_scale_init_value=1e-6, bias_type='rel'):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.InstanceNorm2d(hidden_dim, affine=True)
        self.norm2 = nn.InstanceNorm2d(hidden_dim, affine=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)), 
                            requires_grad=True) if layer_scale_init_value > 0 else None
        self.input_head = nn.Conv2d(hidden_dim, 3*hidden_dim, 1)
        self.output_head = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.qnorm = nn.LayerNorm(hidden_dim//num_heads)
        self.knorm = nn.LayerNorm(hidden_dim//num_heads)
        if bias_type == 'none':
            self.rel_pos_bias = lambda x, y: None
        elif bias_type == 'continuous':
            self.rel_pos_bias = ContinuousPositionBias1D(n_heads=num_heads)
        else:
            self.rel_pos_bias = RelativePositionBias(n_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # input is t x b x c x h x w 
        T, B, C, H, W = x.shape
        input = x.clone()
        # Rearrange and prenorm
        x = rearrange(x, 't b c h w -> (t b) c h w')
        x = self.norm1(x)
        x = self.input_head(x) # Q, K, V projections
        # Rearrange for attention
        x = rearrange(x, '(t b) (he c) h w ->  (b h w) he t c', t=T, he=self.num_heads)
        q, k, v = x.tensor_split(3, dim=-1)
        q, k = self.qnorm(q), self.knorm(k)
        rel_pos_bias = self.rel_pos_bias(T, T)
        if rel_pos_bias is not None:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=rel_pos_bias) 
        else:
            x = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())
        # Rearrange after attention
        x = rearrange(x, '(b h w) he t c -> (t b) (he c) h w', h=H, w=W)
        x = self.norm2(x) 
        x = self.output_head(x)
        x = rearrange(x, '(t b) c h w -> t b c h w', t=T)
        output = self.drop_path(x*self.gamma[None, None, :, None, None]) + input
        return output

class SpaceTimeBlock(nn.Module):
    """
    Alternates spatial and temporal processing. Current code base uses
    1D attention over each axis. Spatial axes share weights.

    Note: MLP is in spatial block. 
    """
    def __init__(self, hidden_dim=768, num_heads=8, drop_path=0., space_override=None, time_override=None,
                    gradient_checkpointing=False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        if space_override is not None:
            self.spatial = space_override(drop_path=drop_path)
        else:
            self.spatial = AxialAttentionBlock(hidden_dim, num_heads, drop_path=drop_path)

        if time_override is not None:
            self.temporal = time_override(drop_path=drop_path)
        else:
            self.temporal = AttentionBlock(hidden_dim, num_heads, drop_path=drop_path)

    def forward(self, x):
        # input is t x b x c x h x w 
        T, B, C, H, W = x.shape

        # Time attention
        if self.gradient_checkpointing:
            # kwargs seem to need to be passed explicitly
            wrapped_temporal = partial(self.temporal)
            x = checkpoint(wrapped_temporal, x, use_reentrant=False)
        else:
            x = self.temporal(x) # Residual in block
        # Temporal handles the rearrange so still is t x b x c x h x w 

        # Now do spatial attention
        x = rearrange(x, 't b c h w -> (t b) c h w')
        if self.gradient_checkpointing:
            # kwargs seem to need to be passed explicitly 
            wrapped_spatial = partial(self.spatial)
            x = checkpoint(wrapped_spatial, x, use_reentrant=False)
        else:
            x = self.spatial(x) # Convnext has the residual in the block
        x = rearrange(x, '(t b) c h w -> t b c h w', t=T) 

        return x

class AViT(nn.Module):
    """
    Naive model that interweaves spatial and temporal attention blocks. Temporal attention 
    acts only on the time dimension. 

    Args:
        patch_size (tuple): Size of the input patch
        embed_dim (int): Dimension of the embedding
        processor_blocks (int): Number of blocks (consisting of spatial mixing - temporal attention)
        n_states (int): Number of input state variables.  
    """
    def __init__(
        self, 
        in_T,
        dset_metadata = None,
        out_steps: int = 4,
        patch_size=(16, 16), 
        embed_dim=768, 
        num_heads=12,
        processor_blocks=8,       
        override_block=None, 
        drop_path=.2
    ):
        super().__init__()
        n_states = dset_metadata.n_fields if dset_metadata else 11
        self.drop_path = drop_path
        self.dp = np.linspace(0, drop_path, processor_blocks)
        self.space_bag = SubsampledLinear(n_states, embed_dim//4)
        self.embed = hMLP_stem(patch_size=patch_size, in_chans=embed_dim//4, embed_dim=embed_dim)
        self.out_steps = out_steps

        # Default to factored spacetime block with default settings (space/time axial attention)
        if override_block is not None:
            inner_block = override_block
        else:
            inner_block = partial(SpaceTimeBlock, hidden_dim=embed_dim, num_heads=num_heads)
        self.blocks = nn.ModuleList([inner_block(drop_path=self.dp[i])
                                     for i in range(processor_blocks)])
        self.debed = hMLP_output(patch_size=patch_size, embed_dim=embed_dim, out_chans=n_states)

    def forward(self, x):
        x = rearrange(x, 'b t c h w -> t b c h w')
        T, B, C = x.shape[:3]
        state_labels = range(C)
        with torch.no_grad():
            data_std, data_mean = torch.std_mean(x, dim=(0, -2, -1), keepdims=True)
            data_std = data_std + 1e-7 # Orig 1e-7
        x = (x - data_mean) / (data_std)

        # Sparse proj
        x = rearrange(x, 't b c h w -> t b h w c')
        x = self.space_bag(x, state_labels)

        # Encode
        x = rearrange(x, 't b h w c -> (t b) c h w')
        x = self.embed(x)            
        x = rearrange(x, '(t b) c h w -> t b c h w', t=T)

        # Process
        for blk in self.blocks:
            x = blk(x)

        # Decode - It would probably be better to grab the last time here since we're only
        # predicting the last step, but leaving it like this for compatibility to causal masking
        x = rearrange(x, 't b c h w -> (t b) c h w')
        x = self.debed(x, state_labels)
        x = rearrange(x, '(t b) c h w -> t b c h w', t=T)

        # Denormalize 
        x = x * data_std + data_mean # All state labels in the batch should be identical

        x = x[-4:]

        x = rearrange(x, 't b c h w -> b t c h w')

        return x

if __name__ == '__main__':
    print(torch.cuda.is_available())
    model = AViT(
        in_T=4,
        out_steps=4,
        patch_size=(16, 16),
        processor_blocks=12,
        embed_dim=384,
        num_heads=6,
        ).cuda()
    
    T = 4
    bs = 16
    c = 11
    nx = 256
    ny = 256
    x = torch.randn(bs, T, c, nx, ny).cuda()
    print('xshape', x.shape)
    y = model(x)
    print('yshape', y.shape)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nontrainable_params = total_params - trainable_params

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {nontrainable_params:,}")
    print(f"Approx. model size (fp32, trainable only): {trainable_params * 4 / 1024 / 1024:.2f} MB")


