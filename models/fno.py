import torch
import torch.nn as nn
import numpy as np
from neuralop.models import FNO as neuralop_FNO
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from data.dataset import TanteMetadata


class NeuralOpsCheckpointWrapper(neuralop_FNO):
    """
    Quick wrapper around neural operator's model to apply checkpointing
    for really big inputs.
    """

    def __init__(self, *args, **kwargs):
        super(NeuralOpsCheckpointWrapper, self).__init__(*args, **kwargs)
        if "gradient_checkpointing" in kwargs:
            self.gradient_checkpointing = kwargs["gradient_checkpointing"]

    def optional_checkpointing(self, layer, *inputs, **kwargs):
        if self.gradient_checkpointing:
            return checkpoint(layer, *inputs, use_reentrant=False, **kwargs)
        else:
            return layer(*inputs, **kwargs)

    def forward(self, x: torch.Tensor, output_shape=None, **kwargs):
        """TFNO's forward pass

        Args:
            x: Input tensor
            output_shape: {tuple, tuple list, None}, default is None
                Gives the option of specifying the exact output shape for odd shaped inputs.
                * If None, don't specify an output shape
                * If tuple, specifies the output-shape of the **last** FNO Block
                * If tuple list, specifies the exact output-shape of each FNO Block
        """

        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]

        x = self.optional_checkpointing(self.lifting, x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            self.optional_checkpointing(
                self.fno_blocks, x, layer_idx, output_shape=output_shape[layer_idx]
            )

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.optional_checkpointing(self.projection, x)

        return x


class FNO(nn.Module):
    def __init__(
        self,
        in_T: int,
        dset_metadata: TanteMetadata, #
        modes1: int = 16,
        modes2: int = 16,
        modes3: int = 16,
        hidden_channels: int = 64,
        gradient_checkpointing: bool = False,
    ):
        super(FNO, self).__init__()
        n_channel = dset_metadata.n_fields
        dim_in = n_channel * in_T
        dim_out = n_channel
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.hidden_channels = hidden_channels
        self.model = None
        self.initialized = False
        self.n_spatial_dims = dset_metadata.n_spatial_dims if dset_metadata else 2
        self.gradient_checkpointing = gradient_checkpointing

        if self.n_spatial_dims == 2:
            self.n_modes = (self.modes1, self.modes2)
        elif self.n_spatial_dims == 3:
            self.n_modes = (self.modes1, self.modes2, self.modes3)

        self.model = NeuralOpsCheckpointWrapper(
            n_modes=self.n_modes,
            in_channels=self.dim_in,
            out_channels=self.dim_out,
            hidden_channels=self.hidden_channels,
            gradient_checkpointing=gradient_checkpointing,
        )

    def forward(self, input) -> torch.Tensor:
        x = rearrange(input, "b t c ... -> b (t c) ...") 
        x = self.model(x)
        x = rearrange(x, "b c ... -> b 1 c ...") 
        return x

if __name__=="__main__":
    x = np.random.random_sample((3, 10, 99, 99))

    x = torch.tensor(x).float()#.to(device)

    model = FNO(10, 1, None)

    x = model(x)

    print(x.shape)
