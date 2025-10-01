import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


class AttentionUNet(nn.Module):

    def __init__(
        self, 
        in_T,
        dset_metadata = None,
        depth = 4,
        out_T = 4,
    ):
        super(AttentionUNet, self).__init__()
        n_channel = dset_metadata.n_fields if dset_metadata else 5
        dim_in = n_channel * in_T
        dim_out = n_channel * out_T
        self.out_T = out_T
        self.depth = depth
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(dim_in, 64)
        self.Conv2 = ConvBlock(64, 128)
        if depth>=3:
            self.Conv3 = ConvBlock(128, 256)
        if depth>=4:
            self.Conv4 = ConvBlock(256, 512)
        if depth>=5:
            self.Conv5 = ConvBlock(512, 1024)

        if depth>=5:
            self.Up5 = UpConv(1024, 512)
            self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
            self.UpConv5 = ConvBlock(1024, 512)

        if depth>=4:
            self.Up4 = UpConv(512, 256)
            self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
            self.UpConv4 = ConvBlock(512, 256)  

        if depth>=3:
            self.Up3 = UpConv(256, 128)
            self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
            self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        self.Conv = nn.Conv2d(64, dim_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = rearrange(x, "b t c ... -> b (t c) ...")
        e1 = self.Conv1(x)

        d2, d3, d4 = None, None, None

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        if self.depth>=3:
            e3 = self.MaxPool(e2)
            e3 = self.Conv3(e3)

        if self.depth>=4:
            e4 = self.MaxPool(e3)
            e4 = self.Conv4(e4)

        if self.depth>=5:
            e5 = self.MaxPool(e4)
            e5 = self.Conv5(e5)
        ############
            d5 = self.Up5(e5)
            s4 = self.Att5(gate=d5, skip_connection=e4)
            d5 = torch.cat((s4, d5), dim=1) # concatenate attention-weighted skip connection with previous layer output
            d4 = self.UpConv5(d5)

        if self.depth>=4:
            d4 = d4 if d4!=None else e4
            d4 = self.Up4(d4)
            s3 = self.Att4(gate=d4, skip_connection=e3)
            d4 = torch.cat((s3, d4), dim=1)
            d3 = self.UpConv4(d4)

        if self.depth>=3:
            d3 = d3 if d3!=None else e3
            d3 = self.Up3(d3)
            s2 = self.Att3(gate=d3, skip_connection=e2)
            d3 = torch.cat((s2, d3), dim=1)
            d2 = self.UpConv3(d3)

        d2 = d2 if d2!=None else e2
        d2 = self.Up2(d2)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d1 = self.UpConv2(d2)

        out = self.Conv(d1)
        out = rearrange(out, "b (c t) ... -> b t c ...", t=self.out_T)

        return out

if __name__=="__main__":
    x = np.random.random_sample((4, 7, 5, 128, 384))

    x = torch.tensor(x).float()#.to(device)

    model = AttentionUNet(7, depth=4)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)

    x = model(x)

    print(x.shape)