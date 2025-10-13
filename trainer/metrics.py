#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Zhikai Wu, Sep. 2025, Philadelphia, PA
# All rights reserved.
#
# This work was developed at Yale University.
#
# Licensed under the MIT License. 

import numpy as np
import torch
import torch.nn as nn
import statistics
import einops
from data.dataset import TanteMetadata

class Metric(nn.Module):

    def forward(self, *args, **kwargs):
        assert len(args) >= 3, "At least three arguments required (x, y, rt)"
        x, y, rt = args[:3]
        if len(args) >= 5:
            _, _, _, eps, n = args[:5]
        else:
            eps, n = 0.5, 2

        # Convert x and y to torch.Tensor if they are np.ndarray
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        assert isinstance(x, torch.Tensor), "x must be a torch.Tensor or np.ndarray"
        assert isinstance(y, torch.Tensor), "y must be a torch.Tensor or np.ndarray"


        Loss_spatial = self.eval(x, y, **kwargs)

        if rt is not None:
            Loss_rt = self.eval_rt(rt, eps, n, **kwargs)
            return Loss_spatial.mean() + Loss_rt
        else:
            return Loss_spatial

    @staticmethod
    def eval(self, x, y, **kwargs):
        raise NotImplementedError

    @staticmethod
    def eval_rt(self, rt, eps, n, **kwargs):
        raise NotImplementedError

class MSE(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        n_spatial_dims = tuple(range(-3, -1))
        return torch.mean((x - y) ** 2, dim=n_spatial_dims) #[bs, T, c]
    
    @staticmethod
    def eval_rt(
        rt,
        eps = 0.5,
        n = 2.0,
    ) -> torch.Tensor:

        beta1 = 5e-3
        beta2 = 1e-1
        rt_loss = 0
        rt_avg = torch.mean(rt)
        up = min(1+eps, 4)
        down = max(1+eps, 4)
        
        if rt_avg < up:
            rt_loss += beta1 * (up - rt_avg)**n
        if rt_avg > down:
            rt_loss += beta2 * (rt_avg - down)**n
        return rt_loss

class NMSE(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        eps: float = 1e-7,
        norm_mode: str = "norm",
    ) -> torch.Tensor:

        n_spatial_dims = tuple(range(-3, -1))
        if norm_mode == "norm":
            norm = torch.mean(y**2, dim=n_spatial_dims)
        elif norm_mode == "std":
            norm = torch.std(y, dim=n_spatial_dims) ** 2
        else:
            raise ValueError(f"Invalid norm_mode: {norm_mode}")
        return MSE.eval(x, y) / (norm + eps)

class L2RE(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        x = einops.rearrange(x, "B T H W C-> B (T H W) C")
        y = einops.rearrange(y, "B T H W C-> B (T H W) C")
        num = torch.linalg.vector_norm(x - y, dim=1)
        den = torch.linalg.vector_norm(y, dim=1) + eps
        return num / den


class NNMSE(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        eps: float = 1e-7,
        norm_mode: str = "norm",
    ) -> torch.Tensor:

        n_spatial_dims = tuple(range(-3, 0))
        if norm_mode == "norm":
            norm = torch.mean(y**2, dim=n_spatial_dims)
        elif norm_mode == "std":
            norm = torch.std(y, dim=n_spatial_dims) ** 2
        else:
            raise ValueError(f"Invalid norm_mode: {norm_mode}")
        return torch.mean(MSE.eval(x, y), dim=-1) / (norm + eps)

class RMSE(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        return torch.sqrt(MSE.eval(x, y))

class NRMSE(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        eps: float = 1e-7,
        norm_mode: str = "norm",
    ) -> torch.Tensor:
        return torch.sqrt(NMSE.eval(x, y, eps=eps, norm_mode=norm_mode))

class VMSE(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        return NMSE.eval(x, y, norm_mode="std")

class VRMSE(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        return NRMSE.eval(x, y, norm_mode="std")

import torch
import torch.fft
import math

def compute_spectral_entropy(tensor):
    B, T, H, W, C = tensor.shape
    tensor = (tensor - tensor.mean(dim=1, keepdim=True)) / (tensor.std(dim=1, keepdim=True) + 1e-10)
    tensor_fft = torch.fft.fftn(tensor, dim=[1])  # Perform FFT over time axis (T dimension)
    psd = (tensor_fft.conj() * tensor_fft).real
    total_power = psd.sum(dim=1, keepdim=True)  # Sum over the time axis to get the total power
    psd_normalized = psd / (total_power + 1e-10)  # Normalize to get the probability distribution
    spectral_entropy = -torch.sum(psd_normalized * torch.log(psd_normalized + 1e-10), dim=1)  # Add a small epsilon to avoid log(0)
    F = psd.size(1)
    spectral_entropy_norm = spectral_entropy / (math.log(F) + 1e-10)
    return torch.mean(spectral_entropy).item(), torch.mean(spectral_entropy_norm).item()

def compute_high_frequency_ratio(tensor, cutoff=[0.2, 0.5, 0.8]):
    B, T, H, W, C = tensor.shape
    tensor = (tensor - tensor.mean(dim=1, keepdim=True)) / (tensor.std(dim=1, keepdim=True) + 1e-10)
    tensor_fft = torch.fft.fftn(tensor, dim=[1])  # Perform FFT over time axis (T dimension)
    psd = (tensor_fft.conj() * tensor_fft).real
    total_power = psd.sum(dim=1, keepdim=True) 
    psd_normalized = psd / (total_power + 1e-10)
    num_freqs = psd.shape[1]
    hfrs = []
    for frequency_threshold in cutoff:
        high_freq_idx = int(frequency_threshold * num_freqs)
        high_freq_power = psd[:, high_freq_idx:].sum(dim=1) 
        hfr = high_freq_power / (total_power.squeeze() + 1e-10)
        hfrs.append(torch.mean(hfr).item())
    return hfrs

def complexity_metrics_torch(data: torch.Tensor,
                             cutoff: float = [0.2, 0.5, 0.8]):
    se, se_norm  = compute_spectral_entropy(data)
    hfr = compute_high_frequency_ratio(data, cutoff=cutoff)
    return {"spectral_entropy": (se, se_norm), "highfreq_ratio": hfr}


# ==================== 示例 ====================

if __name__ == "__main__":
    T, H, W, C = 32, 64, 64, 3
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    random_field = torch.randn(T, H, W, C, device=device)

    t = torch.linspace(0, 2*math.pi, T, device=device)
    x = torch.linspace(0, 2*math.pi, H, device=device)
    y = torch.linspace(0, 2*math.pi, W, device=device)
    Tm, Xm, Ym = torch.meshgrid(t, x, y, indexing="ij")
    smooth  = torch.sin(Tm) + torch.sin(Xm) + torch.sin(Ym)  # (T,H,W)
    smooth  = smooth[..., None].repeat(1, 1, 1, C)           # → (T,H,W,C)

    print("随机场:", complexity_metrics_torch(random_field))
    print("平滑场:", complexity_metrics_torch(smooth))

