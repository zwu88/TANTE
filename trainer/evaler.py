#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Zhikai Wu, Jan 2025, Singapore
# All rights reserved.
#
# This work was developed at Yale University.
#
# Licensed under the MIT License. 


import logging
import os
import time
from typing import Callable, Optional
import statistics
import torch
import torch.distributed as dist
import tqdm
import wandb
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from einops import rearrange, repeat

import matplotlib.pyplot as plt
import numpy as np


from data.datamodule import (
    AbstractDataModule, 
    DefaultChannelsFirstFormatter, 
    DefaultChannelsLastFormatter,
)

logger = logging.getLogger(__name__)

def generate_chunked_coords_with_indices(H, W, L, device='cuda'):

    h_indices = torch.arange(H, device=device)
    w_indices = torch.arange(W, device=device)
    h_grid, w_grid = torch.meshgrid(h_indices, w_indices, indexing='ij')
    integer_indices = torch.stack([h_grid.flatten(), w_grid.flatten()], dim=-1)  # (H*W, 2)
    
    # Convert to normalized coordinates
    h_coords = integer_indices[:, 0].float() / (H - 1)
    w_coords = integer_indices[:, 1].float() / (W - 1)
    normalized_coords = torch.stack([h_coords, w_coords], dim=-1)  # (H*W, 2)
    
    # Divide into chunks
    total_coords = H * W
    coords_chunks = []
    indices_chunks = []
    
    for start_idx in range(0, total_coords, L):
        end_idx = min(start_idx + L, total_coords)
        coords_chunks.append(normalized_coords[start_idx:end_idx])
        indices_chunks.append(integer_indices[start_idx:end_idx])
    
    return coords_chunks, indices_chunks

def reconstruct_full_field(chunked_tensors, indices_chunks, H, W):

    B, T, _, C = chunked_tensors[0].shape
    reconstructed = torch.zeros(B, T, C, H, W, device=chunked_tensors[0].device, dtype=chunked_tensors[0].dtype)
    
    # Iterate through each chunk
    for tensor_chunk, indices_chunk in zip(chunked_tensors, indices_chunks):
        # indices_chunk contains the original (h, w) positions for this chunk
        h_indices = indices_chunk[:, 0]  # (N,)
        w_indices = indices_chunk[:, 1]  # (N,)
        tensor_chunk = rearrange(tensor_chunk, 'b t n c -> b t c n')
        
        # Place each point back to its original position
        reconstructed[:, :, :, h_indices, w_indices] = tensor_chunk
    
    return reconstructed

class Evaler:
    def __init__(
        self,
        checkpoint_folder: str,
        formatter: str,
        model: torch.nn.Module,
        datamodule: AbstractDataModule,
        eval_loss_fn1: Callable,
        eval_loss_fn2: Callable,
        eval_loss_fn3: Callable,
        eval_loss_fn4: Callable,
        device=torch.device("cuda"),
        enable_amp: bool = False,
        amp_type: str = "float16",  
        checkpoint_path: str = "",
        n_steps_rollout: int = 8,
        batch_size: int = 4,
        cvit: bool = False,
        num_query_points: int = 1024,
    ):
        params = locals()
        for k, v in params.items():
            if k != 'self' and not k.startswith('_'):  
                setattr(self, k, v)
        self.starting_epoch = 1  
        
        self.amp_type = torch.bfloat16 if amp_type == "bfloat16" else torch.float16
        self.dset_metadata = self.datamodule.train_dataset.metadata
        if formatter == "channels_first_default":
            self.formatter = DefaultChannelsFirstFormatter(self.dset_metadata)
        elif formatter == "channels_last_default":
            self.formatter = DefaultChannelsLastFormatter(self.dset_metadata)
        self.load_checkpoint(checkpoint_path)

    
    def load_checkpoint(self, checkpoint_path: str):
        """Load the model checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        if self.model is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])


    def rollout_model(self, model, batch, formatter):
        n_steps = self.n_steps_rollout
        moving_batch, y_ref = formatter.process_input(batch)
        moving_batch = moving_batch[0].to(self.device)
        y_preds = []
        cumulative_length = 0
        start_time = time.time()
        while cumulative_length < n_steps:
            y_pred = model(moving_batch)
            cumulative_length += y_pred.shape[1]
            if cumulative_length < n_steps:
                moving_batch = torch.cat([moving_batch[:, y_pred.shape[1]:, ...], y_pred], dim=1) # [bs, T, c, h, w]
            y_preds.append(formatter.process_output(y_pred))
        forward_time = time.time() - start_time
        y_pred_out = torch.cat(y_preds, dim=1)
        y_pred_out = y_pred_out[:, :n_steps, ...]
        y_ref = y_ref.to(self.device)
        return y_pred_out, y_ref, forward_time

    def rollout_cvit(self, model, batch, formatter):
        n_steps = self.n_steps_rollout
        moving_batch, y_ref = formatter.process_input(batch)
        moving_batch = moving_batch[0].to(self.device)
        y_preds = []
        cumulative_length = 0
        start_time = time.time()
        while cumulative_length < n_steps:

            coords_chunks, indices_chunks = generate_chunked_coords_with_indices(y_ref.shape[2], y_ref.shape[3], self.num_query_points)
            chunked_tensors = []
            for i in range(len(coords_chunks)):
                coords = coords_chunks[i].to(self.device)
                y_pred_i = model(moving_batch, coords)
                chunked_tensors.append(y_pred_i)
            y_pred = reconstruct_full_field(chunked_tensors, indices_chunks, y_ref.shape[2], y_ref.shape[3])

            cumulative_length += y_pred.shape[1]
            if cumulative_length < n_steps:
                moving_batch = torch.cat([moving_batch[:, y_pred.shape[1]:, ...], y_pred], dim=1) # [bs, T, c, h, w]
            y_preds.append(formatter.process_output(y_pred))
        forward_time = time.time() - start_time
        y_pred_out = torch.cat(y_preds, dim=1)
        y_pred_out = y_pred_out[:, :n_steps, ...]
        y_ref = y_ref.to(self.device)
        return y_pred_out, y_ref, forward_time

        # create coords, process y_ref (b, t, h, w, d) -> (b, t, n, d)
        coords, y_ref = generate_and_extract_coords(y_ref, self.num_query_points)
        coords = coords.to(self.device)

        start_time = time.time()
        y_pred = model(moving_batch, coords) # -> (b, t, n, d)
        forward_time = time.time() - start_time
        
        y_ref = y_ref.to(self.device)
        return y_pred, y_ref # -> (b, t, n, d)

    def Eval(self, mode="common"):
        test_dataloader = self.datamodule.test_dataloader()
        if mode == "common":
            test_loss, std, time_used = self.validation_loop(test_dataloader)
            logger.info(f"Test Loss: {test_loss}")
            logger.info(f"std:{std}")
            logger.info(f"Time used: {time_used}")
            
    @torch.inference_mode()
    def validation_loop(self, dataloader: DataLoader, epoch: int = 0) -> float:
        self.model.eval()
        validation_loss = 0.0
        loss_dict = {}
        Seq_Loss1 = []
        Seq_Loss2 = []
        Seq_Loss3 = []
        Seq_Loss4 = []
        time_used = []
        with torch.autocast(self.device.type, enabled=self.enable_amp, dtype=self.amp_type):
            for i, batch in enumerate(tqdm.tqdm(dataloader)):
                if self.cvit:
                    y_pred, y_ref, ftime = self.rollout_cvit(self.model, batch, self.formatter)
                else: 
                    y_pred, y_ref, ftime = self.rollout_model(self.model, batch, self.formatter)
                
                assert (y_ref.shape == y_pred.shape), f"Mismatching shapes between reference {y_ref.shape} and prediction {y_pred.shape}"
                loss1 = self.eval_loss_fn1(y_pred, y_ref, None) 
                loss2 = self.eval_loss_fn3(y_pred, y_ref, None) 
                loss3 = self.eval_loss_fn2(y_pred, y_ref, None) 
                loss4 = self.eval_loss_fn4(y_pred, y_ref, None) 
                Seq_Loss1.append(loss1.mean().item())
                Seq_Loss2.append(loss2.mean().item())
                Seq_Loss3.append(loss3.mean().item())
                Seq_Loss4.append(loss4.mean().item())
                time_used.append(ftime)

        validation_loss = [
            sum(Seq_Loss1) / len(dataloader),
            sum(Seq_Loss2) / len(dataloader),
            sum(Seq_Loss3) / len(dataloader),
            sum(Seq_Loss4) / len(dataloader),
        ]

        Std_error = [
            statistics.variance(Seq_Loss1),
            statistics.variance(Seq_Loss2),
            statistics.variance(Seq_Loss3),
            statistics.variance(Seq_Loss4),
        ]

        time_used = sum(time_used)/len(time_used)

        return validation_loss, Std_error, time_used
