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

import matplotlib.pyplot as plt
import numpy as np


from data.datamodule import (
    AbstractDataModule, 
    DefaultChannelsFirstFormatter, 
    DefaultChannelsLastFormatter,
)

logger = logging.getLogger(__name__)


def generate_and_extract_coords(y_ref, M):
    B, T, H, W, C = y_ref.shape
    
    h_indices = torch.arange(H, device=y_ref.device)
    w_indices = torch.arange(W, device=y_ref.device)
    h_grid, w_grid = torch.meshgrid(h_indices, w_indices, indexing='ij')
    all_indices = torch.stack([h_grid.flatten(), w_grid.flatten()], dim=-1)  # (H*W, 2)

    indices = torch.randperm(H * W, device=y_ref.device)[:M]
    selected_indices = all_indices[indices]  # (M, 2)
    
    h_coords = selected_indices[:, 0].float() / (H - 1)
    w_coords = selected_indices[:, 1].float() / (W - 1)
    coords = torch.stack([h_coords, w_coords], dim=-1)  # (M, 2)

    h_idx = selected_indices[:, 0]  # (M,)
    w_idx = selected_indices[:, 1]  # (M,)
    
    # Create proper indexing arrays
    b_idx = torch.arange(B, device=y_ref.device)[:, None, None]  # (B, 1, 1)
    t_idx = torch.arange(T, device=y_ref.device)[None, :, None]  # (1, T, 1)
    h_idx_broadcast = h_idx[None, None, :]  # (1, 1, M)
    w_idx_broadcast = w_idx[None, None, :]  # (1, 1, M)
    
    # Use advanced indexing
    extracted_points = y_ref[
        b_idx,  # (B, 1, 1)
        t_idx,  # (1, T, 1) 
        h_idx_broadcast, # (1, 1, M)
        w_idx_broadcast, # (1, 1, M)
        :    # C dimension
    ]
    
    return coords, extracted_points


class Trainer:
    def __init__(
        self,
        checkpoint_folder: str,
        formatter: str,
        model: torch.nn.Module,
        datamodule: AbstractDataModule, 
        optimizer: torch.optim.Optimizer,
        train_loss_fn: Callable,
        eval_loss_fn: Callable,
        max_epoch: int,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device=torch.device("cuda"),
        is_distributed: bool = False,
        enable_amp: bool = False,
        amp_type: str = "float16",  
        checkpoint_path: str = "",
        n_steps_output: int = 1, 
        n_steps_rollout: int = 8,
        rt_eps: float = 0.5,
        rt_n: int = 2,
        cvit: bool = False,
        num_query_points: int = 1024,
    ):
        params = locals()
        for k, v in params.items():
            if k != 'self' and not k.startswith('_'):  
                setattr(self, k, v)
        self.starting_epoch = 1  
        
        self.amp_type = torch.bfloat16 if amp_type == "bfloat16" else torch.float16
        self.grad_scaler = torch.GradScaler(
            self.device.type, enabled=enable_amp and amp_type != "bfloat16"
        )
        self.best_val_loss = None
        self.starting_val_loss = float("inf")
        self.dset_metadata = self.datamodule.train_dataset.metadata
        if formatter == "channels_first_default":
            self.formatter = DefaultChannelsFirstFormatter(self.dset_metadata)
        elif formatter == "channels_last_default":
            self.formatter = DefaultChannelsLastFormatter(self.dset_metadata)
        if len(checkpoint_path) > 0:
            self.load_checkpoint(checkpoint_path)

    def save_model(self, epoch: int, validation_loss: float, output_path: str):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dit": self.optimizer.state_dict(),
                "validation_loss": validation_loss,
                "best_validation_loss": self.best_val_loss,
            },
            output_path,
        )

    def load_checkpoint(self, checkpoint_path: str):
        """Load the model checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        if self.model is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dit"])
        self.best_val_loss = checkpoint["best_validation_loss"]
        self.starting_val_loss = checkpoint["validation_loss"]
        self.starting_epoch = (checkpoint["epoch"] + 1)
        if self.lr_scheduler:
            for i in range(self.starting_epoch-1):
                self.lr_scheduler.step()


    def rollout_model(self, model, batch, formatter, mode="train"):
        n_steps = self.n_steps_output if mode == "train" else self.n_steps_rollout
        moving_batch, y_ref = formatter.process_input(batch)
        moving_batch = moving_batch[0].to(self.device)
        y_preds = []
        cumulative_length = 0
        while cumulative_length < n_steps:
            y_pred = model(moving_batch)
            cumulative_length += y_pred.shape[1]
            if cumulative_length < n_steps:
                moving_batch = torch.cat([moving_batch[:, y_pred.shape[1]:, ...], y_pred], dim=1) # [bs, T, c, h, w]
            y_preds.append(formatter.process_output(y_pred))
        y_pred_out = torch.cat(y_preds, dim=1)
        y_pred_out = y_pred_out[:, :n_steps, ...]
        y_ref = y_ref.to(self.device)
        return y_pred_out, y_ref

    def rollout_cvit(self, model, batch, formatter, mode="train"):
        moving_batch, y_ref = formatter.process_input(batch)
        moving_batch = moving_batch[0].to(self.device)

        # create coords, process y_ref (b, t, h, w, d) -> (b, t, n, d)
        coords, y_ref = generate_and_extract_coords(y_ref, self.num_query_points)
        coords = coords.to(self.device)
        
        y_pred = model(moving_batch, coords) # -> (b, t, n, d)
        
        y_ref = y_ref.to(self.device)
        return y_pred, y_ref # -> (b, t, n, d)

    def train_one_epoch(self, epoch: int, dataloader: DataLoader) -> float:
        self.model.train()
        epoch_loss = 0.0
        train_logs = {}
        start_time = time.time()  
        batch_start = time.time()
        for i, batch in enumerate(dataloader):
            with torch.autocast(self.device.type, enabled=self.enable_amp, dtype=self.amp_type):
                batch_time = time.time() - batch_start
                if self.cvit:
                    y_pred, y_ref = self.rollout_cvit(self.model, batch, self.formatter, "train")
                else:
                    y_pred, y_ref = self.rollout_model(self.model, batch, self.formatter, "train")
                forward_time = time.time() - batch_start - batch_time
                assert (y_ref.shape == y_pred.shape), f"Mismatching shapes between reference {y_ref.shape} and prediction {y_pred.shape}"
                loss = self.train_loss_fn(y_pred, y_ref, None).mean()
            
            torch.cuda.empty_cache()
            with torch.amp.autocast('cuda', enabled=True):
                self.grad_scaler.scale(loss).backward(create_graph=False)

            # self.grad_scaler.scale(loss).backward()

            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad()
            epoch_loss += loss.item() / len(dataloader)
            backward_time = time.time() - batch_start - forward_time - batch_time
            total_time = time.time() - batch_start
            print(f"Epoch {epoch}, Batch {i+1}/{len(dataloader)}: loss {loss.item()}, forward time {forward_time}")
            batch_start = time.time()
        train_logs["time_per_train_iter"] = (time.time() - start_time) / len(dataloader)
        train_logs["train_loss"] = epoch_loss
        if self.lr_scheduler:
            self.lr_scheduler.step()
            train_logs["lr"] = self.lr_scheduler.get_last_lr()[-1]
        return epoch_loss, train_logs
   
    @torch.inference_mode()
    def validation_loop(self, dataloader: DataLoader, epoch: int = 0) -> float:
        self.model.eval()
        validation_loss = 0.0
        loss_dict = {}
        Seq_Loss = 0
        with torch.autocast(self.device.type, enabled=self.enable_amp, dtype=self.amp_type):
            for i, batch in enumerate(tqdm.tqdm(dataloader)):
                if self.cvit:
                    y_pred, y_ref = self.rollout_cvit(self.model, batch, self.formatter, "eval")
                else: 
                    y_pred, y_ref = self.rollout_model(self.model, batch, self.formatter, "eval")
                
                assert (y_ref.shape == y_pred.shape), f"Mismatching shapes between reference {y_ref.shape} and prediction {y_pred.shape}"
                loss = self.eval_loss_fn(y_pred, y_ref, None) 
                Seq_Loss += loss.mean().item()

        validation_loss = Seq_Loss / len(dataloader)

        with open(self.checkpoint_folder+'/saved_loss.txt', 'a') as f:
            f.write(str(validation_loss)+'\n')

        return validation_loss


    def train(self):
        train_dataloader = self.datamodule.train_dataloader()
        val_dataloder = self.datamodule.val_dataloader()
        val_loss = self.starting_val_loss

        for epoch in range(self.starting_epoch, self.max_epoch + 1):
            if self.is_distributed:
                train_dataloader.sampler.set_epoch(epoch)
            
            logger.info(f"Epoch {epoch}/{self.max_epoch}: starting training")
            train_loss, train_logs = self.train_one_epoch(epoch, train_dataloader)
            logger.info(f"Epoch {epoch}/{self.max_epoch}: avg training loss {train_loss}")
            wandb.log(train_logs, step=epoch)
            self.save_model(epoch, val_loss, os.path.join(self.checkpoint_folder, "recent.pt"))

            logger.info(f"Epoch {epoch}/{self.max_epoch}: starting validation")
            val_loss = self.validation_loop(val_dataloder, epoch=epoch)
            logger.info(f"Epoch {epoch}/{self.max_epoch}: avg validation loss {val_loss}")
            val_loss_dict = {"valid": val_loss}
            wandb.log(val_loss_dict, step=epoch)
            if self.best_val_loss is None or val_loss < self.best_val_loss:
                self.save_model(epoch, val_loss, os.path.join(self.checkpoint_folder, "best.pt"))
            
