#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Zhikai Wu, Jul. 2025, New Haven, CT
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
import math


from data.datamodule import (
    AbstractDataModule, 
    DefaultChannelsFirstFormatter, 
    DefaultChannelsLastFormatter,
)
# from trainer import complexity_metrics_torch
from .metrics import complexity_metrics_torch

logger = logging.getLogger(__name__)

def rt_analyse(rt):
    step = 0
    var = 0
    rt_avg = torch.mean(rt).item()
    step = len(rt)
    var = torch.std(rt, unbiased=True).item() if step>1 else 0
    return rt_avg, step, var

class R_Evaler:
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
        rt_eps: float = 0.5,
        rt_n: int = 2,
    ):
        params = locals()
        for k, v in params.items():
            if k != 'self' and not k.startswith('_'):  
                setattr(self, k, v)
        
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
        moving_batch, y_ref = formatter.process_input(batch)
        moving_batch = moving_batch[0].to(self.device)
        Rts = []
        y_preds = []
        cumulative_length = 0 
        start_time = time.time()
        while cumulative_length < self.n_steps_rollout:
            y_pred, rt = model(moving_batch, self.n_steps_rollout)
            cumulative_length += y_pred.shape[1]
            if cumulative_length < self.n_steps_rollout:
                moving_batch = torch.cat([moving_batch[:, y_pred.shape[1]:, ...], y_pred], dim=1) # [bs, T, c, h, w]
            y_preds.append(formatter.process_output(y_pred))
            Rts.append(rt)
        forward_time = time.time() - start_time
        y_pred_out = torch.cat(y_preds, dim=1)[:, :self.n_steps_rollout, ...]
        y_ref = y_ref.to(self.device)
        Rts = torch.cat(Rts, dim=0) # (m,)
        return y_pred_out, y_ref, Rts, forward_time


    def Eval(self, mode="common"):
        test_dataloader = self.datamodule.test_dataloader()
        if mode == "common":
            test_loss, std, RT, Step, time_used, summary_error, summary_rt= self.validation_loop(test_dataloader)
            logger.info(f"Test Loss: {test_loss}")
            logger.info(f"std:{std}")
            logger.info(f"rt: {RT}, Step: {Step}, Time used: {time_used}")
            logger.info(f"error: {summary_error}, rt: {summary_rt}")

    @torch.inference_mode()
    def validation_loop(self, dataloader: DataLoader) -> float:
        self.model.eval()
        validation_loss = 0.0
        loss_dict = {}
        count = 0
        rt_list = []
        step_list = []
        Seq_Loss1 = []
        Seq_Loss2 = []
        Seq_Loss3 = []
        Seq_Loss4 = []
        time_used = []
        with torch.autocast(self.device.type, enabled=self.enable_amp, dtype=self.amp_type):
            for i, batch in enumerate(tqdm.tqdm(dataloader)):
                y_pred, y_ref, rts, ftime = self.rollout_model(self.model, batch, self.formatter)
                assert (y_ref.shape == y_pred.shape), f"Mismatching shapes between reference {y_ref.shape} and prediction {y_pred.shape}"
                loss1 = self.eval_loss_fn1(y_pred, y_ref, None) 
                loss2 = self.eval_loss_fn2(y_pred, y_ref, None) 
                loss3 = self.eval_loss_fn3(y_pred, y_ref, None) 
                loss4 = self.eval_loss_fn4(y_pred, y_ref, None) 
                Seq_Loss1.append(loss1.mean().item())
                Seq_Loss2.append(loss3.mean().item())
                Seq_Loss3.append(loss2.mean().item())
                Seq_Loss4.append(loss4.mean().item())
                time_used.append(ftime)
                rt_list.append(torch.mean(rts).item())
                step_list.append(len(rts))

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

        RT = sum(rt_list)/len(rt_list)

        Step = sum(step_list)/len(step_list)

        time_used = sum(time_used)/len(time_used)
        def get_five_number_summary(data):
            data = np.array(data)
            return {
                'min': np.min(data),
                'q1': np.percentile(data, 25),
                'median': np.median(data),
                'q3': np.percentile(data, 75),
                'max': np.max(data)
            }
        summary_error = get_five_number_summary(Seq_Loss2)
        summary_rt = get_five_number_summary(rt_list)

        return validation_loss, Std_error, RT, Step, time_used, summary_error, summary_rt
