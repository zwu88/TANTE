# @author: Zhikai Wu, Jan. 2025, Singapore

from datetime import datetime
import os
import h5py
import scipy
import matplotlib.pyplot as plt
import numpy as np
import math, copy
import random
from tqdm import tqdm
import torch
import torch, gc
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import os.path as osp

def set_seed_device(seed=0xD3):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        #torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True  # ensures deterministic behavior
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high") 
        device = "cuda:0" 
    else:  
        device = "cpu"

    return device

def set_ckpt(cfg, choose = 'recent'):
    experiment_folder = os.path.join(cfg.root_path, "experiments", cfg.experiment)
    checkpoint_file = ""
    if osp.exists(experiment_folder):
        last_chpt = osp.join(experiment_folder, choose+".pt")
        if osp.isfile(last_chpt):
            checkpoint_file = last_chpt
    else:
        os.makedirs(experiment_folder)
    cfg.trainer.checkpoint_path = checkpoint_file
    cfg.evaler.checkpoint_path = checkpoint_file
    return cfg, experiment_folder
    