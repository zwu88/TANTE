# @author: Zhikai Wu, Jan. 2025, Singapore

import logging
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import os
import logging
import os.path as osp
import torch
from torchinfo import summary

import wandb

from data import TanteDataModule
from trainer import Trainer, R_Trainer
from utils import set_seed_device, set_ckpt

logger = logging.getLogger("Quantization")
logger.setLevel(level=logging.DEBUG)

@hydra.main(version_base=None, config_path="configs", config_name="tante")
def main(cfg: DictConfig):
    cfg, checkpoint_folder = set_ckpt(cfg, choose = 'recent')
    print(OmegaConf.to_yaml(cfg))

    device = set_seed_device(cfg.seed)

    logger.info(f"Instantiate datamodule {cfg.data._target_}")
    datamodule: TanteDataModule = instantiate(cfg.data)
    dset_metadata = datamodule.train_dataset.metadata
    print(dset_metadata)

    logger.info(f"Instantiate model {cfg.model._target_}",)
    model: torch.nn.Module = instantiate(cfg.model, dset_metadata=dset_metadata)
    model_summary = summary(model, depth=5, verbose=0)
    logger.info("\n%s", model_summary)
    model = model.to(device)

    logger.info(f"Instantiate optimizer {cfg.optimizer._target_}")
    optimizer: torch.optim.Optimizer = instantiate(
        cfg.optimizer, params=model.parameters()
    )

    logger.info(f"Instantiate learning rate scheduler {cfg.lr_scheduler._target_}")
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = instantiate(
        cfg.lr_scheduler,
        optimizer=optimizer,
        max_epochs=cfg.trainer.max_epoch,
        warmup_start_lr=cfg.optimizer.lr * 0.1,
        eta_min=cfg.optimizer.lr * 0.1,
    )

    logger.info(f"Instantiate trainer {cfg.trainer._target_}")
    trainer: Trainer = instantiate(
        cfg.trainer,
        checkpoint_folder=checkpoint_folder,
        model=model,
        datamodule=datamodule,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    with open(osp.join(checkpoint_folder, "extended_config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    wandb_logged_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb_logged_cfg["checkpoint_folder"] = checkpoint_folder
    wandb.init(
        # mode="offline",
        dir=checkpoint_folder,
        project=cfg.wandb_project_name,
        group=f"{cfg.data.dataset_name}",
        config=wandb_logged_cfg,
        name=cfg.experiment,
        resume=True,
    )
    trainer.train()
    wandb.finish()

    

if __name__ == "__main__":
    main()
