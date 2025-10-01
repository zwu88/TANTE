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


from data import TanteDataModule
from trainer import Evaler, R_Evaler
from utils import set_seed_device, set_ckpt

logger = logging.getLogger("TANTE")
logger.setLevel(level=logging.DEBUG)

@hydra.main(version_base=None, config_path="configs", config_name="tante")
def main(cfg: DictConfig):
    cfg.data.eval_steps_output = cfg.evaler.n_steps_rollout
    # cfg.experiment = " "
    cfg, checkpoint_folder = set_ckpt(
        cfg,
        choose="recent", # options: "best", "recent"
    )

    device = set_seed_device(cfg.seed)

    logger.info(f"Instantiate datamodule {cfg.data._target_}")
    datamodule: TanteDataModule = instantiate(cfg.data)
    dset_metadata = datamodule.train_dataset.metadata
    print(dset_metadata)

    logger.info(f"Instantiate model {cfg.model._target_}",)
    model: torch.nn.Module = instantiate(cfg.model, dset_metadata=dset_metadata)
    # model_summary = summary(model, depth=5, verbose=0)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("\n%s", total_params)
    model = model.to(device)

    logger.info(f"Instantiate trainer {cfg.trainer._target_}")

    

    trainer: Evaler = instantiate(
        cfg.evaler,
        checkpoint_folder=checkpoint_folder,
        model=model,
        datamodule=datamodule,
        batch_size=cfg.data.batch_size,
    )

    trainer.Eval(mode="common")


if __name__ == "__main__":
    main()
