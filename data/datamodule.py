# @author: Zhikai Wu, May 2025, Istanbul

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple
import torch
from torch.utils.data import DataLoader, DistributedSampler

from .dataset import TanteDataset
from .dataset import TanteMetadata

from einops import rearrange

logger = logging.getLogger(__name__)

class AbstractDataModule(ABC):
    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError

class TanteDataModule(AbstractDataModule):

    def __init__(
        self,
        base_path: str,
        dataset_name: str,
        batch_size: int,
        include_filters: List[str] = [],
        exclude_filters: List[str] = [],
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        eval_steps_output: int = 2,
        dt_stride: int = 1,
        world_size: int = 1,
        data_workers: int = 4,
        rank: int = 1,
        dataset_kws: Optional[Dict[Literal["train", "val", "test"],Dict[str, Any],]] = None,
        storage_kwargs: Optional[Dict] = None,
    ):
        self.train_dataset = TanteDataset(
            base_path=base_path,
            dataset_name=dataset_name,
            split_name="train",
            include_filters=include_filters,
            exclude_filters=exclude_filters,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            storage_options=storage_kwargs,
            dt_stride=dt_stride,
            **(dataset_kws["train"] if dataset_kws is not None and "train" in dataset_kws else {}),
        )
        self.val_dataset = TanteDataset(
            base_path=base_path,
            dataset_name=dataset_name,
            split_name="valid",
            include_filters=include_filters,
            exclude_filters=exclude_filters,
            n_steps_input=n_steps_input,
            n_steps_output=eval_steps_output,
            storage_options=storage_kwargs,
            dt_stride=dt_stride,
            **(dataset_kws["val"] if dataset_kws is not None and "val" in dataset_kws else {}),
        )
        
        self.test_dataset = TanteDataset(
            base_path=base_path,
            dataset_name=dataset_name,
            split_name="test",
            include_filters=include_filters,
            exclude_filters=exclude_filters,
            n_steps_input=n_steps_input,
            n_steps_output=eval_steps_output,
            storage_options=storage_kwargs,
            dt_stride=dt_stride,
            **(dataset_kws["test"] if dataset_kws is not None and "test" in dataset_kws else {}),
        )
        self.base_path = base_path
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.world_size = world_size
        self.data_workers = data_workers
        self.rank = rank

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def train_dataloader(self) -> DataLoader:
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            logger.debug(
                f"Use {sampler.__class__.__name__} "
                f"({self.rank}/{self.world_size}) for training data"
            )
        shuffle = sampler is None

        return DataLoader(
            self.train_dataset,
            num_workers=self.data_workers,
            pin_memory=True,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            logger.debug(
                f"Use {sampler.__class__.__name__} "
                f"({self.rank}/{self.world_size}) for validation data"
            )
        shuffle = sampler is None  
        return DataLoader(
            self.val_dataset,
            num_workers=self.data_workers,
            pin_memory=True,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            sampler=sampler,
        )

    def test_dataloader(self) -> DataLoader:
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                self.test_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
            logger.debug(
                f"Use {sampler.__class__.__name__} "
                f"({self.rank}/{self.world_size}) for test data"
            )
        return DataLoader(
            self.test_dataset,
            num_workers=self.data_workers,
            pin_memory=True,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            sampler=sampler,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.well_dataset_name} on {self.well_base_path}>"


class AbstractDataFormatter(ABC):
    def __init__(self, metadata: TanteMetadata):
        self.metadata = metadata

    @abstractmethod
    def process_input(self, data: Dict) -> Tuple:
        raise NotImplementedError

    @abstractmethod
    def process_output(self, output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class DefaultChannelsFirstFormatter(AbstractDataFormatter):
    def process_input(self, data: Dict) -> Tuple: ###
        x = data["input"]
        x = rearrange(x, "b t ... c -> b t c ...") 
        y = data["output"]
        return (torch.nan_to_num(x),), torch.nan_to_num(y)
    
    def process_output(self, output: torch.Tensor) -> torch.Tensor:
        return rearrange(output, "b t c ... -> b t ... c")

class DefaultChannelsLastFormatter(AbstractDataFormatter):
    def process_input(self, data: Dict) -> Tuple:
        x = data["input"]
        y = data["output"]
        return (torch.nan_to_num(x),), torch.nan_to_num(y)

    def process_output(self, output: torch.Tensor) -> torch.Tensor:
        return output

if __name__ == "__main__":
    datamodule = TanteDataModule(
        base_path = '/home/zw474/project/TANTE/dataset',
        dataset_name = 'turbulent_radiative_layer_2D',
        batch_size = 7,
        n_steps_input = 3,
        n_steps_output = 5,
        eval_steps_output = 8,
        dt_stride = 2,
        data_workers = 8,
    )
    train_loader = datamodule.train_dataloader()
    for i in train_loader:
        print(i["input"].shape, i["output"].shape)
        break
        