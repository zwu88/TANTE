# @author: Zhikai Wu, May 2025, Istanbul

import itertools
import h5py as h5
import yaml
import numpy as np
import torch
import os
import fsspec

from einops import rearrange
from enum import Enum
from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    cast,
)

IO_PARAMS = {
    "fsspec_params": {
        "cache_type": "blockcache",  
        "block_size": 8 * 1024 * 1024,  
    },
    "h5py_params": {
            "page_buf_size": 8 * 1024 * 1024,
            "rdcc_nbytes": 8 * 1024 * 1024,
    },
}
######## For debug ########

def compute_windows(total_steps, n_steps_input, n_steps_output, dt_stride):
    elapsed_steps_per_sample = 1 + dt_stride * (n_steps_input + n_steps_output - 1)  # Number of steps needed for sample
    return max(0, total_steps - elapsed_steps_per_sample + 1)

@dataclass
class TanteMetadata:

    dataset_name: str
    n_spatial_dims: int
    spatial_resolution: Tuple[int, ...]
    field_names: Dict[int, List[str]]
    boundary_condition_types: List[str]
    n_files: int
    n_trajectories_per_file: List[int]
    n_steps_per_trajectory: List[int]
    n_fields: int


    @property
    def sample_shapes(self) -> Dict[str, List[int]]:
        return {
            "input_fields": [*self.spatial_resolution, self.n_fields],
            "output_fields": [*self.spatial_resolution, self.n_fields],
            "space_grid": [*self.spatial_resolution, self.n_spatial_dims],
        }

class TanteDataset(Dataset):

    def __init__(
        self,
        base_path: Optional[str] = "./dataset",
        dataset_name: Optional[str] = None,
        split_name: str = "train",
        include_filters: List[str] = [],
        exclude_filters: List[str] = [],
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        dt_stride: int = 1,
        cache_small: bool = True,
        max_cache_size: float = 1e9,
        min_std: float = 1e-4,
        storage_options: Optional[Dict] = None,
    ):
        super().__init__()
        self.data_path = os.path.join(base_path, dataset_name, "data", split_name)
        self.normalization_path = os.path.join(base_path, dataset_name, "stats.yaml")
        self.fs, _ = fsspec.url_to_fs(self.data_path, **(storage_options or {}))
        with self.fs.open(self.normalization_path, mode="r") as f:
            stats = yaml.safe_load(f)
        self.means = {field: torch.as_tensor(val) for field, val in stats["mean"].items()}
        self.stds = {field: torch.clip(torch.as_tensor(val), min=min_std) for field, val in stats["std"].items()}
        
        # Copy params
        params = locals()
        for k, v in params.items():
            if k != 'self' and not k.startswith('_'):  
                setattr(self, k, v)

        # Check the directory has hdf5 that meet our exclusion criteria
        sub_files = self.fs.glob(self.data_path + "/*.h5") + self.fs.glob(
            self.data_path + "/*.hdf5"
        )
        # Check filters - only use file if include_filters are present and exclude_filters are not
        if len(self.include_filters) > 0:
            retain_files = []
            for include_string in self.include_filters:
                retain_files += [f for f in sub_files if include_string in f]
            sub_files = retain_files
        if len(self.exclude_filters) > 0:
            for exclude_string in self.exclude_filters:
                sub_files = [f for f in sub_files if exclude_string not in f]
        assert len(sub_files) > 0, "No HDF5 files found in path {}".format(self.data_path)
        self.files_paths = sub_files
        self.files_paths.sort()
        self.caches = [{} for _ in self.files_paths]
        self.metadata = self._build_metadata()

    def _build_metadata(self):
        self.n_files = len(self.files_paths)
        self.n_trajectories_per_file = []
        self.n_steps_per_trajectory = []
        self.n_windows_per_trajectory = []
        self.file_index_offsets = [0]
        size_tuples = set()
        ndims = set()
        names = set()
        bcs = set()
        for index, file in enumerate(self.files_paths):
            with (
                self.fs.open(file, "rb", **IO_PARAMS["fsspec_params"]) as f,
                h5.File(f, "r", **IO_PARAMS["h5py_params"]) as _f,
            ):
                trajectories = int(_f.attrs["n_trajectories"])
                steps = _f["dimensions"]["time"].shape[-1]
                windows_per_trajectory = compute_windows(steps, self.n_steps_input, self.n_steps_output, self.dt_stride)
                assert windows_per_trajectory > 0, (f"{steps} steps is not enough steps for file {file} to allow {self.n_steps_input} input and {self.n_steps_output} output steps with a minimum stride of {self.dt_stride}")
                self.n_trajectories_per_file.append(trajectories)
                self.n_steps_per_trajectory.append(steps)
                self.n_windows_per_trajectory.append(windows_per_trajectory)
                self.file_index_offsets.append(
                    self.file_index_offsets[-1] + trajectories * windows_per_trajectory
                )
                size_tuple = [_f["dimensions"][d].shape[-1] for d in _f["dimensions"].attrs["spatial_dims"]]
                size_tuples.add(tuple(size_tuple))
                ndims.add(_f.attrs["n_spatial_dims"])
                names.add(_f.attrs["dataset_name"])
                # Check BCs
                for bc in _f["boundary_conditions"].keys():
                    bcs.add(_f["boundary_conditions"][bc].attrs["bc_type"])
                
                if index == 0:
                    self.field_names = {i: [] for i in range(3)}
                    for i in range(3):
                        ti = f"t{i}_fields"
                        ti_field_dims = ["".join(xyz) for xyz in itertools.product(_f["dimensions"].attrs["spatial_dims"], repeat=i)]
                        for field in _f[ti].attrs["field_names"]:
                            for dims in ti_field_dims:
                                field_name = f"{field}_{dims}" if dims else field
                                if _f[ti][field].attrs["time_varying"]:
                                    self.field_names[i].append(field_name)

        self.file_index_offsets[0] = -1
        self.files: List[h5.File | None] = [None for _ in self.files_paths]  
        self.len = self.file_index_offsets[-1]
        self.n_spatial_dims = int(ndims.pop()) 
        self.size_tuple = tuple(map(int, size_tuples.pop()))
        self.dataset_name = names.pop()
        # BCs
        self.num_bcs = len(bcs)  # Number of boundary condition type included in data
        self.bc_types = list(bcs)  # List of boundary condition types

        return TanteMetadata(
            dataset_name=self.dataset_name,
            n_spatial_dims=self.n_spatial_dims,
            spatial_resolution=self.size_tuple,
            field_names=self.field_names,
            boundary_condition_types=self.bc_types,
            n_files=self.n_files,
            n_trajectories_per_file=self.n_trajectories_per_file,
            n_steps_per_trajectory=self.n_steps_per_trajectory,
            n_fields=sum(map(len, self.field_names.values()))
        )

    def _open_file(self, file_ind: int):
        _file = h5.File(self.fs.open(self.files_paths[file_ind], "rb", **IO_PARAMS["fsspec_params"]),"r",**IO_PARAMS["h5py_params"])
        self.files[file_ind] = _file
    
    def _check_cache(self, cache: Dict[str, Any], name: str, data: Any):
        if self.cache_small and data.numel() < self.max_cache_size:
            cache[name] = data

    def _reconstruct_fields(self, file, cache, sample_idx, time_idx, n_steps, dt):
        variable_fields = {0: {}, 1: {}, 2: {}}
        for i, order_fields in enumerate(["t0_fields", "t1_fields", "t2_fields"]):
            field_names = file[order_fields].attrs["field_names"]
            for field_name in field_names:
                field = file[order_fields][field_name]
                if field_name in cache:
                    field_data = cache[field_name]
                else:
                    multi_index = ()
                    if field.attrs["sample_varying"]:
                        multi_index = multi_index + (sample_idx,)
                    if field.attrs["time_varying"]:
                        multi_index = multi_index + (slice(time_idx, time_idx + n_steps * dt, dt),)
                    field_data = field[multi_index] if multi_index else field[:]
                    field_data = torch.as_tensor(field_data)
                    if field_name in self.means:
                        field_data = field_data - self.means[field_name]
                    if field_name in self.stds:
                        field_data = field_data / self.stds[field_name]
                    variable_fields[i][field_name] = field_data
        return variable_fields
  
    def field_to_tensor(self, variable_fields):
        fields = []
        for field in list(variable_fields[0].values()):
            field = rearrange(field, "b t ... -> b t ... 1")
            fields.append(field)
        for field in list(variable_fields[1].values()):
            fields.append(field)
        for field in list(variable_fields[2].values()):
            field = rearrange(field, "b t ... c d-> b t ... (c d)")
            fields.append(field)
        field = torch.cat(fields, dim=-1)
        data = {}
        data['input']=field[0:self.n_steps_input, ...]
        data['output']=field[self.n_steps_input:, ...]
        return data

    def __getitem__(self, index):
        # Find specific file and local index
        file_idx = int(np.searchsorted(self.file_index_offsets, index, side="right") - 1)  # which file we are on
        windows_per_trajectory = self.n_windows_per_trajectory[file_idx]
        local_idx = index - max(self.file_index_offsets[file_idx], 0)  # First offset is -1
        sample_idx = local_idx // windows_per_trajectory
        time_idx = local_idx % windows_per_trajectory
        if self.files[file_idx] is None:
            self._open_file(file_idx)
        dt = self.dt_stride

        data = self._reconstruct_fields(self.files[file_idx], self.caches[file_idx], 
            sample_idx, time_idx, self.n_steps_input + self.n_steps_output, dt)
        
        return self.field_to_tensor(data)

    def __len__(self):
        return self.len

    def to_xarray(self, backend: Literal["numpy", "dask"] = "dask"):
        import xarray as xr
        datasets = []
        total_samples = 0
        for file_idx in range(len(self.files_paths)):
            if self.files[file_idx] is None:
                self._open_file(file_idx)
            ds = hdf5_to_xarray(self.files[file_idx], backend=backend)
            if "sample" not in ds.sizes:
                ds = ds.expand_dims("sample")
            if "sample" in ds.coords:
                n_samples = ds.sizes["sample"]
                ds = ds.assign_coords(sample=ds.coords["sample"] + total_samples)
                total_samples += n_samples
            datasets.append(ds)

        combined_ds = xr.concat(datasets, dim="sample")
        return combined_ds

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.data_path}>"
        
if __name__ == "__main__":
    dataset = TanteDataset(
        base_path = '/home/zw474/project/TANTE/dataset',
        #dataset_name = 'turbulent_radiative_layer_2D',
        dataset_name = 'turbulent_radiative_layer_2D',
        n_steps_input = 6,
        n_steps_output = 8,
        dt_stride = 2,
    )

    print(dataset[20]['input'].shape)
    print(dataset[20]['output'].shape)
    #variable = reconstruct_fields(example_file)
    #field = field_to_tensor(variable)
    #print(field.shape)
    #def getitem(index):

