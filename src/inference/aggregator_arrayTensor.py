from typing import Union, Tuple
import torch
import numpy as np
# from ...utils import to_tuple
# from ...torchio import TypeData, TypeTuple

# from torchio import TypeData, TypeTuple
# from torchio.utils import to_tuple





# /home/npnguyen/anaconda3/lib/python3.6/site-packages/torchio/torchio.py
# from pathlib import Path
# from typing import Union, Tuple, Callable

# In PyTorch convention
BATCH_DIMENSION = 0
CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4

# For typing hints
# TypePath = Union[Path, str]
# TypeNumber = Union[int, float]
TypeData = Union[torch.Tensor, np.ndarray]
TypeTripletInt = Tuple[int, int, int]
# TypeSextetInt = Tuple[int, int, int, int, int, int]
# TypeTripletFloat = Tuple[float, float, float]
TypeTuple = Union[int, TypeTripletInt]
# TypeRangeInt = Union[int, Tuple[int, int]]
# TypePatchSize = Union[int, Tuple[int, int, int]]
# TypeRangeFloat = Union[float, Tuple[float, float]]
# TypeCallable = Callable[[torch.Tensor], torch.Tensor]





class GridAggregator:
    """
    Adapted from NiftyNet.
    See https://niftynet.readthedocs.io/en/dev/window_sizes.html
    """
    def __init__(
            self,
            data: TypeData,
            patch_overlap: TypeTuple,
            ):
        data = torch.from_numpy(data) if isinstance(data, np.ndarray) else data
        self._output_tensor = torch.zeros_like(data)
        # self.patch_overlap: Tuple[int, int, int] = to_tuple(patch_overlap)
        self.patch_overlap = patch_overlap

    @staticmethod
    def crop_batch(
            windows: TypeData,
            location: np.ndarray,
            border: Tuple[int, int, int],
            ) -> Tuple[TypeData, np.ndarray]:
        location = location.astype(np.int)
        batch_shape = windows.shape
        spatial_shape = batch_shape[2:]  # ignore batch and channels dim
        num_dimensions = len(spatial_shape)

        if num_dimensions == 3:
            for idx in range(num_dimensions):
                location[:, idx] = location[:, idx] + border[idx]
                location[:, idx + 3] = location[:, idx + 3] - border[idx]
            cropped_shape = np.max(location[:, 3:6] - location[:, 0:3], axis=0)
        else:
            for idx in range(num_dimensions):
                location[:, idx] = location[:, idx] + border[idx]
                location[:, idx + 2] = location[:, idx + 2] - border[idx]
            cropped_shape = np.max(location[:, 2:4] - location[:, 0:2], axis=0)


        diff = spatial_shape - cropped_shape
        left = np.floor(diff / 2).astype(np.int)

        if num_dimensions == 3:
            i_ini, j_ini, k_ini = left
            i_fin, j_fin, k_fin = left + cropped_shape
            batch = windows[
                :,  # batch dimension
                :,  # channels dimension
                i_ini:i_fin,
                j_ini:j_fin,
                k_ini:k_fin,]
        else:
            i_ini, j_ini = left
            i_fin, j_fin = left + cropped_shape
            batch = windows[
                :,  # batch dimension
                :,  # channels dimension
                i_ini:i_fin,
                j_ini:j_fin,]

        return batch, location

    def add_batch(self, windows: TypeData, locations: TypeData) -> None:
        windows = windows.cpu()
        location_init = np.copy(locations)
        init_ones = np.ones_like(windows)
        windows, _ = self.crop_batch(
            windows, location_init, self.patch_overlap)
        location_init = np.copy(locations)
        _, locations = self.crop_batch(
            init_ones, location_init, self.patch_overlap)

        for window, location in zip(windows, locations):
            window = window[0]

            if len(location) > 4:
                i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location
                # window = self.array[i_ini:i_fin, j_ini:j_fin, k_ini:k_fin]
                self._output_tensor[i_ini:i_fin, j_ini:j_fin, k_ini:k_fin] = window

            else:
                i_ini, j_ini, i_fin, j_fin = location
                # window = self.array[i_ini:i_fin, j_ini:j_fin]
                self._output_tensor[i_ini:i_fin, j_ini:j_fin] = window


            # i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location
            # self._output_tensor[i_ini:i_fin, j_ini:j_fin, k_ini:k_fin] = window

    def get_output_tensor(self) -> torch.Tensor:
        return self._output_tensor
