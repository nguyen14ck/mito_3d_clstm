from typing import Union, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
# from torchio import IMAGE, LOCATION
from torchio.utils import to_tuple


# /home/npnguyen/anaconda3/lib/python3.6/site-packages/torchio/torchio.py
# # For aggregator
IMAGE = 'image'
LOCATION = 'location'



class GridSampler(Dataset):
    """
    Adapted from NiftyNet.
    See https://niftynet.readthedocs.io/en/dev/window_sizes.html
    """
    def __init__(
            self,
            data: Union[np.ndarray, torch.Tensor],
            patch_size: Union[int, Sequence[int]],
            patch_overlap: Union[int, Sequence[int]],
            # padding_mode: Union[str, float, None] = 'symmetric',
            ):
        self.array = data
        # patch_size = to_tuple(patch_size)
        # patch_overlap = to_tuple(patch_overlap)
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        # self.padding_mode = padding_mode
        # if padding_mode is not None:
        #     from torchio.transforms import Pad
        #     border = np.array(self.patch_overlap) // 2
        #     padding = border.astype(np.int).repeat(2)
        #     pad = Pad(padding, padding_mode=padding_mode)
        #     self.array = pad(self.array)
        self.locations = self._grid_spatial_coordinates(
            self.array,
            patch_size,
            patch_overlap,
        )

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, index):
        # Assume 2D-3D
        location = self.locations[index]
        if len(location) > 4:
            i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location
            window = self.array[i_ini:i_fin, j_ini:j_fin, k_ini:k_fin]
        else:
            i_ini, j_ini, i_fin, j_fin = location
            window = self.array[i_ini:i_fin, j_ini:j_fin]
        window = window[np.newaxis, ...]  # add channels dimension
        sample = {IMAGE: window, LOCATION: location}
        return sample

    @staticmethod
    def _enumerate_step_points(
            starting: int,
            ending: int,
            win_size: int,
            step_size: int,
            ) -> np.ndarray:
        starting = max(int(starting), 0)
        ending = max(int(ending), 0)
        win_size = max(int(win_size), 1)
        step_size = max(int(step_size), 1)
        if starting > ending:
            starting, ending = ending, starting
        sampling_point_set = []
        while (starting + win_size) <= ending:
            sampling_point_set.append(starting)
            starting = starting + step_size
        additional_last_point = ending - win_size
        sampling_point_set.append(max(additional_last_point, 0))
        sampling_point_set = np.unique(sampling_point_set).flatten()
        if len(sampling_point_set) == 2:
            sampling_point_set = np.append(
                sampling_point_set, np.round(np.mean(sampling_point_set)))
        _, uniq_idx = np.unique(sampling_point_set, return_index=True)
        return sampling_point_set[np.sort(uniq_idx)]

    @staticmethod
    def _grid_spatial_coordinates(
            array: np.ndarray,
            window_shape: Tuple[int],
            border: Tuple[int],
            ):
        shape = array.shape
        num_dims = len(shape)
        grid_size = [
            max(win_size - 2 * border, 0)
            for (win_size, border)
            in zip(window_shape, border)
        ]
        steps_along_each_dim = [
            GridSampler._enumerate_step_points(
                starting=0,
                ending=shape[i],
                win_size=window_shape[i],
                step_size=grid_size[i],
            )
            for i in range(num_dims)
        ]
        starting_coords = np.asanyarray(np.meshgrid(*steps_along_each_dim))
        starting_coords = starting_coords.reshape((num_dims, -1)).T
        n_locations = starting_coords.shape[0]
        # prepare the output coordinates matrix
        spatial_coords = np.zeros((n_locations, num_dims * 2), dtype=np.int32)
        spatial_coords[:, :num_dims] = starting_coords
        for idx in range(num_dims):
            spatial_coords[:, num_dims + idx] = (
                starting_coords[:, idx]
                + window_shape[idx]
            )
        max_coordinates = np.max(spatial_coords, axis=0)[num_dims:]
        assert np.all(max_coordinates <= shape[:num_dims]), \
            "window size greater than the spatial coordinates {} : {}".format(
                max_coordinates, shape)
        return spatial_coords
