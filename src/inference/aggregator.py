import warnings
from typing import Tuple
import torch
import numpy as np
# from torchio import TypeData, CHANNELS_DIMENSION
from torchio import CHANNELS_DIMENSION
from .grid_sampler import GridSampler


class GridAggregator:
    r"""Aggregate patches for dense inference.

    This class is typically used to build a volume made of patches after
    inference of batches extracted by a :py:class:`~torchio.data.GridSampler`.

    Args:
        sampler: Instance of :py:class:`~torchio.data.GridSampler` used to
            extract the patches.
        overlap_mode: If ``'crop'``, the overlapping predictions will be
            cropped. If ``'average'``, the predictions in the overlapping areas
            will be averaged with equal weights. See the
            `grid aggregator tests`_ for a raw visualization of both modes.

    .. _grid aggregator tests: https://github.com/fepegar/torchio/blob/master/tests/data/inference/test_aggregator.py

    .. note:: Adapted from NiftyNet. See `this NiftyNet tutorial
        <https://niftynet.readthedocs.io/en/dev/window_sizes.html>`_ for more
        information about patch-based sampling.
    """
    def __init__(self, sampler: GridSampler, overlap_mode: str = 'crop'):
        subject = sampler.subject
        self.volume_padded = sampler.padding_mode is not None
        self.spatial_shape = subject.shape
        self._output_tensor = None
        self.patch_overlap = sampler.patch_overlap
        self.parse_overlap_mode(overlap_mode)
        self.overlap_mode = overlap_mode
        self._avgmask_tensor = None

    @staticmethod
    def parse_overlap_mode(overlap_mode):
        if overlap_mode not in ('crop', 'average'):
            message = (
                f'Overlap mode must be "crop" or "average" but "{overlap_mode}"'
                ' was passed'
            )
            raise ValueError(message)

    def crop_batch(
            self,
            batch: torch.Tensor,
            locations: np.ndarray,
            overlap: np.ndarray,
            # ) -> Tuple[TypeData, np.ndarray]:
            ):
        num_dimensions = len(overlap)

        border = np.array(overlap) // 2  # overlap is even in grid sampler
        
        crop_locations = locations.astype(int).copy()
        if num_dimensions == 3:
            indices_ini, indices_fin = crop_locations[:, :3], crop_locations[:, 3:]
        else:
            indices_ini, indices_fin = crop_locations[:, :2], crop_locations[:, 2:]
        num_locations = len(crop_locations)

        border_ini = np.tile(border, (num_locations, 1))
        border_fin = border_ini.copy()
        # Do not crop patches at the border of the volume
        # Unless we're padding the volume in the grid sampler. In that case,
        # it doesn't matter if we don't crop patches at the border, because the
        # output volume will be cropped
        if not self.volume_padded:
            mask_border_ini = indices_ini == 0
            border_ini[mask_border_ini] = 0
            for axis, size in enumerate(self.spatial_shape):
                mask_border_fin = indices_fin[:, axis] == size
                border_fin[mask_border_fin, axis] = 0

        indices_ini += border_ini
        indices_fin -= border_fin

        crop_shapes = indices_fin - indices_ini
        patch_shape = batch.shape[2:]  # ignore batch and channels dim
        # num_dimensions = len(patch_shape)
        cropped_patches = []

        for patch, crop_shape in zip(batch, crop_shapes):
            diff = patch_shape - crop_shape
            left = (diff / 2).astype(int)

            if num_dimensions == 3:
                i_ini, j_ini, k_ini = left
                i_fin, j_fin, k_fin = left + crop_shape
                cropped_patch = patch[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin]
            else:
                i_ini, j_ini = left
                i_fin, j_fin  = left + crop_shape
                cropped_patch = patch[:, i_ini:i_fin, j_ini:j_fin]


            # if num_dimensions == 3:
            #     i_ini, j_ini, k_ini = left
            #     i_fin, j_fin, k_fin = left + cropped_shape
            #     batch = windows[
            #         :,  # batch dimension
            #         :,  # channels dimension
            #         i_ini:i_fin,
            #         j_ini:j_fin,
            #         k_ini:k_fin,]
            # else:
            #     i_ini, j_ini = left
            #     i_fin, j_fin = left + cropped_shape
            #     batch = windows[
            #         :,  # batch dimension
            #         :,  # channels dimension
            #         i_ini:i_fin,
            #         j_ini:j_fin,]



            cropped_patches.append(cropped_patch)

        return cropped_patches, crop_locations

    def initialize_output_tensor(self, batch: torch.Tensor) -> None:
        if self._output_tensor is not None:
            return
        num_channels = batch.shape[CHANNELS_DIMENSION]
        self._output_tensor = torch.zeros(
            num_channels,
            *self.spatial_shape,
            dtype=batch.dtype,
        )

    def initialize_avgmask_tensor(self, batch: torch.Tensor) -> None:
        if self._avgmask_tensor is not None:
            return
        num_channels = batch.shape[CHANNELS_DIMENSION]
        self._avgmask_tensor = torch.zeros(
            num_channels,
            *self.spatial_shape,
            dtype=batch.dtype,
        )

    def add_batch(
            self,
            batch_tensor: torch.Tensor,
            locations: torch.Tensor,
            ) -> None:
        """Add batch processed by a CNN to the output prediction volume.

        Args:
            batch_tensor: 5D tensor, typically the output of a convolutional
                neural network, e.g. ``batch['image'][torchio.DATA]``.
            locations: 2D tensor with shape :math:`(B, 6)` representing the
                patch indices in the original image. They are typically
                extracted using ``batch[torchio.LOCATION]``.
        """
        batch = batch_tensor.cpu()
        locations = locations.cpu().numpy()
        self.initialize_output_tensor(batch)
        if self.overlap_mode == 'crop':
            cropped_patches, crop_locations = self.crop_batch(
                batch,
                locations,
                self.patch_overlap,
            )

            for patch, crop_location in zip(cropped_patches, crop_locations):
                if len(crop_location) > 4:
                    i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = crop_location
                    self._output_tensor[
                        :,
                        i_ini:i_fin,
                        j_ini:j_fin,
                        k_ini:k_fin] = patch
                else:
                    i_ini, j_ini, i_fin, j_fin = crop_location
                    self._output_tensor[:, i_ini:i_fin, j_ini:j_fin] = patch


        elif self.overlap_mode == 'average':
            self.initialize_avgmask_tensor(batch)
            for patch, location in zip(batch, locations):
                if len(location) > 4:
                    i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location
                    self._output_tensor[
                        :,
                        i_ini:i_fin,
                        j_ini:j_fin,
                        k_ini:k_fin] += patch
                    self._avgmask_tensor[
                        :,
                        i_ini:i_fin,
                        j_ini:j_fin,
                        k_ini:k_fin] += 1
                else:
                    i_ini, j_ini, i_fin, j_fin = location
                    self._output_tensor[i_ini:i_fin, j_ini:j_fin] = patch


    def get_output_tensor(self) -> torch.Tensor:
        """Get the aggregated volume after dense inference."""
        if self._output_tensor.dtype == torch.int64:
            message = (
                'Medical image frameworks such as ITK do not support int64.'
                ' Casting to int32...'
            )
            warnings.warn(message)
            self._output_tensor = self._output_tensor.type(torch.int32)
        if self.overlap_mode == 'average':
            output = self._output_tensor / self._avgmask_tensor
        else:
            output = self._output_tensor
        if self.volume_padded:
            # from torchio.transforms import Crop
            border = np.array(self.patch_overlap) // 2
            border = border.astype(np.int).repeat(2)
            # cropping = border.astype(np.int).repeat(2)
            # crop = Crop(cropping)
            # return crop(output)
            # image[DATA] = image[DATA][:, i0:i1, j0:j1, k0:k1].clone()
            if len(border) > 4:
                output = output[:, border[0]:-border[0], border[1]:-border[1], border[2]:-border[2]].clone()
            else:
                output = output[:, border[0]:-border[0], border[1]:-border[1]].clone()
            return output
            
        else:
            return output
