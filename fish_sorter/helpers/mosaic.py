import numpy as np
import matplotlib.pyplot as plt

from time import perf_counter
from tqdm import tqdm

from typing import cast
from itertools import product

from useq import MDASequence, Position
from useq._iter_sequence import _used_axes, _iter_axis, _parse_axes

from helpers.constants import IMG_X_PX, IMG_Y_PX

try:
    from pymmcore_widgets.useq_widgets import PYMMCW_METADATA_KEY as PYMMCW_METADATA_KEY
except ImportError:
    # key in MDASequence.metadata where we expect to find pymmcore_widgets metadata
    print('failed')
    PYMMCW_METADATA_KEY = "pymmcore_widgets"

DEFAULT_NAME = "Exp"


class MosaicHandler:
    def __init__(self):
        self.sequence = self.get_sequence()
        
    def get_sequence(self):
        sequence = MDASequence(
            channels = [
                {"config": "GFP","exposure": 100}, 
                {"config": "TXR", "exposure": 100}
            ],
            grid_plan={"rows": 3, "columns": 1, "relative_to": "top_left", "overlap": 5, "mode": "row_wise_snake"},
            # stage_positions = [
            #     # {"x": 110495.44, "y": 10863.76, "z": 2779.09, "name": "top_R"},
            #     # {"x": 17883.77, "y" : 10166.54, "z": 2779.09, "name": "top_L"},
            #     # {"x": 110495.44, "y": 73208.59, "z": 2776.70, "name": "bot_R"},
            #     # {"x": 17492.82, "y": 73208.58, "z": 2776.70, "name": "bot_L"},
            #     Position(
            #         x=17883.77, y=10166.54, z=2779.09, name= "array", 
            #         sequence=MDASequence(
            #             grid_plan={"rows": 3, "columns": 4, "relative_to": "top_left", "overlap": 5, "mode": "row_wise_snake"})
            #     ),
            # ],
            axis_order = "pc",
        )
        return sequence

    def get_dir(self, sequence: MDASequence) -> str:
        """
        Get the file dir from the MDASequence metadata
        
        Copied from napari_micromanager/_mda_handler.py
        """
        meta = cast("dict", sequence.metadata.get(PYMMCW_METADATA_KEY, {}))
        return cast(str, meta.get('save_dir', None))

    def get_filename(self, sequence: MDASequence) -> str:
        """
        Get the file name from the MDASequence metadata
        
        Copied from napari_micromanager/_mda_handler.py
        """
        meta = cast("dict", sequence.metadata.get(PYMMCW_METADATA_KEY, {}))
        return cast(str, meta.get('save_name', DEFAULT_NAME))

    def get_mosaic_metadata(self, sequence: MDASequence):
        """Get mosaic info from the MDASequence metadata"""
        # General metadata
        rows = int(sequence.grid_plan.rows)
        cols = int(sequence.grid_plan.columns)
        channels = len(sequence.channels)
        overlap = sequence.grid_plan.overlap

        # Get position at each id
        pos_order = np.zeros((rows * cols * channels, 2), dtype=int)

        # Snippet below copied from useq._iter_sequence.py
        order = _used_axes(sequence)
        # this needs to be tuple(...) to work for mypyc
        axis_iterators = tuple(enumerate(_iter_axis(sequence, ax)) for ax in order)
        for i, item in enumerate(product(*axis_iterators)):
            if not item:  # the case with no events
                continue  # pragma: no cover
            # get axes objects for this event
            index, time, position, grid, channel, z_pos = _parse_axes(zip(order, item))

            pos_order[i] = [grid.row, grid.col]

        # Save order of positions
        idxs = np.zeros((rows, cols), dtype=int)
        for i, pos in enumerate(np.unique(pos_order, axis=0)):
            idxs[pos[0], pos[1]] = i

        return rows, cols, channels, overlap, idxs

    def get_img(self, zarr, row, col, idxs):
        """Get img for a given row and column"""
        idx = int(idxs[row, col])
        return zarr[0, idx, :, :, :]

    def stitch_mosaic(self, sequence : MDASequence, img_arr):
        """Get img for a given row and column"""
        # Get metadata
        dir = self.get_dir(sequence)
        num_rows, num_cols, num_channels, overlap, idxs = self.get_mosaic_metadata(sequence)

        # Compute key distances
        x_overlap = int(overlap[0] / 100.0)
        y_overlap = int(overlap[1] / 100.0)
        x_translation = IMG_X_PX - x_overlap
        y_translation = IMG_Y_PX - y_overlap

        # Get zarr array
        zarr_id = list(img_arr)[-1]
        zarr = img_arr[zarr_id][0]
        dtype = zarr.dtype

        # TODO check that zarr array has same dims as mosaic?

        # Initialize empty mosaic
        mosaic_x_dim = int((IMG_X_PX * num_cols) - (x_overlap * (num_cols - 1)))
        mosaic_y_dim = int((IMG_Y_PX * num_rows) - (y_overlap * (num_rows - 1)))
        mosaic = np.zeros((num_channels, mosaic_y_dim, mosaic_x_dim), dtype=np.uint32)

        # Assemble mosaic
        print("Stitching images together")
        for row in tqdm(range(num_rows), desc="Row"):
            y_start = int(row * y_translation)
            for col in tqdm(range(0, num_cols), desc="Column"):
                x_start = int(col * x_translation)
                mosaic[:, y_start : y_start + IMG_Y_PX, x_start : x_start + IMG_X_PX] += self.get_img(zarr, row, col, idxs)

        # Take average of overlapping areas
        print("Taking average of overlapping areas")
        for row in tqdm(range(1, num_rows), desc="Row"):
            y_start = int(row * y_translation)
            mosaic[:, y_start : y_start - y_translation + IMG_Y_PX, :] = np.floor_divide(
                mosaic[:, y_start : y_start - y_translation + IMG_Y_PX, :],
                2
            ).astype(np.uint32)
        for col in tqdm(range(1, num_cols), desc="Column"):
            x_start = int(col * x_translation)
            mosaic[:, :, x_start : x_start - x_translation + IMG_X_PX] = np.floor_divide(
                mosaic[:, :, x_start : x_start - x_translation + IMG_X_PX],
                2
            ).astype(np.uint32)

        for channel in range(num_channels):
            plt.imsave(f"test_mosaic_{channel}.png", mosaic[channel, :, :].astype(dtype))

        return mosaic.astype(dtype), zarr_id

    def display_mosaic(self, mosaic, zarr_id):
        """Display mosaic as a napari layer"""
        # Convert into zarr array
        # Create image layer
        pass