import logging
import numpy as np
import matplotlib.pyplot as plt

from time import perf_counter
from tqdm import tqdm

from typing import cast
from itertools import product

from useq import MDASequence, Position, GridFromEdges
from useq._iter_sequence import _used_axes, _iter_axis, _parse_axes

from fish_sorter.constants import FOV_WIDTH, FOV_HEIGHT

# TODO is there an easier way to get the mosaic positions?

try:
    from pymmcore_widgets.useq_widgets import PYMMCW_METADATA_KEY as PYMMCW_METADATA_KEY
except ImportError:
    # key in MDASequence.metadata where we expect to find pymmcore_widgets metadata
    print('failed')
    PYMMCW_METADATA_KEY = "pymmcore_widgets"

DEFAULT_NAME = "Exp"


class Mosaic:
    def __init__(self, viewer):
        self.viewer = viewer

    def init_pos(self):

        sequence = MDASequence(            
            grid_plan = {
                "top": 0.0,
                "left": 0.0,
                "bottom": 0.0,
                "right": 0.0,
                "overlap": 5.0,
                "fov_width": FOV_WIDTH,
                "fov_height": FOV_HEIGHT,
            },
            channels = [
                {"config": "GFP","exposure": 100}, 
                {"config": "TXR", "exposure": 100}
            ],
            axis_order = "gc",
        )

        if isinstance(sequence.grid_plan, GridFromEdges):
            grid_plan = sequence.grid_plan  # Already correct
        else:
            # Convert if not already GridFromEdges
            grid_plan = GridFromEdges(
                fov_width=FOV_WIDTH,
                fov_height=FOV_HEIGHT,
                overlap=(5.0, 5.0),
                top=sequence.grid_plan.top,
                left=sequence.grid_plan.left,
                bottom=sequence.grid_plan.bottom,
                right=sequence.grid_plan.right,
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
        logging.info(f'rows: {rows}')
        cols = int(sequence.grid_plan.columns)
        logging.info(f'cols: {cols}')
        num_chan = len(sequence.channels)
        logging.info(f'num_chan: {num_chan}')
        chan_name = [channel.config for channel in sequence.channels]
        logging.info(f'chan_nam: {chan_name}')
        overlap = sequence.grid_plan.overlap
        logging.info(f'overlap: {overlap}')

        # Get position at each id
        pos_list = [[event.x_pos, event.y_pos] for event in list(sequence)]
        logging.info(f'pos_list: {pos_list}')

        # # Snippet below copied from useq._iter_sequence.py
        # order = _used_axes(sequence)
        # # this needs to be tuple(...) to work for mypyc
        # axis_iterators = tuple(enumerate(_iter_axis(sequence, ax)) for ax in order)
        # for i, item in enumerate(product(*axis_iterators)):
        #     if not item:  # the case with no events
        #         continue  # pragma: no cover
        #     # get axes objects for this event
        #     index, time, position, grid, channel, z_pos = _parse_axes(zip(order, item))


        #     print(grid)
        #     print(index)
        #     print(position)
        #     pos_order[i] = [grid.row, grid.col]

        # Reference abs position to grid position
        row_dict = {pos: i for i, pos in enumerate(np.sort(np.unique(pos_list[:,0])))}
        col_dict = {pos: i for i, pos in enumerate(np.sort(np.unique(pos_list[:,1])))}

        # Save order of positions
        idxs = np.zeros(np.shape(pos_list), dtype=int)
        u, u_idxs = np.unique(pos_list, axis=0, return_index=True)
        for i, pos in enumerate(u[np.argsort(u_idxs)]):
            idxs[row_dict[pos[0]], col_dict[pos[1]]] = i

        return rows, cols, num_chan, chan_name, overlap, idxs

    def get_img(self, zarr, row, col, idxs):
        """Get img for a given row and column"""
        idx = int(idxs[row, col])
        return zarr[0, idx, :, :, :]

    def stitch_mosaic(self, sequence : MDASequence, img_arr):
        """
        Stitch mosaic from MDA sequence and image array.

        Returns 3D array which can be indexed by (channel, y, x)
        """
        # Get metadata
        dir = self.get_dir(sequence)
        num_rows, num_cols, num_channels, overlap, idxs = self.get_mosaic_metadata(sequence)

        # Compute key distances
        x_overlap = int(IMG_X_PX * overlap[0] / 100.0)
        y_overlap = int(IMG_Y_PX * overlap[1] / 100.0)
        x_translation = IMG_X_PX - x_overlap
        y_translation = IMG_Y_PX - y_overlap

        # Get zarr array
        arr_data = self.viewer.layers[-1].data
        dtype = arr_data.dtype

        # TODO check that array has same dims as mosaic?

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
                mosaic[:, y_start : y_start + IMG_Y_PX, x_start : x_start + IMG_X_PX] += self.get_img(arr_data, row, col, idxs)

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

        return mosaic.astype(dtype)

    def display_mosaic(self, mosaic):
        """Display mosaic as a napari layer"""
        # Convert into array
        # Create image layer
        # TODO put mosaic in napari viewer
        pass