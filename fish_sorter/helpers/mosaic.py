import logging
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from time import perf_counter
from tqdm import tqdm
from typing import cast
from useq import MDASequence, Position, GridFromEdges
from useq._iter_sequence import _used_axes, _iter_axis, _parse_axes

from fish_sorter.constants import FOV_WIDTH, FOV_HEIGHT, IMG_X_PX, IMG_Y_PX

# TODO is there an easier way to get the mosaic positions?

try:
    from pymmcore_widgets.useq_widgets import PYMMCW_METADATA_KEY as PYMMCW_METADATA_KEY
except ImportError:
    # key in MDASequence.metadata where we expect to find pymmcore_widgets metadata
    logging.info('failed')
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
        num_chan = len(sequence.channels)
        logging.info(f'num_chan: {num_chan}')
        chan_names = [channel.config for channel in sequence.channels]
        logging.info(f'chan_nam: {chan_names}')
        overlap = sequence.grid_plan.overlap
        logging.info(f'overlap: {overlap}')

        # Get position at each id
        event_iterator = sequence.iter_events()
        pos_list = np.unique([[event.index['g'], event.x_pos, event.y_pos] for event in event_iterator], axis=0)
        xpos_list, x_ids = np.unique(pos_list[:,1], return_inverse=True)
        ypos_list, y_ids = np.unique(pos_list[:,2], return_inverse=True)
        num_rows = len(np.unique(pos_list[:,2]))
        num_cols = len(np.unique(pos_list[:,1]))

        # Save order of positions
        grid_list = np.zeros((num_cols, num_rows, 3), dtype=int)
        for grid_pos, y_id, x_id in zip(pos_list, y_ids, x_ids):
            grid_list[x_id, y_id] = grid_pos

        return grid_list, num_rows, num_cols, num_chan, chan_names, overlap

    def get_img(self, zarr, row, col, grid_list):
        """Get img for a given row and column"""
        idx = int(grid_list[col, row, 0])
        return zarr[0, idx, :, :, :]

    def stitch_mosaic(self, sequence : MDASequence, img_arr):
        """
        Stitch mosaic from MDA sequence and image array.

        Returns 3D array which can be indexed by (channel, y, x)
        """
        # Get metadata
        dir = self.get_dir(sequence)
        grid_list, num_rows, num_cols, num_channels, chan_names, overlap = self.get_mosaic_metadata(sequence)

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

        #TODO figure out right datatype


        mosaic = np.zeros((num_channels, mosaic_y_dim, mosaic_x_dim), dtype=np.uint16)

        # Assemble mosaic
        logging.info("Stitching images together")
        for row in tqdm(range(num_rows), desc="Row"):
            y_start = int(row * y_translation)
            for col in tqdm(range(num_cols), desc="Column"):
                x_start = int(col * x_translation)
                mirrored_col = (num_cols - 1) - col
                mosaic[:, y_start : y_start + IMG_Y_PX, x_start : x_start + IMG_X_PX] += self.get_img(arr_data, row, mirrored_col, grid_list)

        # Take average of overlapping areas
        logging.info("Taking average of overlapping areas")
        for row in tqdm(range(1, num_rows), desc="Row"):
            y_start = int(row * y_translation)
            mosaic[:, y_start : y_start - y_translation + IMG_Y_PX, :] = np.floor_divide(
                mosaic[:, y_start : y_start - y_translation + IMG_Y_PX, :],
                2
            ).astype(np.uint16)

            #TODO figure out right datatype

        for col in tqdm(range(1, num_cols), desc="Column"):
            x_start = int(col * x_translation)
            mosaic[:, :, x_start : x_start - x_translation + IMG_X_PX] = np.floor_divide(
                mosaic[:, :, x_start : x_start - x_translation + IMG_X_PX],
                2
            ).astype(np.uint16)

            #TODO figure out right datatype


        mosaic = np.flip(mosaic, axis=2)

        return mosaic.astype(dtype)

    def display_mosaic(self, mosaic):
        """Display mosaic as a napari layer"""
        # Convert into array
        # Create image layer
        # TODO put mosaic in napari viewer
        pass