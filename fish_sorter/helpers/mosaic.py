# TODO clean up the imports
import napari
import napari_micromanager
import numpy as np
import os
import pymmcore_plus

from time import perf_counter
from tqdm import tqdm
import matplotlib.pyplot as plt

from typing import cast

from pathlib import Path
from useq import MDASequence, Position
from useq._iter_sequence import _sizes, _used_axes, _iter_axis, _parse_axes

from itertools import product

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
        """Get the file dir from the MDASequence metadata."""
        meta = cast("dict", sequence.metadata.get(PYMMCW_METADATA_KEY, {}))
        return cast(str, meta.get('save_dir', None))

    def get_filename(self, sequence: MDASequence) -> str:
        """Get the file name from the MDASequence metadata."""
        # Copied from https://github.com/pymmcore-plus/napari-micromanager/blob/6c895f36502c9d0eb09839abd82d6dd706af96a4/src/napari_micromanager/_mda_handler.py#L38-L41
        meta = cast("dict", sequence.metadata.get(PYMMCW_METADATA_KEY, {}))
        return cast(str, meta.get('save_name', DEFAULT_NAME))

    def get_mosaic_metadata(self, sequence: MDASequence):
        rows = int(sequence.grid_plan.rows)
        cols = int(sequence.grid_plan.columns)
        channels = len(sequence.channels)
        overlap = sequence.grid_plan.overlap
        print(overlap)

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

        pos_order = np.unique(pos_order, axis=0)
        print(pos_order)
        idxs = np.zeros((rows, cols), dtype=int)
        for i, pos in enumerate(pos_order):
            print(pos)
            print(i)
            idxs[pos[0], pos[1]] = i

        print(idxs)

        return rows, cols, channels, overlap, pos_order

    def get_img(self, zarr, row, col, idxs):
        # print(f"GET_IMG OG SHAPE: {zarr.shape}")
        t0 = perf_counter()
        idx = int(idxs[row, col])
        img = zarr[0, idx, :, :, :]
        t1 = perf_counter()
        # print(f"GET_IMG TIME: {t1-t0}")
        # print(f"GET_IMG SHAPE: {img.shape}")

        return img

    def stitch_mosaic(self, sequence : MDASequence, img_arr):
        '''
        Assemble mosaic
        '''
        dir = self.get_dir(sequence)
        num_rows, num_cols, num_channels, overlap, idxs = self.get_mosaic_metadata(sequence)
        x_overlap = int(overlap[0] / 100.0)
        y_overlap = int(overlap[1] / 100.0)

        # print(f"IMG ARR: {img_arr}")
        zarr_id = list(img_arr)[-1]
        
        print(zarr_id)
        # print(img_arr[zarr_id][0])
        zarr = img_arr[zarr_id][0]
        dtype = zarr.dtype
        print(zarr)
        print(dtype)
        # print(f"ZARR: {zarr[0, 0, 0, 0, 0]}")

        # TODO check that zarr array has same dims as mosaic?

        mosaic_x_dim = int((IMG_X_PX * num_cols) - (x_overlap * (num_cols - 1)))
        mosaic_y_dim = int((IMG_Y_PX * num_rows) - (y_overlap * (num_rows - 1)))

        mosaic = np.zeros((num_channels, mosaic_y_dim, mosaic_x_dim), dtype=np.uint32)

        x_translation = IMG_X_PX - x_overlap
        y_translation = IMG_Y_PX - y_overlap

        # Assemble mosaic
        print("Stitching images together")
        for row in tqdm(range(num_rows), desc="Row"):
            y_start = int(row * y_translation)
            for col in tqdm(range(0, num_cols), desc="Column"):
                x_start = int(col * x_translation)

                # if row == 0 and col == 0:
                #     plt.imsave("test.png", self.get_img(zarr, 0, 0)[0, :, :])

                # TODO test image loading
                t0 = perf_counter()
                # print(f"MOSAIC SHAPE {mosaic[:, y_start : y_start + IMG_Y_PX, x_start : x_start + IMG_X_PX].shape}")
                # print(f"ZARR SHAPE {self.get_img(zarr, row, col).shape}")
                mosaic[:, y_start : y_start + IMG_Y_PX, x_start : x_start + IMG_X_PX] += self.get_img(zarr, row, col, idxs)
                t1 = perf_counter()
                print(f"TIME: {t1-t0}")

        # Take average of overlapping areas
        print("Taking average of overlapping areas")
        for row in tqdm(range(1, num_rows), desc="Row"):
            t0 = perf_counter()
            y_start = int(row * y_translation)
            mosaic[:, y_start : y_start - y_translation + IMG_Y_PX, :] = np.floor_divide(
                mosaic[:, y_start : y_start - y_translation + IMG_Y_PX, :],
                2
            ).astype(np.uint32)
            t1 = perf_counter()
            print(t1-t0)
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
        # Convert into zarr array
        # Create image layer
        pass
