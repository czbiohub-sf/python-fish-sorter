import napari
import napari_micromanager
import numpy as np
import os
import pymmcore_plus

from typing import cast

from pathlib import Path
from useq import MDASequence, Position
from gui.pipette_gui import PipetteWidget

try:
    from pymmcore_widgets.useq_widgets import PYMMCW_METADATA_KEY as PYMMCW_METADATA_KEY
except ImportError:
    # key in MDASequence.metadata where we expect to find pymmcore_widgets metadata
    print('failed')
    PYMMCW_METADATA_KEY = "pymmcore_widgets"

# For Prosilica GT 2050
IMG_X_PX = 2048
IMG_Y_PX = 2048

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
            stage_positions = [
                # {"x": 110495.44, "y": 10863.76, "z": 2779.09, "name": "top_R"},
                # {"x": 17883.77, "y" : 10166.54, "z": 2779.09, "name": "top_L"},
                # {"x": 110495.44, "y": 73208.59, "z": 2776.70, "name": "bot_R"},
                # {"x": 17492.82, "y": 73208.58, "z": 2776.70, "name": "bot_L"},
                Position(
                    x=17883.77, y=10166.54, z=2779.09, name= "array", 
                    sequence=MDASequence(
                        grid_plan={"rows": 3, "columns": 4, "relative_to": "top_left", "overlap": 5, "mode": "row_wise_snake"})
                ),
            ],
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
        print(cast("dict", sequence.stage_positions))
        meta = cast("dict", sequence.stage_positions.sequence)
        print(meta)
        print(cast(str, meta.get('stage_positions', DEFAULT_NAME)))

    def stitch_mosaic(self, sequence : MDASequence, img_arr):
        '''
        Assemble mosaic
        '''
        dir = self.get_dir(sequence)
        self.get_mosaic_metadata(sequence)

        # mosaic_x_dim = (IMG_X_PX * self.num_cols) - (self.x_overlap * (self.num_cols- 1))
        # mosaic_y_dim = (IMG_Y_PX * self.num_rows) - (self.y_overlap * (self.num_rows - 1))

        # mosaic = np.zeros((mosaic_y_dim, mosaic_x_dim, self.img_z_dim), dtype=np.uint32)

        # x_translation = IMG_X_PX - self.x_overlap
        # y_translation = IMG_Y_PX - self.y_overlap

        # # Assemble mosaic
        # print("Stitching images together")
        # for row in tqdm(range(self.num_rows), desc="Row"):
        #     y_start = row * y_translation
        #     for col in tqdm(range(0, self.num_cols), desc="Column"):
        #         x_start = col * x_translation

        #         mosaic[y_start : y_start + IMG_Y_PX, x_start : x_start + IMG_X_PX, :] += self._get_img(row, col)

        # # Take average of overlapping areas
        # print("Taking average of overlapping areas")
        # for row in tqdm(range(1, self.num_rows), desc="Row"):
        #     y_start = row * y_translation
        #     mosaic[y_start : y_start - y_translation + IMG_Y_PX, :, :] = np.floor_divide(
        #         mosaic[y_start : y_start - y_translation + IMG_Y_PX, :, :],
        #         2
        #     ).astype(np.uint32)
        # for col in tqdm(range(1, self.num_cols), desc="Column"):
        #     x_start = col * x_translation
        #     mosaic[:, x_start : x_start - x_translation + IMG_X_PX, :] = np.floor_divide(
        #         mosaic[:, x_start : x_start - x_translation + IMG_X_PX, :],
        #         2
        #     ).astype(np.uint32)

        # self.mosaic = mosaic.astype(self.dtype)    

