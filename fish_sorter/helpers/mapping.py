# TODO clean up the imports
import napari
import napari_micromanager
import numpy as np
import os
import json
import pymmcore_plus

from tqdm import tqdm
from useq import MDASequence, Position
from abc import ABC, abstractmethod

from typing import cast

from pathlib import Path
from useq import MDASequence, Position
from useq._iter_sequence import _sizes, _used_axes, _iter_axis, _parse_axes

# TODO dynamically load pixel count
from fish_sorter.constants import (
    IMG_X_PX,
    IMG_Y_PX,
)

# NOTE TL corner needs to have image center at corner,
#      unless manually overriden by set_center_to_corner_offset_um
# TODO add type hints

class Mapping:
    def __init__(self, mmc):
        self.mmc = mmc

        # NOTE Does mda return z values?
        self.um_home = None
        self.um_TR = None

        self.px2um = self.mmc.getPixelSizeUm() # Scaling factor
        self.um_center_to_corner_offset = self._get_center_to_corner_offset_um()

        self.theta = 0.0
        self.transform = np.array([[1, 0], [0, 1]])
        self.wells = {}

        # TODO save TL/TR locations in experiment savefile

    @abstractmethod
    def set_calib_pts(self):
        pass

    @abstractmethod
    def go_to_well(self, well, offset):
        pass

    def _get_center_to_corner_offset_um(self):
        # Compute home in px units assuming TR mosaic tile is centered on home
        return np.array(
            [
                IMG_X_PX * self.px2um / 2,
                IMG_Y_PX * self.px2um / 2,
            ]
        )

    def set_center_to_corner_offset_um(self, px_home ):
        # Manually set home in px units
        self.um_center_to_corner_offset = np.multiply(px_home, self.px2um)

    def set_home_and_transform(self):
        # User needs to previously set home in TL slot and navigate to TR corner before pressing "calibrate"
        vector = self.um_TR[0:2] - self.um_home[0:2]
        self.theta = np.arctan(vector[1] / vector[0])

        self.transform = np.array(
            [
                [np.cos(self.theta), np.sin(self.theta)],
                [-np.sin(self.theta), np.cos(self.theta)]
            ]
        )

    def get_transform(self):
        # Get transformation matrix and corresponding angle
        return self.transform, self.theta

    def apply_transform(self, pos):
        # Assume input is a np array, with each position as a row array [[x1; y1], [x2, y2], ...] 

        # Ideally, user has previously set transform
        # TODO: Add user prompt if not
        return np.matmul(pos, self.transform)

    def load_wells(self, filename):
        # User needs to previously set home in TL slot and set transform
        # TODO: Add user prompt

        with open(filename) as f:
            plate_data = json.load(f)

        # Load metadata
        # TODO redo file compatibility
        well_names = plate_data['wells']['well_names']
        unformatted_well_pos = np.array(plate_data['wells']['well_coordinates'])

        # Format well positions
        well_count = int(plate_data['array_design']['rows']) * int(plate_data['array_design']['columns'])
        well_pos = unformatted_well_pos.reshape(well_count, 2)

        # Transform wells
        transformed_well_pos = self._apply_transform(well_pos)
        abs_well_pos = self.rel_um_to_abs_um(transformed_well_pos)
        px_well_pos = self.rel_um_to_px(transformed_well_pos)

        # Load sequence
        self.wells = {
            'names' : well_names,
            'raw_rel_um' : well_pos,
            "calib_rel_um": transformed_well_pos,
            "calib_abs_um": abs_well_pos,
            "calib_px": px_well_pos, # NOTE px is unused for dispense plate
        }

    def _get_well_pos(self, well: str, offset):
        if well not in self.wells:
            return

        pos = self.wells[well].abs_um
        x = pos[0] + offset[0]
        y = pos[1] + offset[1]

        return x, y

    def px_to_rel_um(self, px_pos):
        # Wellplate coords to stage coords
        return (px_pos * self.px2um) - self.um_center_to_corner_offset

    def rel_um_to_px(self, rel_um_pos):
        # Wellplate coords to image coords        
        return (rel_um_pos + self.um_center_to_corner_offset) / self.px2um

    def rel_um_to_abs_um(self, rel_um_pos):
        # Wellplate coords to stage coords
        return rel_um_pos += self.um_home + self.um_center_to_corner_offset

    def abs_to_rel(self, abs_um_pos):
        # Stage coords to wellplate coords
        return abs_um_pos -= self.um_home + self.um_center_to_corner_offset

    def px_to_abs_um(self, px_pos):
        # Image coords to stage coords
        return self.rel_um_to_abs_um(self.px_to_rel_um(px_pos))

    def abs_um_to_px(self, abs_um_pos):
        # Stage coords to image coords
        return self.rel_um_to_px(self.abs_um_to_rel_um(px_pos))
