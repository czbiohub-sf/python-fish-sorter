# TODO clean up the imports
import logging
import napari
import napari_micromanager
import numpy as np
import os
import json
import pymmcore_plus

from tqdm import tqdm
from useq import MDASequence, Position
from abc import ABC, abstractmethod

from typing import cast, Optional, Tuple

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
    def __init__(self, mmc, array_file):
        self.mmc = mmc

        # NOTE Does mda return z values?
        self.um_TL = None
        self.um_BR = None

        self.px2um = self.mmc.getPixelSizeUm() # Scaling factor
        self.um_center_to_corner_offset = 0.0

        self.theta_diff = 0.0
        self.transform = np.array([[1, 0], [0, 1]])
        self.wells = {}

        with open(array_file) as f:
            self.plate_data = json.load(f)

        logging.info(f'plate data: {self.plate_data}')

        # TODO save TL/BR locations in experiment savefile

    @abstractmethod
    def set_calib_pts(self):
        pass

    @abstractmethod
    def go_to_well(self, well, offset):
        pass

    def _get_center_to_corner_offset_px(self):
        # Compute home in px units assuming TL mosaic tile is centered on home
        return np.array(
            [
                IMG_X_PX * self.px2um / 2,
                IMG_Y_PX * self.px2um / 2,
            ]
        )

    def set_center_to_corner_offset_um(self, px_home ):
        # Manually set home in px units
        self.um_center_to_corner_offset = np.multiply(
            self._get_center_to_corner_offset_px(),
            self.px2um
        )

    def calc_transform(self):
        # User needs to previously set home in TL slot and navigate to BR corner before pressing "calibrate"
        vector_actual = self.um_BR[0:2] - self.um_TL[0:2]
        theta_actual = np.arctan(vector_actual[1] / vector_actual[0])

        # User needs to previously load wells
        all_wells = np.reshape(np.array(self.plate_data['wells']['well_coordinates']), (-1, 2))
        vector_expected = np.max(all_wells, axis=0)
        theta_expected = np.arctan(vector_expected[1] / vector_expected[0])

        self.theta_diff = theta_actual - theta_diff

        self.transform = np.array(
            [
                [np.cos(self.theta_diff), np.sin(self.theta_diff)],
                [-np.sin(self.theta_diff), np.cos(self.theta_diff)]
            ]
        )

    def get_transform(self):
        # Get transformation matrix and corresponding angle
        return self.transform, self.theta_diff

    def apply_transform(self, pos):
        # Assume input is a np array, with each position as a row array [[x1; y1], [x2, y2], ...] 

        # Ideally, user has previously set transform
        # TODO: Add user prompt if not
        return np.matmul(pos, self.transform)

    def load_wells(self):
        # User needs to previously set home in TL slot and set transform
        # TODO: Add user prompt

        # Load metadata
        # TODO redo file compatibility
        well_names = self.plate_data['wells']['well_names']
        unformatted_well_pos = np.array(self.plate_data['wells']['well_coordinates'])

        # Format well positions
        well_count = int(self.plate_data['array_design']['rows']) * int(self.plate_data['array_design']['columns'])
        well_pos = unformatted_well_pos.reshape(well_count, 2)

        # Transform wells
        transformed_well_pos = self._apply_transform(well_pos)
        abs_well_pos = self.rel_um_to_abs_um(transformed_well_pos)
        px_well_pos = self.rel_um_to_px(transformed_well_pos)

        # Load sequence
        self.wells = {
            'array_design' : self.plate_data['array_design'],
            'names': well_names,
            'raw_rel_um' : well_pos,
            "calib_rel_um": transformed_well_pos,
            "calib_abs_um": abs_well_pos,
            "calib_px": px_well_pos, # NOTE px is unused for dispense plate
        }
        logging.info(f'wells {self.wells}')

    def get_well_id(self, well_name: str):
        return self.wells['names'].index(well_name)

    def get_abs_um_from_well_name(self, well_name: str):
        return self.wells['calib_abs_um'][self.get_well_id(well_name)]

    def get_px_from_well_name(self, well_name: str):
        return self.wells['calib_px'][self.get_well_id(well_name)]

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
        return rel_um_pos + self.um_TL

    def abs_um_to_rel_um(self, abs_um_pos):
        # Stage coords to wellplate coords
        return abs_um_pos - self.um_TL

    def px_to_abs_um(self, px_pos):
        # Image coords to stage coords
        return (px_pos * self.px2um) - self.um_center_to_corner_offset + self.um_TL

    def abs_um_to_px(self, abs_um_pos):
        # Stage coords to image coords
        return (abs_um_pos - self.um_TL + self.um_center_to_corner_offset) / self.px2um

