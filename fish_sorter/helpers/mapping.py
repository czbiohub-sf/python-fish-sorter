# TODO clean up the imports
import napari
import napari_micromanager
import numpy as np
import os
import json
import pymmcore_plus

from tqdm import tqdm
from useq import MDASequence, Position

from typing import cast

from pathlib import Path
from useq import MDASequence, Position
from useq._iter_sequence import _sizes, _used_axes, _iter_axis, _parse_axes

# TODO dynamically load pixel count
from constants import (
    IMG_X_PX,
    IMG_Y_PX,
    DISPENSE_PLATE_PREFIX,
    IMAGING_PLATE_PREFIX,
    TL_WELL_NAME,
    TR_WELL_NAME,
)

# TODO standardize coordinate format
# NOTE TL corner needs to have image center at corner TODO add user prompt

# TODO TODO add offset for head

# TODO add type hints

class Mapping:
    def __init__(self, mda, mmc):
        self.mda = mda
        self.mmc = mmc

        self.transform = np.array([[1, 0], [0, 1]])
        self.um_home, = np.array([0, 0, 0])
        self.um_center_to_corner_offset = self._get_center_to_corner_offset_um()

        self.wells = {}

        # TODO save TL/TR locations in experiment savefile

    def _get_center_to_corner_offset_um(self, px2um):
        return np.array(
            [
                IMG_X_PX * px2um / 2,
                IMG_Y_PX * px2um / 2,
            ]
        )

    def _get_calib_pos(self, prefix):
        seq = self.mda.value()

        # TODO initialize position list with these names
        for pos in seq.stage_positions:
            if pos.name == prefix + TL_WELL_NAME:
                TL = np.array([pos.x, pos.y, pos.z])
            if pos.name == prefix + TR_WELL_NAME:
                TR = np.array([pos.x, pos.y, pos.z])
        
        # TODO throw an exception if calib was not set
        return TL, TR

    def _set_home_and_transform(self, prefix):
        # User needs to previously set home in TL slot and navigate to TR corner before pressing "calibrate"
        # TODO: Add user prompt

        TL, TR = self._get_calib_pos(prefix)
        vector = TR[0:2] - TL[0:2]
        theta = np.arctan(vector[1] / vector[0])

        # NOTE Z position is based on TL corner only
        self.home = np.concatenate(TL)

        self.transform = np.array(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)]
            ]
        )
    
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
        self.wells = [
            {
                name : {
                "raw_rel_um":  well_pos,
                "calib_rel_um": transformed_well_pos,
                "abs_um": abs_well_pos,
                "px": px_well_pos,
                }
            } for name, pos in zip(well_names, calib_well_positions)
        ]

    def go_to_well(self, well: str, offset=np.array([0,0])):
        if well not in self.wells:
            return

        xyz = self.wells[well].abs_um
        self.mmc.run_mda(Position(x=xyz[0]+offset[0], y=xyz[1]+offset[1], z=xyz[2], name=well))

    def px_to_rel_um(self, px_pos):
        # Wellplate coords to stage coords
        return (px_pos * px2um) - self.um_center_to_corner_offset

    def rel_um_to_px(self, rel_um_pos):
        # Wellplate coords to image coords
        px2um = self.mmc.getPixelSizeUm()
        
        return (rel_um_pos + self.um_center_to_corner_offset) / px2um

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
