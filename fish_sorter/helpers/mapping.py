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

# TODO standardize coordinate format

class Mapping:
    def __init__(self, zaber, mda, mmc):
        self.mda = mda
        self.mmc = mmc
        self.zaber = zaber
        self.home = np.array([0, 0]) # stage units
        self.transform = [[1, 0], [0, 1]]
        self.wells = {}

        # TODO save home location in experiment savefile


    def set_home(self):
        # User needs to navigate to home location (TL corner) before pressing "calibrate"
        # TODO: Add user prompt

        # self.zaber.home_arm(['x','y'])
        self.home = np.array([self.zaber.get_pos('x'), self.zaber.get_pos('y')])

    def get_home(self):
        seq = self.mda.value()

        for pos in seq.stage_positions:
            if pos.name == 'TL_well':
                return (pos.x, pos.y, pos.z)

        return (0.0, 0.0, 0.0)

    def get_calib_point(self):
        seq = self.mda.value()

        for pos in seq.stage_positions:
            if pos.name == 'TR_well':
                return (pos.x, pos.y, pos.z)
        
        # TODO replace this with a constant to match initialized stage_positions
        return (100.0, 0.0, 0.0)

    def px_to_rel_um(self, px_pos):
        # Wellplate coords to stage coords
       
        if self.home is None:
            # TODO add user prompt to set home
            return

        # Assume px pos is 2x1 array or list
        # TODO! make this computationally cleaner
        return [
            (px_pos[0] * self.mmc.getPixelSizeUm()) + self.home[0],
            (px_pos[1] * self.mmc.getPixelSizeUm()) + self.home[1],
        ]

    def rel_um_to_px(self, um_pos):
        # Wellplate coords to image coords
   
        if self.home is None:
            # TODO add user prompt to set home
            return

        # Assume mm pos is 2x1 array or list
        # TODO! make this computationally cleaner
        return [
            (um_pos[0] - self.home[0]) / self.mmc.getPixelSizeUm(),
            (um_pos[1] - self.home[1]) / self.mmc.getPixelSizeUm(),
        ]

    def px_to_abs_um(self, px_pos):
        # Image coords to stage coords
       
        if self.home is None:
            # TODO add user prompt to set home
            return

        # Assume px pos is 2x1 array or list
        # TODO! make this computationally cleaner
        return [
            (px_pos[0] * self.mmc.getPixelSizeUm()) + self.home[0],
            (px_pos[1] * self.mmc.getPixelSizeUm()) + self.home[1],
        ]

    def abs_um_to_px(self, um_pos):
        # Stage coords to image coords
   
        if self.home is None:
            # TODO add user prompt to set home
            return

        # Assume mm pos is 2x1 array or list
        # TODO! make this computationally cleaner
        return [
            (um_pos[0] - self.home[0]) / self.mmc.getPixelSizeUm(),
            (um_pos[1] - self.home[1]) / self.mmc.getPixelSizeUm(),
        ]

    def rel_um_to_abs_um(self, rel_pos):
        # Wellplate coords to stage coords
        # Assume input is a np array, with each position as a row array [[x1; y1], [x2, y2], ...] 
        
        # Ideally, user has previously set home
        # TODO: Add user prompt if not
        return rel_pos += self.home

    def abs_to_rel(self, abs_pos):
        # Stage coords to wellplate coords
        # Assume input is a np array, with each position as a row array [[x1; y1], [x2, y2], ...] 

        # Ideally, user has previously set home
        # TODO: Add user prompt if not
        return rel_pos -= self.home

    def set_transform(self, pos):
        # User needs to previously set home in TL slot and navigate to TR corner before pressing "calibrate"
        # TODO: Add user prompt

        # Assume input is [x1; y1] (ie. col array)
        vector = [
            [self.zaber.get_pos('x') - self.home[0]],
            [self.zaber.get_pos('y') - self.home[1]],
        ]
        # TODO! Does this index correctly
        theta = np.arctan(vector[1] / vector[0])

        self.transform = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])

    def apply_transform(self, pos):
        # Assume input is a np array, with each position as a row array [[x1; y1], [x2, y2], ...] 

        # Ideally, user has previously set transform
        # TODO: Add user prompt if not
        return np.dir(pos, self.transform)

    def load_wells(self, filename):
        # User needs to previously set home in TL slot and set transform
        # TODO: Add user prompt

        with open(filename) as f:
            plate_data = json.load(f)

        # Load metadata
        plate_name = list(plate_data['dest_plates'])[0]
        well_names = plate_data['dest_plates'][plate_name]['names']
        well_pos = np.array(plate_data['dest_plates'][plate_name]['positions']).T

        # Transform wells
        transformed_well_pos = self.apply_transform(well_pos)
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

    def go_to_well(self, well: str):
        if well not in self.wells:
            return

        _, _, z = self.get_home()
        xy = self.wells[well].abs_um
        self.mmc.run_mda(Position(x=xy[0], y=xy[1], z, name=well))

# TODO make this work with dispense plate too