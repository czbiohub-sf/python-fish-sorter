# TODO clean up the imports
import logging
import numpy as np
import os
import json

from abc import ABC, abstractmethod

# TODO dynamically load pixel count
from fish_sorter.constants import (
    IMG_X_PX,
    IMG_Y_PX,
    PIXEL_SIZE_UM,
)

# NOTE TL corner needs to have image center at corner,
#      unless manually overriden by set_center_to_corner_offset_um
# TODO add type hints

class Mapping:
    def __init__(self, array_file):
        # NOTE Does mda return z values?
        self.um_TL = None
        self.um_BR = None

        # self.px2um = self.mmc.getPixelSizeUm() # Automatically load pixel size
        self.um_center_to_corner_offset = np.array([0.0, 0.0])
        self.px_center_to_corner_offset = np.array(
                [
                    IMG_X_PX / 2,
                    IMG_Y_PX / 2,
                ]
            )

        self.transform_exp2actual = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0]
            ]
        )
        self.wells = {}

        with open(array_file) as f:
            self.plate_data = json.load(f)

        logging.info(f'plate data: {self.plate_data}')

        # TODO save TL/BR locations in experiment savefile

    @abstractmethod
    def set_calib_pts(self, pipettor_cfg=None):
        pass

    @abstractmethod
    def go_to_well(self, well, offset):
        pass

    def calc_transform(self, exp_rel_um):
        # Compute transformation from expected rel pos [mm] to actual rel pos [mm]

        self.um_center_to_corner_offset = self.um_TL[0:2]

        vector_actual = self.um_BR[0:2] - self.um_TL[0:2]
        theta_actual = np.arctan(vector_actual[1] / vector_actual[0])

        # User needs to previously load wells
        vector_expected = np.max(exp_rel_um, axis=0)
        theta_expected = np.arctan(vector_expected[1] / vector_expected[0])

        theta_diff = theta_expected - theta_actual        
        theta_transform = np.array(
            [
                [np.cos(-theta_diff), np.sin(-theta_diff)],
                [-np.sin(-theta_diff), np.cos(-theta_diff)]
            ]
        )

        scale = np.sqrt(
            (vector_actual[1]**2 + vector_actual[0]**2) / (vector_expected[1]**2 + vector_expected[0]**2)
        )
        scale_transform = np.array(
            [
                [scale, 0],
                [0, scale]
            ]
        )

        self.transform_exp2actual = np.dot(theta_transform, scale_transform)

    def get_transform(self):
        # Get transformation matrix and corresponding angle
        return self.transform_exp2actual

    def exp_to_actual(self, pos):
        # Assume input is a np array, with each position as a row array [[x1; y1], [x2, y2], ...] 

        # Ideally, user has previously set transform
        # TODO: Add user prompt if not
        return np.matmul(pos, self.transform_exp2actual)

    def actual_to_exp(self, pos):
        return np.matmul(pos, np.linalg.inv(self.transform_exp2actual))

    def load_wells(self, xflip=False):

        # Load metadata
        well_names = self.plate_data['wells']['well_names']
        unformatted_well_pos = np.array(self.plate_data['wells']['well_coordinates'])

        # Format well positions
        exp_rel_um = unformatted_well_pos.reshape(-1, 2)
        if xflip:
            vector_expected = np.matmul(exp_rel_um, np.array([[-1,0], [0,1]]))

        self.calc_transform(exp_rel_um)

        # Transform wells
        actual_rel_um = self.exp_to_actual(exp_rel_um)
        actual_abs_um = self.rel_um_to_abs_um(actual_rel_um)
        px_pos = self.rel_um_to_px(actual_rel_um)

        # Load sequence
        self.wells = {
            'array_design' : self.plate_data['array_design'],
            'names': well_names,
            'exp_rel_um' : exp_rel_um,
            "actual_abs_um": actual_abs_um,
            "actual_px": px_pos, # NOTE px is unused for dispense plate
        }
        logging.info(f'wells {self.wells}')

    def get_well_id(self, well_name: str):
        return self.wells['names'].index(well_name)

    def get_abs_um_from_well_name(self, well_name: str):
        return self.wells['actual_abs_um'][self.get_well_id(well_name)]

    def get_px_from_well_name(self, well_name: str):
        return self.wells['actual_px'][self.get_well_id(well_name)]

    def _get_well_pos(self, well_name: str, offset):
        if well_name not in self.wells['names']:
            return

        pos = self.wells['actual_abs_um'][self.get_well_id(well_name)]
        x = pos[0] + offset[0]
        y = pos[1] + offset[1]

        return x, y

    def px_to_rel_um(self, px_pos):
        # Wellplate coords to stage coords
        return (px_pos - self.px_center_to_corner_offset) * PIXEL_SIZE_UM

    def rel_um_to_px(self, rel_um_pos):
        # Wellplate coords to image coords        
        return (rel_um_pos / PIXEL_SIZE_UM) + self.px_center_to_corner_offset

    def rel_um_to_abs_um(self, rel_um_pos):
        # Wellplate coords to stage coords
        return rel_um_pos + self.um_center_to_corner_offset

    def abs_um_to_rel_um(self, abs_um_pos):
        # Stage coords to wellplate coords
        return abs_um_pos - self.um_center_to_corner_offset

    def px_to_abs_um(self, px_pos):
        # Image coords to stage coords
        return self.rel_um_to_abs_um(self.px_to_rel_um(px_pos))

    def abs_um_to_px(self, abs_um_pos):
        # Stage coords to image coords
        return self.rel_um_to_px(self.abs_um_to_rel_um(abs_um_pos))

