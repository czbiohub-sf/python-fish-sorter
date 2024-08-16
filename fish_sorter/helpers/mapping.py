# TODO clean up the imports
import napari
import napari_micromanager
import numpy as np
import os
import pymmcore_plus

from tqdm import tqdm

from typing import cast

from pathlib import Path
from useq import MDASequence, Position
from useq._iter_sequence import _sizes, _used_axes, _iter_axis, _parse_axes

from helpers.constants import PIXELS_TO_MM

# TODO standardize coordinate format

class MappingHandler:
    def __init__(self, zaber):
        self.zaber = zaber
        self.home = None # stage units
        self.transform = None

        # TODO save home location in experiment savefile


    def set_home(self):
        # User needs to navigate to home location (TL corner) before pressing "calibrate"
        # TODO: Add user prompt

        # self.zaber.home_arm(['x','y'])
        x_home = self.zaber.get_pos('x')
        y_home = self.zaber.get_pos('y')
        self.home = (x_home, y_home)


    def pixels_to_mm(self, px_pos):
        # Image coords to stage coords
       
        if self.home is None:
            # TODO add user prompt to set home
            return

        # Assume px pos is 2x1 array or list
        # TODO! make this computationally cleaner
        return [
            (px_pos[0] * PIXELS_TO_MM) + self.home[0],
            (px_pos[1] * PIXELS_TO_MM) + self.home[1],
        ]

    def mm_to_px(self, mm_pos):
        # Stage coords to image coords
   
        if self.home is None:
            # TODO add user prompt to set home
            return

        # Assume mm pos is 2x1 array or list
        # TODO! make this computationally cleaner
        return [
            (mm_pos[0] - self.home[0]) / PIXELS_TO_MM,
            (mm_pos[1] - self.home[1]) / PIXELS_TO_MM,
        ]

    def set_transform(self, pos):
        # User needs to previously set home in TL slot and navigate to TR corner before pressing "calibrate"
        # TODO: Add user prompt

        # Assume input is [x1; y1] (ie. col array)
        vector = [
            [self.zaber.get_pos('x') - self.home[0]],
            [self.zaber.get_pos('y') - self.home[1]],
        ]
        # TODO! Does this index correctly
        theta = np.atan(vector[1] / vector[0])

        self.transform = [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ]


    def transform_pos(self, pos):
        # Assume input is [x1; y1] (ie. col array)
        return np.matmul(pos, self.transform)

        

        