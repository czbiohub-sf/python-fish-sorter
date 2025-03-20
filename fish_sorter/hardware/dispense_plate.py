# TODO manually set grid array
# TODO if user manually overrides grid update transform?
import json
import logging
import numpy as np
from typing import Optional

from fish_sorter.helpers.mapping import Mapping

TL_WELL_NAME = 'TL_well'
TR_WELL_NAME = 'TR_well'

MM_TO_UM = 1000.0

# NOTE calibrate by setting positions in UI. Replace with dialogs?

# TODO create widget

# QUESTION should we switch to mm instead of um?

class DispensePlate(Mapping):
    def __init__(self, zc, array_file):
        self.zc = zc
        super().__init__(array_file)

    def set_calib_pts(self, pipettor_cfg=None):
        # MK TODO don't use optional argument here!
        logging.info(f'in dplate: {pipettor_cfg}')
        self.set_calib_pts_default(pipettor_cfg)

    def set_calib_pts_default(self, pipettor_cfg):
        with open(pipettor_cfg) as f:
            self.cfg_data = json.load(f)
            logging.info(f'in dplate: {self.cfg_data}')
        self.um_TL = np.array(
            [
                self.cfg_data['dispense_plate']['TL_corner']['x'],
                self.cfg_data['dispense_plate']['TL_corner']['y'],
            ]
        )
        self.um_TR = (
            [
                self.cfg_data['dispense_plate']['TR_corner']['x'],
                self.cfg_data['dispense_plate']['TR_corner']['y'],
            ]
        )
        logging.info(f'{self.cfg_data}')

    def set_calib_pts_manually(self):
        # TODO prompt home
        x = self.get_pos('x') * MM_TO_UM
        y = self.get_pos('y') * MM_TO_UM
        self.um_TL = np.array([x, y])

        # TODO prompt calib point
        sleep(5)
        x = self.get_pos('x') * MM_TO_UM
        y = self.get_pos('y') * MM_TO_UM
        self.um_TR = np.array([x, y])

        # For temporary testing only
        print(f'HOME={self.um_TL}\nTR={self.um_TR}')

    def go_to_well(self, well: Optional[str], offset=np.array([0,0])):
        logging.info(f'move to well {well}')
        if well is not None:
            x, y = self._get_well_pos(well, offset)
            self.move_arm('x', x / MM_TO_UM, is_relative=False)
            self.move_arm('y', y / MM_TO_UM, is_relative=False)
