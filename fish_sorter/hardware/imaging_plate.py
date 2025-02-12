# TODO manually set grid array
# TODO if user manually overrides grid update transform?
import numpy as np
from typing import Optional

from fish_sorter.helpers.mapping import Mapping

TL_WELL_NAME = 'TL_well'
TR_WELL_NAME = 'TR_well'

# NOTE calibrate by setting positions in UI. Replace with dialogs? 
class ImagingPlate(Mapping):
    def __init__(self, mmc, mda, filename):
        self.mda = mda
        super().__init__(mmc, filename)

    def set_calib_pts(self):
        seq = self.mda.value()

        # TODO initialize position list with these names
        for pos in seq.stage_positions:
            if pos.name == TL_WELL_NAME:
                self.um_TL = np.array([pos.x, pos.y, pos.z])
            if pos.name == TR_WELL_NAME:
                self.um_TR = np.array([pos.x, pos.y, pos.z])
        
        # TODO throw an exception if calib was not set

    def go_to_well(self, well: Optional[str], offset=np.array([0,0])):
        if well is not None:
            x, y = self._get_well_pos(well, offset)
            # Move z pos too?
            self.mmc.run_mda(Position(x=x, y=y, name=well))
