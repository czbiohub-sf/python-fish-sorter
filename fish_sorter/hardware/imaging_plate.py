# TODO manually set grid array
# TODO if user manually overrides grid update transform?
import logging
import numpy as np
from typing import Optional

from fish_sorter.helpers.mapping import Mapping

# NOTE calibrate by setting positions in UI. Replace with dialogs? 
class ImagingPlate(Mapping):
    def __init__(self, mmc, mda, array_file):
        self.mda = mda
        super().__init__(array_file)

    def set_calib_pts(self):
        seq = self.mda.value()

        # TODO initialize position list with these names
        self.um_TL = np.array([seq.grid_plan.left, seq.grid_plan.top])
        self.um_BR = np.array([seq.grid_plan.right, seq.grid_plan.bottom])
        
        # TODO throw an exception if calib was not set

    def go_to_well(self, well: Optional[str], offset=np.array([0,0])):
        if well is not None:
            x, y = self._get_well_pos(well, offset)
            # Move z pos too?
            self.mmc.run_mda(Position(x=x, y=y, name=well))
