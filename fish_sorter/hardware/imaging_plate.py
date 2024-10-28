# TODO manually set grid array
# TODO if user manually overrides grid update transform?

from fish_sorter.helpers.mapping import Mapping


TL_WELL_NAME = 'TL_well'
TR_WELL_NAME = 'TR_well'

# NOTE calibrate by setting positions in UI. Replace with dialogs? 
class ImagingPlate(Mapping):
    def __init__(self, mmc, mda):
        self.mda = mda
        super().__init__(mmc)

    def set_calib_pos(self):
        seq = self.mda.value()

        # TODO initialize position list with these names
        for pos in seq.stage_positions:
            if pos.name == prefix + TL_WELL_NAME:
                self.um_home = np.array([pos.x, pos.y, pos.z])
            if pos.name == prefix + TR_WELL_NAME:
                self.um_TR = np.array([pos.x, pos.y, pos.z])
        
        # TODO throw an exception if calib was not set

    def go_to_well(self, well: str, offset=np.array([0,0])):
        xyz = self._get_well_pos(well)
        self.mmc.run_mda(Position(x=xyz[0]+offset[0], y=xyz[1]+offset[1], z=xyz[2], name=well))
