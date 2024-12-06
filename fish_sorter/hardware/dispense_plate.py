# TODO manually set grid array
# TODO if user manually overrides grid update transform?

from fish_sorter.helpers.mapping import Mapping


TL_WELL_NAME = 'TL_well'
TR_WELL_NAME = 'TR_well'

MM_TO_UM = 1000.0

# NOTE calibrate by setting positions in UI. Replace with dialogs?

# TODO create widget

# QUESTION should we switch to mm instead of um?

class DispensePlate(Mapping):
    def __init__(self, mmc, zc):
        self.zc = zc
        super().__init__(mmc)

    def set_calib_pts(self):
        # TODO prompt home
        x = self.get_pos('x') * MM_TO_UM
        y = self.get_pos('y') * MM_TO_UM
        self.um_home = np.array([x, y])

        # TODO prompt calib point
        sleep(5)
        x = self.get_pos('x') * MM_TO_UM
        y = self.get_pos('y') * MM_TO_UM
        self.um_TR = np.array([x, y])

        # For temporary testing only
        print(f'HOME={self.um_home}\nTR={self.um_TR}')

    def go_to_well(self, well: str, offset=np.array([0,0])):
        x, y = self._get_well_pos(well, offset)
        self.move_arm('x', x / MM_TO_UM, is_relative=False)
        self.move_arm('y', y / MM_TO_UM, is_relative=False)
