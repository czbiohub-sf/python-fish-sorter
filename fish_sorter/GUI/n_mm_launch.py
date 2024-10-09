import argparse
import logging
import napari
import numpy as np
import os
import types

from pathlib import Path
from typing import overload

from pymmcore_plus import DeviceType
from pymmcore_widgets import StageWidget
# from pymmcore_widgets.hcs._plate_calibration_widget import PlateCalibrationWidget

from fish_sorter.gui.pipette_gui import PipetteWidget
# TODO delete this
from fish_sorter.helpers.mosaic import Mosaic
from fish_sorter.gui.tester_gui import TesterWidget

from PyQt5.QtWidgets import QGroupBox, QHBoxLayout

# For simulation
try:
    from mda_simulator.mmcore import FakeDemoCamera
except ModuleNotFoundError:
    FakeDemoCamera = None

os.environ['MICROMANAGER_PATH'] = "C:/Program Files/Micro-Manager-2.0-20240130"
micromanager_path = os.environ.get('MICROMANAGER_PATH')


class nmm:
    def __init__(self, sim=False):

        self.v = napari.Viewer()
        dw, self.main_window = self.v.window.add_plugin_dock_widget("napari-micromanager")
        
        self.core = self.main_window._mmc
        # Overwrite default function so that image is mirrored
        self.core.getImage = types.MethodType(self.getImageMirrored, self.core)

        if sim:
            if FakeDemoCamera is not None:
                # override snap to look at more realistic images from a microscoppe
                # with underlying random walk simulation of spheres
                # These act as though "Cy5" is BF and other channels are fluorescent
                fake_cam = FakeDemoCamera(timing=2)
                # make sure we start in a valid channel group
                self.core.setConfig("Channel", "Cy5")
        else:
            cfg_dir = Path().absolute().parent / "fish_sorter/configs/micromanager"
            cfg_file = "20240718 - LeicaDMI - AndorZyla.cfg"
            cfg_path = cfg_dir / cfg_file
            logging.info(f'Micromanager config: {cfg_path}')
            self.core.loadSystemConfiguration(str(cfg_path))

        # Load and push sequence
        self.mosaic = Mosaic(self.v)
        self.assign_widgets(self.mosaic.get_sequence())

        napari.run()

    # # Overload copied from super class (pymmcore_plus/core/_mmcore_plus.py)
    # @overload
    # def getImageMirrored(self, *, fix: bool = True) -> np.ndarray:  # noqa: D418
    #     """Return the internal image buffer."""

    # # Overload copied from super class (pymmcore_plus/core/_mmcore_plus.py)
    # @overload
    # def getImageMirrored(self, numChannel: int, *, fix: bool = True) -> np.ndarray:  # noqa
    #     """Return the internal image buffer for a given Camera Channel."""

    @overload
    def getImageMirrored(
        self, numChannel: int | None = None, *, fix: bool = True
    ) -> np.ndarray:
        # Mirror image
        print('flip')
        return np.flip(self.core.getImage(numChannel), axis=1)

    def assign_widgets(self, sequence):
        # MDA
        self.main_window._show_dock_widget("MDA")
        self.mda = self.v.window._dock_widgets.get("MDA").widget()
        self.mda.setValue(sequence)
        self.v.window._qt_viewer.console.push(
            {"main_window": self.main_window, "mmc": self.core, "sequence": sequence, "np": np}
        )

        # Tester
        # TODO delete
        self.tester = TesterWidget(sequence)
        self.v.window.add_dock_widget(self.tester, name='tester')
        self.tester.btn.clicked.connect(self.run)
        # self.tester.calibrate.clicked.connect(self.set_home)
        # self.tester.pos.clicked.connect()

        # Pipette
        self.pipette = PipetteWidget()
        self.v.window.add_dock_widget(self.pipette, name='pipette')

        # # Stage
        # stages = list(self.core.getLoadedDevicesOfType(DeviceType.XYStage))
        # stages.extend(self.core.getLoadedDevicesOfType(DeviceType.Stage))
        # for stage in stages:
        #     lbl = "Z" if self.core.getDeviceType(stage) == DeviceType.Stage else "XY"
        #     bx = QGroupBox(f"{lbl} Control")
        #     bx_layout = QHBoxLayout(bx)
        #     bx_layout.setContentsMargins(0, 0, 0, 0)
        #     bx_layout.addWidget(StageWidget(device=stage))
        #     self.v.window.add_dock_widget(bx)

    def run(self):
        sequence = self.mda.value()
        img_arr = self.main_window._core_link._mda_handler._tmp_arrays
        self.mosaic.stitch_mosaic(sequence, img_arr)
        # self.mosaic.get_mosaic_metadata(sequence)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="FishSorter"
    )
    parser.add_argument('-s', '--sim', action='store_true')
    args = parser.parse_args()

    nmm(sim=args.sim)