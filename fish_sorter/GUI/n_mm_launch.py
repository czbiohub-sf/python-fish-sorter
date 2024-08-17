import napari
import numpy as np
import os
import argparse

from pathlib import Path
from gui.pipette_gui import PipetteWidget

from gui.pipette_gui import PipetteWidget
# TODO delete this
from helpers.mosaic import MosaicHandler
from gui.tester_gui import TesterWidget

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
            print(cfg_path)
            self.core.loadSystemConfiguration(str(cfg_path))

        # Load and push sequence
        self.mosaic = MosaicHandler(self.v)
        self.assign_widgets(self.mosaic.get_sequence())

        napari.run()

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

        # Pipette
        self.pipette = PipetteWidget()
        self.v.window.add_dock_widget(self.pipette, name='pipette')

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