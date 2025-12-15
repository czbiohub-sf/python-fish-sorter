import logging
import napari
import napari_micromanager
import numpy as np
import os
import pymmcore_plus
import argparse
import types

from pathlib import Path
from useq import MDASequence, Position

from fish_sorter.GUI.pipette_test_gui import PipetteWidget
from fish_sorter.paths import MM_DIR

# For simulation
try:
    from mda_simulator.mmcore import FakeDemoCamera
except ModuleNotFoundError:
    FakeDemoCamera = None

os.environ['MICROMANAGER_PATH'] = MM_DIR
# micromanager_path = os.environ.get('MICROMANAGER_PATH')

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
            self.cfg_dir = Path().absolute().parent / "python-fish-sorter/fish_sorter/configs/"
            mm_dir = self.cfg_dir / "micromanager"
            if mm_dir.exists() and mm_dir.is_dir():
                mm_cfg_files = list(mm_dir.glob("*.cfg"))
                if mm_cfg_files:
                    mm_cfg_path = mm_cfg_files[0]
                    logging.info(f'Micromanager config: {mm_cfg_path}')
                    self.core.loadSystemConfiguration(str(mm_cfg_path))
                else:
                    logging.critical("Micromanager config file not found")
            else:
                logging.critical("Micromanager config folder does not exisit")

        self.sequence = self._get_seq()
        logging.info(f'{self.sequence}')

        # Load and push sequence
        self.assign_widgets()

        napari.run()

    def _get_seq(self):

        sequence = MDASequence(
            channels = [
                {"config": "GFP","exposure": 100}, 
                {"config": "TXR", "exposure": 100}
            ],
            stage_positions = [
                {"x": 0.0, "y": 0.0, "z": 0.0, "name": "TL_well"},
                {"x": 100.0, "y": 0.0, "z": 0.0, "name": "TR_well"},
                {"x": 1000.0, "y": 1000.0, "z": 0.0, "name": "Test_well"},

            ],
            axis_order = "pc",
        )
        return sequence

    def assign_widgets(self):
        # MDA
        self.main_window._show_dock_widget("MDA")
        self.mda = self.v.window._dock_widgets.get("MDA").widget()
        self.mda.setValue(self.sequence)
        self.v.window._qt_viewer.console.push(
            {"main_window": self.main_window, "mmc": self.core, "sequence": self.sequence, "np": np}
        )

        # Pipette
        self.pipette = PipetteWidget()
        self.v.window.add_dock_widget(self.pipette, name='pipette')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="FishSorter"
    )
    parser.add_argument('-s', '--sim', action='store_true')
    args = parser.parse_args()

    nmm(sim=args.sim)