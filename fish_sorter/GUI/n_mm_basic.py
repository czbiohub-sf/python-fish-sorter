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
            cfg_dir = Path().absolute().parent / "python-fish-sorter/fish_sorter/configs/micromanager"
            cfg_file = "20240718 - LeicaDMI - AndorZyla.cfg"
            cfg_path = cfg_dir / cfg_file
            logging.info(f'{cfg_path}')
            self.core.loadSystemConfiguration(str(cfg_path))

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