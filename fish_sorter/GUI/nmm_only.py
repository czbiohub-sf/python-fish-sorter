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

        self.main_window._show_dock_widget("MDA")
        self.mda = self.v.window._dock_widgets.get("MDA").widget()

        napari.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="FishSorter"
    )
    parser.add_argument('-s', '--sim', action='store_true')
    args = parser.parse_args()

    nmm(sim=args.sim)