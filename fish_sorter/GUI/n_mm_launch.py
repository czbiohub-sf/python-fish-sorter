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

from fish_sorter.GUI.classify import Classify
from fish_sorter.GUI.picking import Pick
from fish_sorter.GUI.picking_gui import Picking
from fish_sorter.GUI.setup_gui import SetupWidget
# TODO delete this
from fish_sorter.helpers.mosaic import Mosaic
from fish_sorter.GUI.tester_gui import TesterWidget

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

        self.cfg_dir = Path().absolute().parent / "fish_sorter/configs/"
        
        #TODO replace with the local directory to where experiments are saved
        self.expt_parent_dir = Path().absolute().parent

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

        # Load and push sequence
        self.mosaic = Mosaic(self.v)
        self.assign_widgets(self.mosaic.set_sequence())

        napari.run()

    def assign_widgets(self, sequence):
        
        #Setup
        self.setup = SetupWidget(self.cfg_dir, self.expt_parent_dir)
        self.v.window.add_dock_widget(self.setup, name = 'Setup', area='right')
        self.start_setup()
        
        # MDA
        self.main_window._show_dock_widget("MDA")
        self.mda = self.v.window._dock_widgets.get("MDA").widget()
        self.mda.setValue(sequence)
        # Move destination plate for fluorescence imaging with pipette tip
        self.pick.move_for_calib(pick=False)
        self.v.window._qt_viewer.console.push(
            {"main_window": self.main_window, "mmc": self.core, "sequence": sequence, "np": np}
        )

        # Stitch Mosaic
        self.stitch = MosaicWidget(sequence)
        self.v.window.add_dock_widget(self.stitch, name='Stitch Mosaic')
        self.stitch.btn.clicked.connect(self.run)
        # self.tester.calibrate.clicked.connect(self.set_home)
        # self.tester.pos.clicked.connect()

    
    def run(self):
        """Runs the mosaic processing, dispay and setup of classification
        """

        sequence = self.mda.value()
        img_arr = self.main_window._core_link._mda_handler._tmp_arrays 
        self.stitch = self.mosaic.stitch_mosaic(sequence, img_arr)
        rows, cols, num_chan, chan_name, overlap, idxs = self.mosaic.get_mosaic_metadata(sequence)

        # TODO are any images open, if so close prior to loading mosaic
        for chan in num_chan:
            mosaic = self.stitch[chan, :, :]
            if chan_name == 'FITC':
                color = 'green'
            elif chan_name == 'TXR':
                color = 'red'
            else:
                color = 'grey'
            self.v.add_image(mosaic, colormap=color, opacity=0.5, name=chan_name)

        logging.info('Start Classification')
        # TODO make sure all of the input parameters are here
        self.classify = Classify(self.cfg_dir, self.array_type, self.core, self.mda, self.pick_type, self.expt_prefix, self.expt_path, self.v)
        logging.info('Completed Classification')

    def start_setup(self):
        """
        Collect setup information and initialize picking hardware
        """

        # TODO need to collect both the imaging array type and the dispense plate array type

        self.expt_path = self.setup.get_expt_path()
        self.expt_prefix = self.setup.get_expt_prefix()
        self.array_type = self.setup.get_array()
        self.pick_type, self.offset = self.setup.get_pick_type()
        self.pick = Pick(self.cfg_dir, self.expt_path, self.expt_prefix, self.offset, self.core, self.mda)
        logging.info('Loaded picking hardware')
        
        logging.info('Load picker GUI')
        self.picking = Picking(self.pick)
        self.v.window.add_dock_widget(self.picking, name='Picking')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="FishSorter"
    )
    parser.add_argument('-s', '--sim', action='store_true')
    args = parser.parse_args()

    nmm(sim=args.sim)