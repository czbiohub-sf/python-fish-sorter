import argparse
import logging

import numpy as np
import os

#TODO delete and clean up once figured out

os.environ["VISPY_LOG_LEVEL"] = "DEBUG"
os.environ["VISPY_GL_DEBUG"] = "True"
import napari
from vispy.app import use_app

import types

from pathlib import Path
from pymmcore_plus import DeviceType
from pymmcore_widgets import StageWidget
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QVBoxLayout,
    QWidget
)
from typing import overload
from useq import GridFromEdges, MDASequence

from fish_sorter.GUI.classify import Classify
from fish_sorter.GUI.picking import Pick
from fish_sorter.GUI.picking_gui import Picking
from fish_sorter.GUI.setup_gui import SetupWidget
from fish_sorter.GUI.image_gui import ImageWidget
from fish_sorter.helpers.mosaic import Mosaic

# For simulation
try:
    from mda_simulator.mmcore import FakeDemoCamera
except ModuleNotFoundError:
    FakeDemoCamera = None

os.environ['MICROMANAGER_PATH'] = "C:/Program Files/Micro-Manager-2.0-20240130"
micromanager_path = os.environ.get('MICROMANAGER_PATH')

class nmm:
    def __init__(self, sim=False):
        
        logging.info(f'Napari is using: {napari.__version__}')
        logging.info(f'Vispy is using: {use_app()}')

        self.expt_parent_dir = Path("D:/fishpicker_expts/")
        self.cfg_dir = Path().absolute().parent / "python-fish-sorter/fish_sorter/configs/"
        self.v = napari.Viewer()
        self.dw, self.main_window = self.v.window.add_plugin_dock_widget("napari-micromanager")
        
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

        # Load sequence and Mosaic class
        self.mosaic = Mosaic(self.v)
        self.assign_widgets()

        napari.run()

    def assign_widgets(self):
        
        # Setup
        #TODO delete logging steps if desired
        logging.info(f'Config Dir: {self.cfg_dir}')
        logging.info(f'Parent Expt Path: {self.expt_parent_dir}')
        self.setup = SetupWidget(self.cfg_dir, self.expt_parent_dir)
        self.v.window.add_dock_widget(self.setup, name = 'Setup', area='right')
        self.setup.pick_setup.clicked.connect(self.setup_picker)

        # Image Manipulation Widget
        self.img_tools = ImageWidget(self.v)

        self.wrap_widget = QWidget()
        self.ww_layout = QVBoxLayout()
        self.wrap_widget.setLayout(self.ww_layout)
        self.ww_layout.addWidget(self.main_window)
        self.ww_layout.addWidget(self.img_tools)
        self.dw.setWidget(self.wrap_widget)

        self.img_tools.btn.clicked.connect(self.run)
        self.img_tools.class_btn.clicked.connect(self.run_class)

    def run_class(self):
        """Classification GUI startup from image widget class_btn
        """
        
        remove_layers = []
        for layer in self.v.layers:
            if layer.name == 'preview' or layer.name == 'crosshairs' or self.expt_prefix in layer.name:
                remove_layers.append(layer)
        for layer in remove_layers:
            self.v.layers.remove(layer)
            
        logging.info('Start Classification')
        self.classify = Classify(self.cfg_dir, self.img_array, self.core, self.mda, self.pick_type, self.expt_prefix, self.expt_path, self.v)

    def setup_picker(self):
        """After collecting required setup information, setup the picker
        """

        self.expt_path = self.setup.get_expt_path().strip()
        self.expt_prefix = self.setup.get_expt_prefix()
        self.img_array = self.setup.get_img_array()
        self.dp_array = self.setup.get_dp_array()
        self.pick_type, self.offset = self.setup.get_pick_type()

        logging.info('Picker setup parameters: ')
        logging.info(f'Expt Path: {self.expt_path}')
        logging.info(f'Expt Prefix: {self.expt_prefix}')
        logging.info(f'Image array: {self.img_array}')
        logging.info(f'Dispense array: {self.dp_array}')
        logging.info(f'cfg dir: {self.cfg_dir}')
        logging.info(f'Pick offset: {self.offset}')

        self.setup_MDA()

        logging.info('Loading picking hardware')
        self.pick = Pick(self.cfg_dir, self.expt_path, self.expt_prefix, self.offset, self.core, self.mda, self.img_array, self.dp_array)
        self.picking = Picking(self.pick)
        self.v.window.add_dock_widget(self.picking, name='Picking')

    def setup_MDA(self):
        """Setup the MDA from Picker setup information and the starting configuration
        """

        sequence = self.mosaic.init_pos()
        self.main_window._show_dock_widget("MDA")
        self.mda = self.v.window._dock_widgets.get("MDA").widget()
        self.mda.setValue(sequence)
        seq = self.mda.value()
        new_seq = MDASequence(
            axis_order = seq.axis_order,  
            grid_plan=seq.grid_plan,  
            channels=seq.channels,
            metadata={
                "pymmcore_widgets": {
                "save_dir": self.expt_path.strip(),
                "save_name": self.expt_prefix.strip(),
                "should_save": True,
                },
                "napari_micromanager": {
                    "axis_order": ("g", "c"),
                    "grid_plan": seq.grid_plan
                }
             }
        )
        self.mda.setValue(new_seq)
        final_seq = self.mda.value()
        logging.info(f'Initial MDA setup sequence prior to TL and BR bounds: {final_seq}')
        self.v.window._qt_viewer.console.push(
            {"main_window": self.main_window, "mmc": self.core, "sequence": final_seq, "np": np}
        )

    def run(self):
        """Runs the mosaic processing, dispay and setup of classification
        """

        sequence = self.mda.value()
        img_arr = self.main_window._core_link._mda_handler._tmp_arrays
        self.stitch, self.mosaic_grid = self.mosaic.stitch_mosaic(sequence, img_arr)
        mosaic_metadata = self.mosaic.get_mosaic_metadata(sequence)
        num_chan, chan_names = mosaic_metadata[2], mosaic_metadata[3]

        for chan, chan_name in zip(range(num_chan), chan_names):
            mosaic = self.stitch[chan, :, :]
            if chan_name == 'GFP':
                color = 'green'
            elif chan_name == 'TXR':
                color = 'red'
            else:
                color = 'grey'
            self.v.add_image(mosaic, colormap=color, opacity=0.5, name=chan_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="FishSorter"
    )
    parser.add_argument('-s', '--sim', action='store_true')
    args = parser.parse_args()

    nmm(sim=args.sim)