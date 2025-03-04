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
from PyQt5.QtWidgets import QGroupBox, QHBoxLayout

from fish_sorter.GUI.classify import Classify
from fish_sorter.GUI.picking import Pick
from fish_sorter.GUI.pick_setup_gui import PickSetup
from fish_sorter.GUI.picking_gui import Picking
from fish_sorter.GUI.setup_gui import SetupWidget
from fish_sorter.GUI.mosaic_gui import MosaicWidget
from fish_sorter.helpers.mosaic import Mosaic

from useq import GridFromEdges, MDASequence


# For simulation
try:
    from mda_simulator.mmcore import FakeDemoCamera
except ModuleNotFoundError:
    FakeDemoCamera = None

os.environ['MICROMANAGER_PATH'] = "C:/Program Files/Micro-Manager-2.0-20240130"
micromanager_path = os.environ.get('MICROMANAGER_PATH')

class nmm:
    def __init__(self, sim=False):

        self.expt_parent_dir = Path("D:/fishpicker_expts/")
        self.cfg_dir = Path().absolute().parent / "python-fish-sorter/fish_sorter/configs/"
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
        self.assign_widgets(self.mosaic.init_pos())

        napari.run()

    def assign_widgets(self, sequence):
        
        # Setup
        #TODO delete logging steps if desired
        logging.info(f'Config Dir: {self.cfg_dir}')
        logging.info(f'Parent Expt Path: {self.expt_parent_dir}')
        self.setup = SetupWidget(self.cfg_dir, self.expt_parent_dir)
        self.v.window.add_dock_widget(self.setup, name = 'Setup', area='right')
        
        # MDA
        self.main_window._show_dock_widget("MDA")
        self.mda = self.v.window._dock_widgets.get("MDA").widget()
        self.mda.setValue(sequence)
        # Move destination plate for fluorescence imaging with pipette tip
        
        self.v.window._qt_viewer.console.push(
            {"main_window": self.main_window, "mmc": self.core, "sequence": sequence, "np": np}
        )
        
        # Picker
        logging.info('Load picker GUI')
        self.pick_setup = PickSetup()
        self.v.window.add_dock_widget(self.pick_setup, name='Picker Setup')
        self.pick_setup.setup.clicked.connect(self.setup_picker)

        # Stitch Mosaic
        self.stitch = MosaicWidget(sequence)
        self.v.window.add_dock_widget(self.stitch, name='Stitch Mosaic')
        self.stitch.btn.clicked.connect(self.run)
        self.stitch.dummy.clicked.connect(self.image_now)

    def image_now(self):
        seq = self.mda.value()
        logging.info(f'{self.mda.value()}')
        logging.info(f'{seq.grid_plan}')

        # Update the MDA widget with the modified sequence
        # self.mda.setValue(self.mosaic.set_grid(seq))
        
        # updated_seq = self.mosaic.set_grid(seq)


        new_seq = MDASequence(
            axis_order = seq.axis_order,
            # stage_positions=updated_seq.stage_positions,  
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


        logging.info(f'new sequence: {new_seq}')
        logging.info(f'new sequence axis order: {new_seq.axis_order}')

        self.mda.setValue(new_seq)
        final_seq = self.mda.value()
        logging.info(f'{final_seq}')


    def run(self):
        """Runs the mosaic processing, dispay and setup of classification
        """

        sequence = self.mda.value()
        img_arr = self.main_window._core_link._mda_handler._tmp_arrays
        logging.info(f'{type(img_arr)}') 
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

    def setup_picker(self):
        """After collecting required setup information, setup the picker
        """

        self.expt_path = self.setup.get_expt_path()
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

        logging.info('Loading picking hardware')
        self.pick = Pick(self.cfg_dir, self.expt_path, self.expt_prefix, self.offset, self.core, self.mda, self.img_array, self.dp_array)
        self.picking = Picking(self.pick)
        self.pick.pp.move_for_calib(pick=False)
        self.v.window.add_dock_widget(self.picking, name='Picking')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="FishSorter"
    )
    parser.add_argument('-s', '--sim', action='store_true')
    args = parser.parse_args()

    nmm(sim=args.sim)