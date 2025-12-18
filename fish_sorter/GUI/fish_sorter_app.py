import argparse
import datetime
import json
import logging
import napari
import numpy as np
import os
import re
import sys
import types

from napari.utils.colormaps import Colormap
from pathlib import Path
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QTabWidget,
    QVBoxLayout,
    QWidget
)
from qtpy.QtCore import Qt, QTimer
from tifffile import imwrite
from typing import overload
from useq import GridFromEdges, MDASequence

from fish_sorter.GUI.classify import Classify
from fish_sorter.GUI.image_gui import ImageWidget
from fish_sorter.GUI.picking import Pick
from fish_sorter.GUI.picking_gui import PickGUI
from fish_sorter.GUI.selection_gui import SelectGUI
from fish_sorter.GUI.setup_gui import SetupWidget
from fish_sorter.hardware.imaging_plate import ImagingPlate
from fish_sorter.hardware.picking_pipette import PickingPipette
from fish_sorter.constants import CAM_PX_UM, CAM_X_PX, CAM_Y_PX
from fish_sorter.helpers.mosaic import Mosaic
from fish_sorter.paths import MM_DIR

# For simulation
try:
    from mda_simulator.mmcore import FakeDemoCamera
except ModuleNotFoundError:
    FakeDemoCamera = None

os.environ['MICROMANAGER_PATH'] = MM_DIR
# micromanager_path = os.environ.get('MICROMANAGER_PATH')

# Setup logger
log_dir = "../log"
Path(log_dir).mkdir(parents=True, exist_ok=True)
LOGGING_DATETIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
logging.basicConfig(filename=f"{log_dir}/{LOGGING_DATETIME}_log", filemode='a',
                    format='%(asctime)s - %(message)s', level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
log.setLevel(logging.INFO)

# Logging to terminal
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s', '%Y-%m-%d %H: %M: %S')
handler.setFormatter(formatter)
log.addHandler(handler)

class FishSorter:
    def __init__(self, sim=False):
        self.expt_parent_dir = Path("D:/fishpicker_expts/")
        self.cfg_dir = Path(__file__).parent.parent.absolute() / "configs/"
        self.v = napari.Viewer()
        self.dw, self.main_window = self.v.window.add_plugin_dock_widget("napari-micromanager")
        qtwindow = self.v.window._qt_window
        qtwindow.addDockWidget(Qt.TopDockWidgetArea, self.dw)
        self.dw.show()
        
        logging.info('Loading mmcore')
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

        logging.info('Initialize picking hardware controller')
        self.phc = PickingPipette(self.cfg_dir)
        # Load sequence and Mosaic class
        self.image_init()
        self.assign_widgets()
        self.main_window._show_dock_widget("MDA")

        napari.run()

    def assign_widgets(self):
        
        # Setup
        self.setup = SetupWidget(self.cfg_dir)
        self.v.window.add_dock_widget(self.setup, name = 'Setup', area='right', tabify=True)
        self.setup.pick_setup.clicked.connect(self.setup_picker)
    
        # Picking GUI Widget
        self.pick = Pick(self.phc)
        self.pick_gui = PickGUI(self.pick)
        self.pick_gui.new_expt.new_exp_req.connect(self._new_exp)
        self.pick_gui.calib_pick.save_pick_h.connect(self._save_pick_h)
        self.v.window.add_dock_widget(self.pick_gui, name='Picking', area='right', tabify=True)
    
    def image_init(self):
        """Sets up the mosaic, MDA, and image tools widget
        """

        self.mosaic = Mosaic(self.v)
        self.mda = None

        # Image Manipulation Widget
        self.img_tools = ImageWidget(self.v)
        self.wrap_widget = QWidget()
        self.ww_layout = QVBoxLayout()
        self.wrap_widget.setLayout(self.ww_layout)
        self.ww_layout.addWidget(self.main_window)
        self.ww_layout.addWidget(self.img_tools)
        self.dw.setWidget(self.wrap_widget)

        self.img_tools.mosaic_btn.clicked.connect(self.run)
        self.img_tools.class_btn.clicked.connect(self.run_class)
        self.core.events.pixelSizeChanged.connect(self.main_mag)

        self.main_mag()
        self.setup_MDA()

    def main_mag(self):
        """Main level callback on the objective change (adjusts magnification and images)
        """

        self.img_tools.get_mag()

        if 'crosshairs' in self.v.layers:
            self.img_tools._create_crosshairs()

    def run_class(self):
        """Classification GUI startup from image widget class_btn
        """
            
        logging.info('Start Classification')
        sequence = self.mda.value()
        mosaic_metadata = self.mosaic.get_mosaic_metadata(sequence)

        self.iplate.set_calib_pts()
        self.iplate.load_wells(grid_list=self.mosaic.grid_list)

        self.classify = Classify(self.cfg_dir, self.pick_type, self.expt_prefix, self.expt_path, self.iplate, self.v)
        self.v.reset_view()

    def setup_picker(self):
        """After collecting required setup information, setup the picker
        """

        sequence = self.mda.value()
        self.main_mag()
        update_fov_gp = sequence.grid_plan.replace(fov_width=self.img_tools.fov_w, fov_height=self.img_tools.fov_h)
        update_fov_seq = sequence.replace(grid_plan=update_fov_gp) 
        new_seq = self.mda.setValue(update_fov_seq)

        self.expt_path = sequence.metadata['pymmcore_widgets']['save_dir'].strip()
        self.expt_prefix = sequence.metadata['pymmcore_widgets']['save_name'].removesuffix('.ome.zarr')
        settings_path = Path(self.expt_path) / f'{self.expt_prefix}_settings'
        mda = self.mda.value()
        logging.info(f'Saving MDA sequence: {mda} to {settings_path}')
        self.mda.save(settings_path)

        self.img_array = self.setup.get_img_array()
        self.dp_array = self.setup.get_dp_array()
        self.pick_type, offset, dtime, pick_h = self.setup.get_pick_type()

        logging.info('Picker setup parameters: ')
        logging.info(f'Expt Path: {self.expt_path}')
        logging.info(f'Expt Prefix: {self.expt_prefix}')
        logging.info(f'Image array: {self.img_array}')
        logging.info(f'Dispense array: {self.dp_array}')
        logging.info(f'cfg dir: {self.cfg_dir}')
        logging.info(f'Pick type: {self.pick_type}')
        logging.info(f'Pick offset: {offset}')
        logging.info(f'Pick delay time: {dtime}')
        logging.info(f'Previous pick height: {pick_h}')

        self.setup_iplate()

        logging.info('Enabling full pick functionality')
        self.pick.setup_exp(self.cfg_dir, self.expt_path, self.expt_prefix, offset, dtime, pick_h, self.iplate, self.dp_array, self.img_tools.pixel_size_um)
        self.pick_gui.update_pick_widgets(status=True)
        self._pick_selection_gui()

    def setup_MDA(self):
        """Setup the MDA from Picker setup information and the starting configuration
        """

        self.main_window._show_dock_widget("MDA")
        self.mda = self.v.window._dock_widgets.get("MDA").widget()
        sequence = self.mosaic.init_pos(self.img_tools.fov_w, self.img_tools.fov_h)
        self.mda.setValue(sequence)
        seq = self.mda.value()
        new_seq = MDASequence(
            axis_order = seq.axis_order,  
            grid_plan=seq.grid_plan,  
            channels=seq.channels,
            metadata={
                "pymmcore_widgets": {
                "save_dir": str(self.expt_parent_dir),
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
        self.v.reset_view()

        if not hasattr(self, "_mda_finished_connect"):
            self._cancel = False

            def _mda_cancel(seq):
                logging.info('MDA canceled — user aborted acquisition')
                self._cancel = True
            
            def _mda_finish(seq):
                if self._cancel:
                    logging.info('Skipping stitching and classification.')
                else:
                    logging.info('MDA sequence finished. Triggering Mosaic stitching and classification')
                    self.run()
                self._cancel = False
            
            self.core.mda.events.sequenceCanceled.connect(_mda_cancel)
            self.core.mda.events.sequenceFinished.connect(_mda_finish)
            self._mda_finished_connect = True

    def setup_iplate(self):
        """Setup image plate instance to pass to Pick and Classify classes
        """

        array = self.cfg_dir / 'arrays' / self.img_array
        logging.info(f'{array}')
        self.iplate = ImagingPlate(self.core, self.mda, array, self.img_tools.pixel_size_um)
        logging.info('Loaded image plate')

    def run(self):
        """Runs the mosaic processing, dispay and setup of classification
        """

        sequence = self.mda.value()
        img_arr = self.main_window._core_link._mda_handler._tmp_arrays
        self.stitch = self.mosaic.stitch_mosaic(sequence, img_arr)
        mosaic_metadata = self.mosaic.get_mosaic_metadata(sequence)
        num_chan, chan_names = mosaic_metadata[2], mosaic_metadata[3]

        for chan, chan_name in zip(range(num_chan), chan_names):
            mosaic = self.stitch[chan, :, :]
            if chan_name == 'DAPI':
                color = Colormap([[0, 0, 0], [0.16, 0.82, 0.79]], name='DAPI-cyan')
            elif chan_name == 'GFP':
                color = Colormap([[0, 0, 0], [0, 1, 0]], name='GFP-green')
            elif chan_name == 'TXR':
                color = Colormap([[0, 0, 0], [1, 0.25, 0]], name='tiger-orange')
            elif chan_name == 'CIT':
                color = Colormap([[0, 0, 0], [1, 1, 0]], name='CIT-yellow')
            elif chan_name == 'CY5':
                color = Colormap([[0, 0, 0], [0.93, 0.13, 0.53]], name='CY5-plasma')
            else:
                color = 'grey'
            self.v.add_image(mosaic, colormap=color, blending='additive', name=chan_name)

        logging.info('Remove unncessary layers')
        remove_layers = []
        save_layers = []
        for layer in self.v.layers:
            logging.info(f'Layer {layer}')
            if layer.name == 'preview' or layer.name == 'crosshairs' or 'ome.zarr' in layer.name or self.expt_prefix in layer.name:
                remove_layers.append(layer)
            else:
                save_layers.append(layer)
        
        logging.info(f'Remove Layer List: {remove_layers}')
        logging.info(f'Save Layer List: {save_layers}')

        self._remove_layers(remove_layers)
        QTimer.singleShot(500, lambda: self._save_mosaic(save_layers))

    def _remove_layers(self, layers):
        """Safety remove layers to prevent QT crashes

        :param layers: list of layers to save
        :type layers: napari layers
        """

        for layer in layers:
            self.v.layers.remove(layer)
            logging.info(f'Removed layer {layer}')        

    def _save_mosaic(self, layers):
        """Saves the mosaic layers safely to prevent QT crashes

        :param layers: list of layers to save
        :type layers: napari layers
        """

        logging.info('Saving mosaic layers')
        for layer in layers:
            save_path = Path(self.expt_path) / f"{layer.name}.tif"
            imwrite(save_path, layer.data)
            logging.info(f'Saved layer {layer}')

        logging.info('Ready to classify')
        self.run_class()
    
    def _new_exp(self):
        """Set up to start a new experiment after running one
        """

        logging.info('Remove all layers')
        for layer in list(self.v.layers):
            logging.info(f'Layer {layer}')
            self.v.layers.remove(layer)
            logging.info(f'Removed layer {layer}')
        
        if hasattr(self, 'classify') and self.classify is not None:
            try:
                if hasattr(self.classify, 'classify_widget'):
                    self.v.window.remove_dock_widget(self.classify.classify_widget)
                if hasattr(self.classify, 'save_widget'):
                    self.v.window.remove_dock_widget(self.classify.save_widget)
                if hasattr(self.classify, 'fish_widget'):
                    self.v.window.remove_dock_widget(self.classify.fish_widget)
            except Exception as e:
                logging.warning(f'Could not remove classify dock widget: {e}')
        self.classify = None

        if hasattr(self, 'selection') and self.selection is not None:
            try:
                    self.v.window.remove_dock_widget(self.selection)
            except Exception as e:
                logging.warning(f'Could not remove selection dock widget: {e}')
        self.selection = None

    def _save_pick_h(self):
        """Saves the calibrated pick height for the specific pick type to the pick type config
        """

        logging.info("Saving pick height to the pick type config")
        cfg = self.cfg_dir / 'pick/pick_type_config.json' 
        with open(cfg, 'r') as pc:
            pick_cfg = json.load(pc)
            pick_cfg[self.pick_type]['picker']['pick_height'] = self.phc.pick_h
            pc.close()
        with open(cfg, 'w') as pc:
            pick_update = json.dump(pick_cfg, pc, indent = 4, separators= (',',': '))
            pc.close()

        logging.info('Saved pick_type_config.json with updated value')

    def _pick_selection_gui(self):
        """Loads pick selection gui on save of classification
        """

        if hasattr(self, 'selection') and self.selection is not None:
            if self.selection.isVisible():
                self.selection.raise_()
                self.selection.setFocus()
                return

        self.selection = SelectGUI(self.pick, self.pick_type)
        self.v.window.add_dock_widget(self.selection, name = 'Pick Selection', area='right', tabify=True)
        self.selection.destroyed.connect(lambda: setattr(self, 'selection', None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="FishSorter"
    )
    parser.add_argument('-s', '--sim', action='store_true')
    args = parser.parse_args()

    FishSorter(sim=args.sim)