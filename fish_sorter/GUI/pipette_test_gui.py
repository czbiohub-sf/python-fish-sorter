import logging
import sys
from json import load
from pathlib import Path
from time import sleep
from typing import List, Optional, Union

from pymmcore_plus import CMMCorePlus
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QPushButton, QSizePolicy, QWidget, QGridLayout

from fish_sorter.hardware.picking_pipette import PickingPipette
from fish_sorter.hardware.zaber_controller import ZaberController

COLOR_TYPES = Union[
    QColor,
    int,
    str,
    Qt.GlobalColor,
    "tuple[int, int, int, int]",
    "tuple[int, int, int]"
]

class PipetteWidget(QWidget):

    def __init__(self, parent: QWidget | None=None):
        
        super().__init__(parent=parent)
        CMMCorePlus.instance()

        xxx = ZaberImageWidget()
        xyz = ZaberHomeWidget()
        zzz = ZaberTestWidget()
        ddd = PipetteDrawWidget()
        vvv = PipetteExpelWidget()
        ppp = PipettePressureWidget()

        layout = QGridLayout(self)
        layout.addWidget(xyz, 1, 0)
        layout.addWidget(xxx, 1, 1)
        layout.addWidget(zzz, 1, 2)
        layout.addWidget(ddd, 2, 0)
        layout.addWidget(vvv, 2, 1)
        layout.addWidget(ppp, 2, 2)

class ZaberImageWidget(QPushButton):
    """A push button widget to move the Zaber stages to image.

    This is linked to the [hardware][zaber_controller] method
    """
    
    def __init__(self, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        self._mmc = CMMCorePlus.instance()

        self._create_button()

    def _create_button(self)->None:
        
        self.setText("Move for Fluor Imaging")
        self.clicked.connect(self._zaber_image)

    def _zaber_image(self)->None:

        cfg_dir = Path().absolute().parent / "python-fish-sorter/fish_sorter/configs/hardware"
        zaber_cfg_file = "zaber_config.json"
        zaber_cfg_path = cfg_dir / zaber_cfg_file
        image_cfg_file = "image_config.json"
        image_cfg_path = cfg_dir / image_cfg_file
        # Initialize and connect to hardware controller
        try:
            with open(zaber_cfg_path, 'r') as f:
                z = load(f)
            zaber_config = z['zaber_config']
            zc = ZaberController(zaber_config, env='prod')
        except Exception as e:
            logging.info("Could not initialize and connect hardware controller")

        with open(image_cfg_path, 'r') as f:
            p = load(f)
        image_config = p['fluor']
        print(image_config)

        stages = ['x', 'y', 'p']
        logging.info('Move for fluorecent imaging')
        logging.info('Swing height')
        zc.move_arm('p', image_config['stage']['p'])
        zc.move_arm('y', image_config['stage']['y'])
        zc.move_arm('x', image_config['stage']['x'])
        zc.disconnect()

class ZaberHomeWidget(QPushButton):
    """A push button widget to connect to the Zaber stage.

    This is linked to the [hardware][zaber_controller] method
    """
    
    def __init__(self, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        self._mmc = CMMCorePlus.instance()

        self._create_button()

    def _create_button(self)->None:
        
        self.setText("Home Zaber")
        self.clicked.connect(self._zaber_home)

    def _zaber_home(self)->None:

        cfg_dir = Path().absolute().parent / "python-fish-sorter/fish_sorter/configs/hardware"
        cfg_file = "zaber_config.json"
        cfg_path = cfg_dir / cfg_file
        # Initialize and connect to hardware controller
        try:
            with open(cfg_path, 'r') as f:
                p = load(f)
            zaber_config = p['zaber_config']
            zc = ZaberController(zaber_config, env='prod')
        except Exception as e:
            logging.critical("Could not initialize and connect hardware controller")
    
        # Test moving the pipette, x, and y stages to max position
        stages = ['x', 'y', 'p']

        logging.info('Move stages to max and back home')
        for stage in stages:
            zc.move_arm(stage, zaber_config['home'][stage])
            sleep(2)
        
        zc.disconnect()

class ZaberTestWidget(QPushButton):
    """A push button widget to connect to the Zaber stage.

    This is linked to the [hardware][zaber_controller] method
    """
    
    def __init__(self, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        self._mmc = CMMCorePlus.instance()

        self._create_button()

    def _create_button(self)->None:
        
        self.setText("Test Zaber")
        self.clicked.connect(self._zaber_test)

    def _zaber_test(self)->None:

        cfg_dir = Path().absolute().parent / "python-fish-sorter/fish_sorter/configs/hardware"
        zaber_cfg_file = "zaber_config.json"
        zaber_cfg_path = cfg_dir / zaber_cfg_file
        picker_cfg_file = "picker_config.json"
        picker_cfg_path = cfg_dir / picker_cfg_file
        # Initialize and connect to hardware controller
        try:
            with open(zaber_cfg_path, 'r') as f:
                z = load(f)
            zaber_config = z['zaber_config']
            zc = ZaberController(zaber_config, env='prod')
        except Exception as e:
            logging.info("Could not initialize and connect hardware controller")

        with open(picker_cfg_path, 'r') as f:
            p = load(f)
        picker_config = p['pipette']
    
        # Test moving the pipette, x, and y stages to max position
        stages = ['x', 'y', 'p']

        logging.info('Move stages to max and back home')
        for stage in stages:
            zc.move_arm(stage, zaber_config['max_position'][stage])
            sleep(2)
            zc.move_arm(stage, zaber_config['home'][stage])
            sleep(2)
        
        logging.info('Move pipette to set locations')
        logging.info('Swing height')
        zc.move_arm('p', picker_config['stage']['pipette_swing']['p'])
        sleep(2)
        zc.move_arm('p', zaber_config['home']['p'])
        sleep(2)
        
        logging.info('Pick height')
        zc.move_arm('p', picker_config['stage']['pick']['p'])
        sleep(2)
        zc.move_arm('p', zaber_config['home']['p'])
        sleep(2)

        logging.info('Clearance height')
        zc.move_arm('p', picker_config['stage']['clearance']['p'])
        sleep(2)
        zc.move_arm('p', zaber_config['home']['p'])
        sleep(2)

        logging.info('Dispense height')
        zc.move_arm('p', picker_config['stage']['dispense']['p'])
        sleep(2)
        zc.move_arm('p', zaber_config['home']['p'])
        sleep(2)

        logging.info('Zaber test complete')

        zc.disconnect()

class PipetteDrawWidget(QPushButton):
    """A push button widget to connect to the valve controller to actuate the draw function

    This is linked to the [hardware][picking_pipette] method
    """
    
    def __init__(self, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        self._mmc = CMMCorePlus.instance()
        self._create_button()

    def _create_button(self)->None:
        
        self.setText("Pipette Draw")
        self.clicked.connect(self._draw)

    def _draw(self)->None:
        # Initialize and connect to hardware controller
        cfg_dir = Path().absolute().parent / "python-fish-sorter/fish_sorter/configs/"
        array_file = Path().absolute().parent / "python-fish-sorter/fish_sorter/configs/arrays/6well_plate20240822.json"
        try:
            pp = PickingPipette(cfg_dir, self._mmc, array_file)
        except Exception as e:
            logging.info("Could not initialize and connect hardware controller")

        logging.info('Pipette is Drawing')
        pp.draw()
        pp.disconnect()

class PipetteExpelWidget(QPushButton):
    """A push button widget to connect to the valve controller to actuate the expel function

    This is linked to the [hardware][picking_pipette] method
    """
    
    def __init__(self, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        self._mmc = CMMCorePlus.instance()
        self._create_button()

    def _create_button(self)->None:
        
        self.setText("Pipette Expel")
        self.clicked.connect(self._expel)

    def _expel(self)->None:
        # Initialize and connect to hardware controller
        cfg_dir = Path().absolute().parent / "python-fish-sorter/fish_sorter/configs/"
        array_file = Path().absolute().parent / "python-fish-sorter/fish_sorter/configs/arrays/6well_plate20240822.json"
        try:
            pp = PickingPipette(cfg_dir, self._mmc, array_file)
        except Exception as e:
            logging.critical("Could not initialize and connect hardware controller")

        logging.info('Pipette is Expelling')
        pp.expel()
        pp.disconnect()

class PipettePressureWidget(QPushButton):
    """A push button widget to connect to the valve controller to toggle the pressure

    This is linked to the [hardware][picking_pipette] method
    """
    
    def __init__(self, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        self._mmc = CMMCorePlus.instance()
        self._create_button()
        self.pressure_state = False

    def _create_button(self)->None:
        
        self.setText("Toggle Pressure Valve")
        self.clicked.connect(self._pressure)

    def _pressure(self)->None:
        # Initialize and connect to hardware controller
        cfg_dir = Path().absolute().parent / "python-fish-sorter/fish_sorter/configs/" 
        array_file = Path().absolute().parent / "python-fish-sorter/fish_sorter/configs/arrays/6well_plate20240822.json"
        try:
            pp = PickingPipette(cfg_dir, self._mmc, array_file)
        except Exception as e:
            logging.critical("Could not initialize and connect hardware controller")

        logging.info(f'Toggle Pressure Valve: {self.pressure_state}')
        self.pressure_state = not self.pressure_state
        pp.pressure(self.pressure_state)
        pp.disconnect()