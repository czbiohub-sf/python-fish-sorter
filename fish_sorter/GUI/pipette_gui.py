import sys
from json import load
from pathlib import Path
from time import sleep
from typing import List, Optional, Union

from pymmcore_plus import CMMCorePlus
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QPushButton, QSizePolicy, QWidget, QGridLayout

from fish_sorter.hardware.zaber_controller import ZaberController

COLOR_TYPES = Union[
    QColor,
    int,
    str,
    Qt.GlobalColor,
    "tuple[int, int, int, int]",
    "tuple[int, int, int]"
]

# TODO Swap zaber config with picker_defaults_config for several parameters

class PipetteWidget(QWidget):

    def __init__(self, parent: QWidget | None=None):
        
        super().__init__(parent=parent)
        CMMCorePlus.instance()

        xxx = ZaberInitWidget()
        xyz = ZaberHomeWidget()
        zzz = ZaberTestWidget()

        layout = QGridLayout(self)
        layout.addWidget(xxx, 1, 0)
        layout.addWidget(xyz, 2, 0)
        layout.addWidget(zzz, 3, 0)


class ZaberInitWidget(QPushButton):
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
        
        self.setText("Initialize Zaber")
        self.clicked.connect(self._zaber_init)

    def _zaber_init(self)->None:

        cfg_dir = Path().absolute().parent / "fish_sorter/configs/hardware"
        cfg_file = "zaber_config.json"
        cfg_path = cfg_dir / cfg_file
        # Initialize and connect to hardware controller
        try:
            with open(cfg_path, 'r') as f:
                p = load(f)
            zaber_config = p['zaber_config']
            print(zaber_config)
            zc = ZaberController(zaber_config, env='prod')
        except Exception as e:
            print("Could not initialize and connect hardware controller")
    
        # Test moving the pipette, x, and y stages to max position
        stages = ['x', 'y', 'p']

        print('Move stages to max and back home')
        for stage in stages:
            zc.move_arm(stage, zaber_config['max_position'][stage])
            sleep(2)
            zc.move_arm(stage, zaber_config['home'][stage])
            sleep(2)
        
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

        cfg_dir = Path().absolute().parent / "fish_sorter/configs/hardware"
        cfg_file = "zaber_config.json"
        cfg_path = cfg_dir / cfg_file
        # Initialize and connect to hardware controller
        try:
            with open(cfg_path, 'r') as f:
                p = load(f)
            zaber_config = p['zaber_config']
            zc = ZaberController(zaber_config, env='prod')
        except Exception as e:
            print("Could not initialize and connect hardware controller")
    
        # Test moving the pipette, x, and y stages to max position
        stages = ['x', 'y', 'p']

        print('Move stages to max and back home')
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

        cfg_dir = Path().absolute().parent / "fish_sorter/configs/hardware"
        cfg_file = "zaber_config.json"
        cfg_path = cfg_dir / cfg_file
        # Initialize and connect to hardware controller
        try:
            with open(cfg_path, 'r') as f:
                p = load(f)
            zaber_config = p['zaber_config']
            zc = ZaberController(zaber_config, env='prod')
        except Exception as e:
            print("Could not initialize and connect hardware controller")
    
        # Test moving the pipette, x, and y stages to max position
        stages = ['x', 'y', 'p']

        print('Move stages to max and back home')
        for stage in stages:
            zc.move_arm(stage, zaber_config['max_position'][stage])
            sleep(2)
            zc.move_arm(stage, zaber_config['home'][stage])
            sleep(2)
        
        print('Move pipette to set locations')
        print('Swing height')
        zc.move_arm('p', zaber_config['pipette']['swing']['p'])
        sleep(2)
        zc.move_arm('p', zaber_config['home']['p'])
        sleep(2)
        
        print('Pick height')
        zc.move_arm('p', zaber_config['pipette']['pick']['p'])
        sleep(2)
        zc.move_arm('p', zaber_config['home']['p'])
        sleep(2)

        print('Clearance height')
        zc.move_arm('p', zaber_config['pipette']['clearance']['p'])
        sleep(2)
        zc.move_arm('p', zaber_config['home']['p'])
        sleep(2)

        print('Dispense height')
        zc.move_arm('p', zaber_config['pipette']['dispense']['p'])
        sleep(2)
        zc.move_arm('p', zaber_config['home']['p'])
        sleep(2)

        print('Test complete')

        zc.disconnect()