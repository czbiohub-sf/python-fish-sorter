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

from fish_sorter.GUI.picking import Pick

COLOR_TYPES = Union[
    QColor,
    int,
    str,
    Qt.GlobalColor,
    "tuple[int, int, int, int]",
    "tuple[int, int, int]"
]

class Picking(QWidget):

    def __init__(self, picker, parent: QWidget | None=None):
        """Initialize Picker GUI

        :param picker: Pick class object to control picker hardware
        :type picker: class instance
        """
        
        super().__init__(parent=parent)
        CMMCorePlus.instance()

        self.calibrated = False
        self.pick = picker

        calib = PipetteCalibrateWidget()
        picking = PickWidget()
        disconnect = DisconnectWidget()
        
        #TODO do these still need to be here
        #Could they live in a service hardware picker GUI 
        ddd = PipetteDrawWidget()
        vvv = PipetteExpelWidget()
        ppp = PipettePressureWidget()
        
        
        layout = QGridLayout(self)
        layout.addWidget(calib, 1, 0)
        layout.addWidget(ddd, 2, 0)
        layout.addWidget(vvv, 2, 1)
        layout.addWidget(ppp, 2, 2)
        layout.addWidget(picking, 3, 0)
        layout.addWidget(disconnect, 3, 1)

     
 class PipetteCalibrateWidget(QPushButton):
    """A push button widget to calibrate the pipette
    """
    
    def __init__(self, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        self._mmc = CMMCorePlus.instance()
        self._create_button()

    def _create_button(self)->None:
        
        self.setText("Calibrate Pipette")
        self.clicked.connect(self._calibrate)

    def _calibrate(self)->None:

        #TODO separate into separate buttons
        #Click should be after moving to the position
        #TODO allow option for different destination plate well
        
        logging.info('Calibrate pick height into array')
        self.pick.check_calib(self.calibrate)
        self.pick.set_calib(pick=True)

        logging.info('Calibrate dispense height into destination plate')
        
        self.pick.check_calib(self.calibrate, pick=False, well='A1')
        self.pick.set_calib(pick=False)

        logging.info('Pipette draw and expel locations set')
        self.calibrate = True       

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
        
        self.pick.draw()

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
        
        self.pick.expel()
        
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
        
        self.pressure_state = not self.pressure_state
        self.pick.pressure(self.pressure_state)

 class PickWidget(QPushButton):
    """A push button widget to start picking

    """
    
    def __init__(self, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        self._mmc = CMMCorePlus.instance()
        self._create_button()

    def _create_button(self)->None:
        
        self.setText("Pick!")
        self.clicked.connect(self._start_picking) 
    
    def _start_picking(self):
        
        if self.calibrate:
            logging.info('Opening classifications')
            self.pick.get_classified()
            logging.info('Matching to pick parameters')
            self.pick.match_pick()
            logging.info('Start of picking')
            self.pick.pick_me()
        else:
            logging.info('Pipette not calibrated')

 class DisconnectWidget(QPushButton):
    """A push button widget to disconnect hardware

    """
    
    def __init__(self, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        self._mmc = CMMCorePlus.instance()
        self._create_button()

    def _create_button(self)->None:
        
        self.setText("Disconnect Hardware")
        self.clicked.connect(self.pick.disconnect_hardware()) 