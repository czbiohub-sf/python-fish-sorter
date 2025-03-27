import logging
import sys
from json import load
from pathlib import Path
from time import sleep
from typing import List, Optional, Union

from pymmcore_plus import CMMCorePlus
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QComboBox, 
    QGridLayout, 
    QLabel, 
    QPushButton, 
    QSizePolicy, 
    QDoubleSpinBox, 
    QVBoxLayout, 
    QWidget
)

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

        self.pick = picker
        self.pick_calib = False
        self.disp_calib = False
        
        calib_pick = PipettePickCalibWidget(self)
        calib_disp = PipetteDispCalibWidget(self)
        move2pick = Pipette2PickWidget(self)
        move2disp = Pipette2DispWidget(self)
        move2clear = Pipette2ClearWidget(self)
        img = ImageWidget(self)
        home = HomeWidget(self)
        move_pipette = MovePipette(self)
        pw = PickWidget(self)
        disconnect = DisconnectWidget(self)
        
        draw = PipetteDrawWidget(self)
        expel = PipetteExpelWidget(self)
        ppp = PipettePressureWidget(self)
        
        
        layout = QGridLayout(self)
        layout.addWidget(calib_pick, 1, 0)
        layout.addWidget(calib_disp, 1, 1)
        layout.addWidget(move2pick, 2, 0)
        layout.addWidget(move2disp, 2, 1)
        layout.addWidget(move2clear, 2, 2)
        layout.addWidget(move_pipette, 3, 0)
        layout.addWidget(img, 4, 0)
        layout.addWidget(home, 4, 1)
        layout.addWidget(draw, 5, 0)
        layout.addWidget(expel, 5, 1)
        layout.addWidget(ppp, 5, 2)
        layout.addWidget(pw, 6, 0)
        layout.addWidget(disconnect, 6, 1)
     
class PipettePickCalibWidget(QPushButton):
    """A push button widget to calibrate the pick position for the pipette
    """
    
    def __init__(self, picking, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        self.picking = picking
        self._mmc = CMMCorePlus.instance()
        self._create_button()

    def _create_button(self)->None:
        
        self.setText("Calibrate Pick Position")
        self.clicked.connect(self._pick_calib)

    def _pick_calib(self)->None:
        
        logging.info('Calibrate pick height into array')
        self.picking.pick.set_calib(pick=True)
        self.picking.pick_calib = True

class PipetteDispCalibWidget(QPushButton):
    """A push button widget to calibrate the dispense position for the pipette
    """
    
    def __init__(self, picking, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        self.picking = picking
        self._mmc = CMMCorePlus.instance()
        self._create_button()

    def _create_button(self)->None:
        
        self.setText("Calibrate Dispense Position")
        self.clicked.connect(self._disp_calib)

    def _disp_calib(self)->None:

        logging.info('Calibrate dispense height into destination plate')
        self.picking.pick.set_calib(pick=False)
        self.picking.disp_calib = True              

class Pipette2PickWidget(QPushButton):
    """A push button widget to connect to the pipette widget move the pipette to the pick position 

    This is linked to the [hardware][picking_pipette] method
    """
    
    def __init__(self, picking, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        self.picking = picking
        self._mmc = CMMCorePlus.instance()
        self._create_button()

    def _create_button(self)->None:
        
        self.setText("Move to Pick Position")
        self.clicked.connect(self._pick_pos)

    def _pick_pos(self)->None:
        
        self.picking.pick.move_calib(pick=True)
        self.picking.pick.pp.move_pipette(pos='pick')

class Pipette2DispWidget(QPushButton):
    """A push button widget to connect to the pipette widget move the pipette to the dispense position 

    This is linked to the [hardware][picking_pipette] method
    """
    
    def __init__(self, picking, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        self.picking = picking
        self._mmc = CMMCorePlus.instance()
        self._create_button()

    def _create_button(self)->None:
        
        self.setText("Move to Dispense Position")
        self.clicked.connect(self._disp_pos)

    def _disp_pos(self)->None:
        self.picking.pick.move_calib(pick=False, well='A01')
        self.picking.pick.pp.move_pipette(pos='dispense')

class Pipette2ClearWidget(QPushButton):
    """A push button widget to connect to the pipette widget move the pipette to the clearance position 

    This is linked to the [hardware][picking_pipette] method
    """
    
    def __init__(self, picking, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        self.picking = picking
        self._mmc = CMMCorePlus.instance()
        self._create_button()

    def _create_button(self)->None:
        
        self.setText("Move to Clearance Position")
        self.clicked.connect(self._clear_pos)

    def _clear_pos(self)->None:
        
        self.picking.pick.pp.move_pipette(pos='clearance')

class MovePipette(QWidget):
    """A widget to move the pipette a user-defined distance"""

    def __init__(self, picking, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.picking = picking
        self._mmc = CMMCorePlus.instance()
        self._create_gui()

    def _create_gui(self):

        layout = QGridLayout(self)
        label = QLabel('Move Pipette')
        layout.addWidget(label, 0, 0)

        self.distance_spinbox = QDoubleSpinBox()
        self.distance_spinbox.setRange(-1000.00, 1000.00)
        self.distance_spinbox.setSingleStep(0.05)
        self.distance_spinbox.setDecimals(2)
        self.distance_spinbox.setSuffix(" ")
        layout.addWidget(self.distance_spinbox, 1, 0)

        self.units_dropdown = QComboBox()
        self.units_dropdown.addItems(['mm', 'um'])
        layout.addWidget(self.units_dropdown, 1, 1)

        self.move_button = QPushButton('Move Pipette')
        self.move_button.clicked.connect(self._move_pipette)
        layout.addWidget(self.move_button, 1, 2)

    def _move_pipette(self):

        dist = self.distance_spinbox.value()
        units = self.units_dropdown.currentText()
        unit_bool = units == 'mm'

        logging.info(f'Moving pipette by {dist} {units}')
        self.picking.pick.pp.move_pipette_increment(dist, unit_bool)
        
class PipetteDrawWidget(QPushButton):
    """A push button widget to connect to the valve controller to actuate the draw function

    This is linked to the [hardware][picking_pipette] method
    """
    
    def __init__(self, picking, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        self.picking = picking
        self._mmc = CMMCorePlus.instance()
        self._create_button()

    def _create_button(self)->None:
        
        self.setText("Pipette Draw")
        self.clicked.connect(self._draw)

    def _draw(self)->None:
        
        self.picking.pick.pp.draw()

class PipetteExpelWidget(QPushButton):
    """A push button widget to connect to the valve controller to actuate the expel function

    This is linked to the [hardware][picking_pipette] method
    """
    
    def __init__(self, picking, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        self.picking = picking
        self._mmc = CMMCorePlus.instance()
        self._create_button()

    def _create_button(self)->None:
        
        self.setText("Pipette Expel")
        self.clicked.connect(self._expel)

    def _expel(self)->None:
        
        self.picking.pick.pp.expel()
        
class PipettePressureWidget(QPushButton):
    """A push button widget to connect to the valve controller to toggle the pressure

    This is linked to the [hardware][picking_pipette] method
    """
    
    def __init__(self, picking, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        self.picking = picking
        self._mmc = CMMCorePlus.instance()
        self._create_button()
        self.pressure_state = False

    def _create_button(self)->None:
        
        self.setText("Toggle Pressure Valve")
        self.clicked.connect(self._pressure)

    def _pressure(self)->None:
        
        self.pressure_state = not self.pressure_state
        self.picking.pick.pp.pressure(self.pressure_state)

class PickWidget(QPushButton):
    """A push button widget to start picking
    """
    
    def __init__(self, picking, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        self.picking = picking
        self._mmc = CMMCorePlus.instance()
        self._create_button()

    def _create_button(self)->None:
        
        self.setText("Pick!")
        self.clicked.connect(self._start_picking) 
    
    def _start_picking(self):
        
        if self.picking.pick_calib and self.picking.disp_calib:
            logging.info('Opening classifications')
            self.picking.pick.get_classified()
            logging.info('Matching to pick parameters')
            self.picking.pick.match_pick()
            logging.info('Start of picking')
            self.picking.pick.pick_me()
        else:
            logging.info('Pipette not calibrated')

class DisconnectWidget(QPushButton):
    """A push button widget to disconnect hardware
    """
    
    def __init__(self, picking, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        self.picking = picking
        self._mmc = CMMCorePlus.instance()
        self._create_button()

    def _create_button(self)->None:
        
        self.setText("Disconnect Hardware")
        self.clicked.connect(self.picking.pick.disconnect_hardware) 

class HomeWidget(QPushButton):
    """A push button widget to move the dispense stages to the home position
    """
    
    def __init__(self, picking, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        self.picking = picking
        self._mmc = CMMCorePlus.instance()
        self._create_button()

    def _create_button(self)->None:
        
        self.setText("Move Dispense Stages to Home")
        self.clicked.connect(self.picking.pick.pp.dest_home)  

class ImageWidget(QPushButton):
    """A push button widget to move the stages for fluorescence imaging
    """
    
    def __init__(self, picking, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        self.picking = picking
        self._mmc = CMMCorePlus.instance()
        self._create_button()

    def _create_button(self)->None:
        
        self.setText("Move Stages to Image")
        self.clicked.connect(self.picking.pick.pp.move_fluor_img)  