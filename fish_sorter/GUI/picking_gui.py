import logging
import sys
from json import load
from pathlib import Path
from time import sleep
from typing import List, Optional, Union

from pymmcore_plus import CMMCorePlus
from qtpy.QtCore import (
    QSize,
    Qt
)
from PyQt6.QtCore import (
    pyqtSignal,
    QThread
)
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QComboBox, 
    QGridLayout,
    QHBoxLayout, 
    QLabel, 
    QPushButton, 
    QSizePolicy, 
    QDoubleSpinBox, 
    QSpinBox,
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
        move2swing = Pipette2SwingWidget(self)
        img = ImageWidget(self)
        home = HomeWidget(self)
        move_pipette = MovePipette(self)
        pw = PickWidget(self)
        disconnect = DisconnectWidget(self)
        reset = ResetWidget(self)
        
        draw = PipetteDrawWidget(self)
        expel = PipetteExpelWidget(self)
        ppp = PipettePressureWidget(self)
        
        time = ChangeTimeWidget(self)
        single = SinglePickWidget(self)

        layout = QGridLayout(self)
        layout.addWidget(calib_pick, 1, 0)
        layout.addWidget(calib_disp, 1, 1)
        layout.addWidget(move2swing, 1, 2)
        layout.addWidget(move2pick, 2, 0)
        layout.addWidget(move2disp, 2, 1)
        layout.addWidget(move2clear, 2, 2)
        layout.addWidget(move_pipette, 3, 0)
        layout.addWidget(time, 4, 0)
        layout.addWidget(img, 5, 0)
        layout.addWidget(home, 5, 1)
        layout.addWidget(draw, 6, 0)
        layout.addWidget(expel, 6, 1)
        layout.addWidget(ppp, 6, 2)
        layout.addWidget(single, 7, 0)
        layout.addWidget(pw, 8, 0)
        layout.addWidget(disconnect, 8, 0)
        layout.addWidget(reset, 8, 2)


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


class Pipette2SwingWidget(QPushButton):
    """A push button widget to connect to the pipette widget move the pipette to the swing position 

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
        
        self.setText("Move to Swing Position")
        self.clicked.connect(self._swing_pos)

    def _swing_pos(self)->None:
        
        self.picking.pick.pp.move_pipette(pos='pipette_swing')


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
        self.distance_spinbox.setRange(0.00, 1000.00)
        self.distance_spinbox.setSingleStep(0.05)
        self.distance_spinbox.setDecimals(2)
        self.distance_spinbox.setSuffix(" ")
        layout.addWidget(self.distance_spinbox, 1, 0)

        self.units_dropdown = QComboBox()
        self.units_dropdown.addItems(['mm', 'um'])
        layout.addWidget(self.units_dropdown, 1, 1)

        self.move_up_button = QPushButton('Pipette Up')
        self.move_up_button.clicked.connect(self._move_pipette_up)
        layout.addWidget(self.move_up_button, 1, 2)

        self.move_down_button = QPushButton('Pipette Down')
        self.move_down_button.clicked.connect(self._move_pipette_down)
        layout.addWidget(self.move_down_button, 1, 3)

    def _move_pipette_up(self):

        dist = -self.distance_spinbox.value()
        units = self.units_dropdown.currentText()
        unit_bool = units == 'mm'

        logging.info(f'Moving pipette by {dist} {units}')
        self.picking.pick.pp.move_pipette_increment(dist, unit_bool)

    def _move_pipette_down(self):

        dist = self.distance_spinbox.value()
        units = self.units_dropdown.currentText()
        unit_bool = units == 'mm'

        logging.info(f'Moving pipette by {dist} {units}')
        self.picking.pick.pp.move_pipette_increment(dist, unit_bool)


class ChangeTimeWidget(QWidget):
    """A widget to change the draw and expel times"""

    def __init__(self, picking, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.picking = picking
        self._mmc = CMMCorePlus.instance()
        self._create_gui()

    def _create_gui(self):

        layout = QGridLayout(self)
        label = QLabel('Change Time')
        layout.addWidget(label, 0, 0)

        self.time_spinbox = QSpinBox()
        self.time_spinbox.setRange(0, 1000)
        layout.addWidget(self.time_spinbox, 1, 0)
        unit_label = QLabel('ms')
        layout.addWidget(unit_label, 1, 1)

        self.change_draw_button = QPushButton('Change Draw Time')
        self.change_draw_button.clicked.connect(self._change_draw)
        layout.addWidget(self.change_draw_button, 1, 2)

        self.change_expel_button = QPushButton('Change Expel Time')
        self.change_expel_button.clicked.connect(self._change_expel)
        layout.addWidget(self.change_expel_button, 1, 3)

    def _change_draw(self):

        time = self.time_spinbox.value()
        logging.info(f'Change Draw time to {time} ms')
        self.picking.pick.pp.draw_time(time)

    def _change_expel(self):

        time = self.time_spinbox.value()
        logging.info(f'Change Expel time to {time} ms')
        self.picking.pick.pp.expel_time(time)


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


class PickerThread(QThread):
    """Thread picking so that live preview stay on during full picking
    """

    status_update = pyqtSignal(str)
    picking_done = pyqtSignal()

    def __init__(self, picking, parent = None):
        super().__init__(parent=parent)
        self.picking = picking
    
    def run(self):

        try:
            self.status_update.emit('Start Picking Thread')
            self.picking.pick.get_classified()
            self.status_update.emit('Matching to pick parameters')
            self.picking.pick.match_pick()
            self.status_update.emit('Start of picking')
            self.picking.pick.pick_me()
            self.status_update.emit('Picking complete!')
        except Exception as e:
            self.status_update.emit(f'Exepction {str(e)}')
        finally:
            self.picking_done.emit()


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
        
        self.setText("Full Pick!")
        self.clicked.connect(self._start_full_picking) 
    
    def _start_full_picking(self):

        self._mmc.live_mode = True
        
        if self.picking.pick_calib and self.picking.disp_calib:
            self.thread = PickerThread(self.picking)
            self.thread.status_update.connect(self._update_status)
            self.thread.picking_done.connect(self._picking_finished)
            self.thread.start()
        else:
            logging.info('Pipette not calibrated')

    def _update_status(self, msg):
        """Helper to update logging
        """

        logging.info(f'{msg}')

    def _picking_finished(self):
        """Helper to update logging on thread
        """

        logging.info('Picker thread finished')


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


class ResetWidget(QPushButton):
    """A push button widget to reset the hardware connection
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
        
        self.setText("Reset Hardware")
        self.clicked.connect(self.picking.pick.reset_hardware) 


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

class SinglePickThread(QThread):
    """Thread picking so that live preview stay on during a single pick
    """

    status_update = pyqtSignal(str)
    picking_done = pyqtSignal()

    def __init__(self, picking, parent = None):
        super().__init__(parent=parent)
        self.picking = picking
    
    def run(self, dtime):
        """Run the single pick thread

        :param dtime: delay time passed by user input
        :type dtime: float
        """

        try:
            self.status_update.emit('Start Single Pick Thread')
            self.picking.single_pick(dtime)
            self.status_update.emit('Single Pick Complete!')
        except Exception as e:
            self.status_update.emit(f'Exepction {str(e)}')
        finally:
            self.status_update.emit('Finished single pick')


class SinglePickWidget(QWidget):
    """A widget to pick once with the automation at the current stage
    position
    """

    def __init__(self, picking, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.picking = picking
        self._mmc = CMMCorePlus.instance()
        self._create_gui()

    def _create_gui(self):

        layout = QGridLayout(self)
        label = QLabel('Single Pick')
        layout.addWidget(label, 0, 0)

        self.delay_time_spinbox = QDoubleSpinBox()
        self.delay_time_spinbox.setRange(0.00, 10..)
        self.delay_time_spinbox.setSingleStep(0.05)
        layout.addWidget(self.delay_time_spinbox, 1, 0)
        unit_label = QLabel('s')
        layout.addWidget(unit_label, 1, 1)

        self.single_pick_button = QPushButton('Single Pick')
        self.single_pick_button.clicked.connect(self._start_pick)
        layout.addWidget(self.single_pick_button, 2, 0)

    def _start_pick(self):
        """Runs the single pick thread
        """

        self._mmc.live_mode = True

        time = self.delay_time_spinbox.value()
        logging.info(f'Delay time set to {time} s')

        if self.picking.pick_calib and self.picking.disp_calib:
            self.thread = SinglePickThread(self.picking)
            self.thread.status_update.connect(self._update_status)
            self.thread.start(dtime=time)
        else:
            logging.info('Pipette not calibrated')     

    def _update_status(self, msg):
        """Helper to update logging
        """

        logging.info(f'{msg}')