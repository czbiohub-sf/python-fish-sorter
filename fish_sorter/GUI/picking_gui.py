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

class PickGUI(QWidget):

    def __init__(self, picker=None, parent: QWidget | None=None):
        """Initialize Picker GUI

        :param picker: Pick class object to control picking
        :type picker: class instance
        """
        
        super().__init__(parent=parent)
        CMMCorePlus.instance()

        self.pick = picker
        self.pick_calib = False
        self.disp_calib = False
        
        self.calib_pick = PipettePickCalibWidget(self)
        self.pick_calib_status = QLabel('❌ Pick Not Calibrated')
        calib_disp = PipetteDispCalibWidget(self)
        self.disp_calib_status = QLabel('❌ Disp Not Calibrated')
        move2pick = Pipette2PickWidget(self)
        move2disp = Pipette2DispWidget(self)
        move2clear = Pipette2ClearWidget(self)
        move2swing = Pipette2SwingWidget(self)
        img = ImageWidget(self)
        home = HomeWidget(self)
        move_pipette = MovePipette(self)
        self.pw = PickWidget(self)
        self.pw.setEnabled(False)
        self.new_expt = NewExptWidget(self)
        reset = ResetWidget(self)
        
        draw = PipetteDrawWidget(self)
        expel = PipetteExpelWidget(self)
        ppp = PipettePressureWidget(self)
        
        time = ChangeTimeWidget(self)
        self.single = SinglePickWidget(self)
        self.single.setEnabled(False)

        layout = QGridLayout(self)
        layout.addWidget(self.calib_pick, 1, 0)
        layout.addWidget(self.pick_calib_status, 1, 3)
        layout.addWidget(calib_disp, 1, 1)
        layout.addWidget(move2swing, 1, 2)
        layout.addWidget(move2pick, 2, 0)
        layout.addWidget(move2disp, 2, 1)
        layout.addWidget(move2clear, 2, 2)
        layout.addWidget(img, 3, 0)
        layout.addWidget(home, 3, 1)
        layout.addWidget(self.pick_calib_status, 4, 0)
        layout.addWidget(self.disp_calib_status, 4, 1)
        self._update_calib_status()
        
        layout.addWidget(move_pipette, 5, 0)
        layout.addWidget(time, 6, 0)
        layout.addWidget(draw, 7, 0)
        layout.addWidget(expel, 7, 1)
        layout.addWidget(ppp, 7, 2)
        layout.addWidget(self.single, 8, 0)
        layout.addWidget(self.pw, 9, 0)
        layout.addWidget(self.new_expt, 10, 0)
        layout.addWidget(reset, 10, 1)

    def _update_calib_status(self):
        """Update the GUI that the pick and or dispense heights
        are calibrated
        """
        
        if self.pick_calib:
            self.pick_calib_status.setText('✅ Pick Calibrated')
        else:
            self.pick_calib_status.setText('❌ Pick Not Calibrated')
        if self.disp_calib:
            self.disp_calib_status.setText('✅ Disp Calibrated')
        else:
            self.disp_calib_status.setText('❌ Disp Not Calibrated')

    def update_pick_widgets(self, status: bool=True):
        """Updates the widgets dependent on the Pick class
        """
        
        self.pw.setEnabled(status)
        self.single.setEnabled(status)

class PipettePickCalibWidget(QPushButton):
    """A push button widget to calibrate the pick position for the pipette
    """
    
    save_pick_h = pyqtSignal()

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
        self.picking._update_calib_status()
        self.save_pick_h.emit()


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
        self.picking._update_calib_status()     


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
        self.picking.pick.phc.move_pipette(pos='pick')


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
        self.picking.pick.phc.move_pipette(pos='dispense')


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
        
        self.picking.pick.phc.move_pipette(pos='clearance')


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
        
        self.picking.pick.phc.move_pipette(pos='pipette_swing')


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
        self.picking.pick.phc.move_pipette_increment(dist, unit_bool)

    def _move_pipette_down(self):

        dist = self.distance_spinbox.value()
        units = self.units_dropdown.currentText()
        unit_bool = units == 'mm'

        logging.info(f'Moving pipette by {dist} {units}')
        self.picking.pick.phc.move_pipette_increment(dist, unit_bool)


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
        self.picking.pick.phc.draw_time(time)

    def _change_expel(self):

        time = self.time_spinbox.value()
        logging.info(f'Change Expel time to {time} ms')
        self.picking.pick.phc.expel_time(time)


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
        
        self.picking.pick.phc.draw()


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
        
        self.picking.pick.phc.expel()


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
        self.picking.pick.phc.pressure(self.pressure_state)


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
            self.status_updatse.emit('Picking complete!')
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
            self.fp_thread = PickerThread(self.picking)
            self.fp_thread.status_update.connect(self._update_status)
            self.fp_thread.picking_done.connect(self._picking_finished)
            self.fp_thread.start()
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


class NewExptWidget(QPushButton):
    """A push button widget to start a new experiment
    """
    
    new_exp_req = pyqtSignal()

    def __init__(self, picking, parent: QWidget | None=None):
        
        super().__init__(parent=parent)

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )

        self.picking = picking
        self._mmc = CMMCorePlus.instance()

        self._create_button()

    def _create_button(self)->None:
        
        self.setText("New Experiment")
        self.clicked.connect(self._new)

    def _new(self):
        logging.info('Start new experiment')

        self.picking.update_pick_widgets(False)
        self.new_exp_req.emit()


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
        self.clicked.connect(self.picking.pick.phc.dest_home)  


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
        self.clicked.connect(self.picking.pick.phc.move_fluor_img)


class SinglePickThread(QThread):
    """Thread picking so that live preview stay on during a single pick
    """

    status_update = pyqtSignal(str)
    picking_done = pyqtSignal()

    def __init__(self, picking, parent = None):
        """Thread for single picking

        :param picking: picking gui class
        :type picking: pick gui class instance
        """

        super().__init__(parent=parent)
        self.picking = picking
    
    def run(self):
        """Run the single pick thread
        """

        try:
            self.status_update.emit('Start Single Pick Thread')
            self.picking.pick.single_pick(self.dtime)
            self.status_update.emit('Single Pick Complete!')
        except Exception as e:
            self.status_update.emit(f'Exception {str(e)}')
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
        delay_label = QLabel('Delay Time')
        layout.addWidget(delay_label, 1, 0)

        self.delay_time_spinbox = QDoubleSpinBox()
        self.delay_time_spinbox.setRange(0.00, 10.00)
        self.delay_time_spinbox.setSingleStep(0.05)
        self.delay_time_spinbox.setValue(1.00)
        layout.addWidget(self.delay_time_spinbox, 2, 0)
        unit_label = QLabel('s')
        layout.addWidget(unit_label, 2, 1)

        self.single_pick_btn = QPushButton('Single Pick')
        self.single_pick_btn.clicked.connect(self._start_pick)
        layout.addWidget(self.single_pick_btn, 3, 0)

    def _start_pick(self):
        """Runs the single pick thread
        """

        self._mmc.live_mode = True

        if self.picking.pick_calib and self.picking.disp_calib:
            self.sp_thread = SinglePickThread(picking = self.picking)
            self.sp_thread.dtime = self.delay_time_spinbox.value()
            logging.info(f'Delay time set to {self.sp_thread.dtime} s')
            self.sp_thread.status_update.connect(self._update_status)
            self.sp_thread.start()
        else:
            logging.info('Pipette not calibrated')

    def _update_status(self, msg):
        """Helper to update logging
        """

        logging.info(f'{msg}')