import sys
from json import load
from pathlib import Path
from time import sleep
from typing import List, Optional, Union

from pymmcore_plus import CMMCorePlus
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QPushButton, QSizePolicy, QWidget, QGridLayout


class MosaicWidget(QWidget):

    def __init__(self, parent: QWidget | None=None):
        
        super().__init__(parent=parent)
        CMMCorePlus.instance()

        self.btn = QPushButton("Stitch mosaic")
        self.dummy =  QPushButton("do something")
        self.calibrate = QPushButton("Calibrate")
        self.pos = QPushButton("Position")


        layout = QGridLayout(self)
        layout.addWidget(self.btn, 1, 0)
        layout.addWidget(self.dummy, 1, 1)
        layout.addWidget(self.calibrate, 2, 0)
        layout.addWidget(self.pos, 3, 0)