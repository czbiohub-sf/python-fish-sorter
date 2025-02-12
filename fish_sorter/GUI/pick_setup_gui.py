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

COLOR_TYPES = Union[
    QColor,
    int,
    str,
    Qt.GlobalColor,
    "tuple[int, int, int, int]",
    "tuple[int, int, int]"
]

class PickSetup(QWidget):

    def __init__(self, parent: QWidget | None=None):
        """Initialize Picker GUI

        """
        
        super().__init__(parent=parent)
        CMMCorePlus.instance()

        self.setup = QPushButton("Setup Picker")

        layout = QGridLayout(self)
        layout.addWidget(self.setup, 1, 0)