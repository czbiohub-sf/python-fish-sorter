import json
import logging
import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit, 
    QPushButton,
    QRadioButton, 
    QSizePolicy, 
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

class SetupWidget(QWidget):
    """A napari micromanager widget to input setup parameters,
    including the file path for image storage, the imaging plate config,
    and the pick type
    """

    def __init__(self, cfg_path, parent: QWidget | None=None):
        """
        Initialization for widgets to return user inputs

        :param cfg_path: parent path directory for all of the config files
        :type cfg_path: Path
        :param expt_parent_dir: parent directory to the experiment folder
        :type expt_parent_dir: Path
        """

        super().__init__(parent)
        self.layout = QGridLayout(self)
        self.layout.setSpacing(50)
        
        self.config = Path(cfg_path)

        self.img_array_label = QLabel("Select Imaging Plate Array:")
        self.img_array_dropdown = QComboBox()
        self.layout.addWidget(self.img_array_label, 0, 0)
        self.layout.addWidget(self.img_array_dropdown, 0, 1, 1, 2)

        self.dp_array_label = QLabel("Select Dispense Plate Array:")
        self.dp_array_dropdown = QComboBox()
        self.layout.addWidget(self.dp_array_label, 1, 0)
        self.layout.addWidget(self.dp_array_dropdown, 1, 1, 1, 2)
        
        self.refresh_list()

        self.pick_type = self.load_config("pick", "pick_type_config.json")
        self.pick_type_label = QLabel("Select Pick Type:")
        self.layout.addWidget(self.pick_type_label, 2, 0)
        self.pick_type_grp = QButtonGroup(self)
        self.radio_layout = QHBoxLayout()
        self.radio_layout.setSpacing(25)
        self.populate_options()
        self.layout.addLayout(self.radio_layout, 2, 1)

        self.pick_setup = QPushButton("Setup Picker")
        self.layout.addWidget(self.pick_setup, 3, 0, 1, 2)
        self.layout.setRowStretch(self.layout.rowCount(), 1)
        self.layout.setColumnStretch(self.layout.columnCount(), 1)

    def load_config(self, cfg_folder, cfg_file):
        """
        Loads the config file

        :param cfg_folder: desired config folder to load
        :type cfg_folder: str
        :param cfg_file: desired config file to load
        :type cfg_file: str

        :returns: loaded json file
        :rtype: dict
        """

        cfg_path = self.config / cfg_folder / cfg_file

        try:
            with open(cfg_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            logging.critical("Config file not found")
            return {}

    def refresh_list(self):
        """
        Refreshes the list of array types to select
        """

        config_path = self.config / "arrays"
        if os.path.exists(config_path):
            array_files = [f for f in os.listdir(config_path) if f.endswith('.json')]
            self.img_array_dropdown.clear()
            self.img_array_dropdown.addItems(array_files)

            self.dp_array_dropdown.clear()
            self.dp_array_dropdown.addItems(array_files)
        else:
            self.img_array_dropdown.addItems('Config Files Not Found')
            self.dp_array_dropdown.addItems('Config Files Not Found')

    def populate_options(self):
        """
        Populates the pick type options from the config file
        """

        for key in self.pick_type.keys():
            radio_button = QRadioButton(key)
            self.pick_type_grp.addButton(radio_button)
            self.radio_layout.addWidget(radio_button)

    def get_pick_type(self):
        """
        Returns user input of the selected option for the pick type

        :returns: pick type selection, offset
        :rtype: str, np array
        """

        for button in self.pick_type_grp.buttons():
            if button.isChecked():
                offset = np.array([self.pick_type[button.text()]['picker']['length_offset'], 
                        self.pick_type[button.text()]['picker']['width_offset']])
                dtime = self.pick_type[button.text()]['picker']['dtime']
                pick_height = self.pick_type[button.text()]['picker']['pick_height']
                return button.text(), offset, dtime, pick_height
        return "default_pick_type", np.array([0.0, 0.0]), float(0.0), float(0.0)
    
    def get_img_array(self):
        """
        Returns the user input selected image plate array 

        :returns: image plate array in use
        :rtype: str
        """

        return self.img_array_dropdown.currentText()

    def get_dp_array(self):
        """
        Returns the user input selected dispense plate array 

        :returns: dispense plate array in use
        :rtype: str
        """

        return self.dp_array_dropdown.currentText()