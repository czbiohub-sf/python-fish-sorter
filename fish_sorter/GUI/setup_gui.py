import json
import logging
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

    def __init__(self, pick_type_cfg_path, parent=None):
        """
        """

        super().__init__(parent)

        self.layout = QVBoxLayout(self)

        self.img_path_label = QLabel("selected Path: None")
        self.img_path_button = QPushButton("Select Mosaic Image Filepath")
        self.img_path_button.clicked.connect(self.select_img_path)
        self.layout.addWidget(self.img_path_label)
        self.layout.addWidget(self.img_path_button)

        self.array_label = QLabel("Select Imaging Plate Array:")
        self.array_dropdown = QComboBox()
        self.refresh_list()
        self.layout.addWidget(self.array_label)
        self.layout.addWidget(self.array_dropdown)

        self.pick_type_cfg_path = pick_type_cfg_pth
        self.config_data = self.load_config
        self.pick_type_label = QLabel("Select Pick Type:")
        self.pick_type_grp = QButtonGroup(self)
        self.populate_options()
        self.layout.addWidget(self.pick_type_label)

    def load_config(self):
        """
        """

        try:
            with open(self.pick_type_cfg_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            logging.critical("Config file not found")
            return {}

    def select_img_path(self):
        """
        """

        #TODO have it preset to a fish sorter default directory
        filepath = QFileDialog.getExistingDirectory(self, "Select Directory")
        if filepath:
            self.img_path_label.setText(f"Selected Path: {filepath}")
            logging.info(f"Selected Path:" {filepath})

    def refresh_list(self):
        """
        """

        #TODO this should be handled better from the setup 
        config_path = Path().absolute().parent / "configs/arrays"
        if os.path.exists(config_path):
            array_files = [f for f in os.listdir(config_path) if f.endswith('.json')]
            self.array_dropdown.clear()
            self.array_dropdown.addItems(array_files)
        else:
            self.array_dropdown.addItems('Config Files Not Found')

    def populate_options(self):
        """
        """

        if not self.config_data:
            no_cfg_label = QLabel("Config Data Not Found")
            self.layout.addWidget(no_cfg_label)

        for key in self.config_data.keys():
            radio_button = QRadioButton(key)
            self.pick_type_grp.addButton(radio_button)
            self.layout.addWidget(radio_button)

    def get_selected_option(self):
        """
        """

        for button in self.pick_type_grp.buttons():
            if button.isChecked():
                return button.text()
        return None

    def get_img_path(self):
        """
        """

        return self.img_path_label.text().replace("Selected Path:", "")

    def get_array(self):
        """
        """

        return self.array_dropdown.currentText()