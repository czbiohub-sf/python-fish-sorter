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

    def __init__(self, pick_type_cfg_path, parent=None):
        """
        :param pick_type_cfg_path: path directory for the pick type config file
        :type pick_type_cfg_path: str
        """

        super().__init__(parent)

        self.layout = QVBoxLayout(self)

        self.expt_path_label = QLabel("Selected Path: None")
        self.expt_path_button = QPushButton("Select Mosaic Image Filepath")
        self.expt_path_button.clicked.connect(self.select_expt_path)
        self.layout.addWidget(self.expt_path_label)
        self.layout.addWidget(self.expt_path_button)

        self.prefix_label = QLabel("Experiment Prefix:")
        self.prefix_input - QLineEdit()
        self.prefix_input.setPlaceholderText("Enter prefix for experiment")
        self.layout.addWidget(self.prefix_label)
        self.layout.addWidget(self.prefix_input)

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
        Loads the pick type config file
        """

        #TODO should this be handled differently loading all of the configs?
        #This does require user input to decide which part of the config to use

        try:
            with open(self.pick_type_cfg_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            logging.critical("Config file not found")
            return {}

    def select_expt_path(self):
        """
        Selects the filepath directory for the experiment
        """
        #TODO should this be in the higher level init and passed in?
        preset_dir = "fish_picker/expt/parent_dir"

        if os.path.exists(preset_dir):
            default_path = preset_dir
        else:
            default_path = os.path.expanduser("~")        

        filepath = QFileDialog.getExistingDirectory(self, "Select Directory", default_path)
        if filepath:
            self.expt_path_label.setText(f"Selected Path: {filepath}")
            logging.info(f"Selected Path:" {filepath})

    def refresh_list(self):
        """
        Refreshes the list of array types to select
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
        Populates the pick type options from the config file
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
        Returns user input of the selected option for the pick type

        :returns: pick type selection
        :rtype: str
        """

        for button in self.pick_type_grp.buttons():
            if button.isChecked():
                return button.text()
        return None

    def get_expt_path(self):
        """
        Returns the user input for the experiment folder path to
        save the experiment data

        :returns: experiment folder path
        :rtype: str
        """

        return self.expt_path_label.text().replace("Selected Path:", "")

    def get_expt_prefix(self):
        """
        Returns the user input of the experiment prefix to use when saving
        experiment data

        :returns: experiment prefix
        :rtype: str
        """

        return self.prefix_input.text()
    
    def get_array(self):
        """
        Returns the user input selected image plate array 

        :returns: image plate array in use
        :rtype: str
        """

        return self.array_dropdown.currentText()