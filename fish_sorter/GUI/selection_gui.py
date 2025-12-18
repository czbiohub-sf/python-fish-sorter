import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import List, Optional, Union

from pymmcore_plus import CMMCorePlus
from qtpy.QtCore import (
    QSize,
    Qt
)
from PyQt6.QtCore import QThread
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox, 
    QHBoxLayout, 
    QLabel,
    QMessageBox,
    QPushButton, 
    QSizePolicy, 
    QScrollArea, 
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

log = logging.getLogger(__name__)

class SelectGUI(QWidget):

    def __init__(self, picker=None, pick_type=None, parent: QWidget | None=None):
        """Initialize Selection GUI

        :param picker: Pick class object to reference pick paramter information
        :type picker: class instance
        :param pick_type: user-input pick type from pick type config options
        :type pick_type: str
        """
        
        super().__init__(parent=parent)
        CMMCorePlus.instance()

        self.pick = picker
        self._setup(pick_type)

        self.rows = []
        self.hide = True
        self.layout = QVBoxLayout(self)
        self.hide_cb = QCheckBox('Select Singlets Only')
        self.hide_cb.setChecked(True)
        self.hide_cb.stateChanged.connect(self.toggle_hidden)
        self.layout.addWidget(self.hide_cb)
        self.rows_layout = QVBoxLayout()
        self.rows_container = QWidget()
        self.rows_container.setLayout(self.rows_layout)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.rows_container)
        self.layout.addWidget(self.scroll)

        self.add_row_btn = QPushButton('Add Row')
        self.add_row_btn.clicked.connect(self.add_row)

        self.save_btn = QPushButton('Save')
        self.save_btn.clicked.connect(self.save_select)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.add_row_btn)
        btn_layout.addWidget(self.save_btn)
        self.layout.addLayout(btn_layout)

        self.add_row()

    def _setup(self, pick_type):
        """Setup the features

        :param pick_type: user-input pick type from pick type config options
        :type pick_type: str
        """

        self.well = self.pick.phc.dplate.wells['names']

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pickable_file_name = f"{timestamp}_{self.pick.prefix}_pickable.csv"
        self.pickable_path = os.path.normpath(os.path.join(self.pick.pick_dir, pickable_file_name))

        feat_dir = self.pick.cfg / "pick"
        feat_data = {}
        for filename in os.listdir(feat_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(feat_dir, filename)
                try:
                    with open(file_path, 'r') as file:
                       data = json.load(file)
                       feat_data = data[pick_type]
                       logging.info('Loaded {} config file'.format(filename))
                except FileNotFoundError:
                    logging.critical("Config file not found")

        self.deselect = list(
            feat for feat in list(feat_data['well_class']['deselect'].keys())
            if feat != 'description'
        )
        exclude = {'lHead', 'deselect'}
        self.features = [
            feat for feat in list(feat_data['well_class'].keys()) + list(feat_data['feature_class'].keys())
            if feat not in exclude
            ]

    def add_row(self):
        """Adds a row to the selection GUI
        """

        row = AddRow(self.well, self.features, self.deselect, self.hide, on_delete=self.delete_row)
        self.rows.append(row)
        self.rows_layout.addWidget(row)

    def delete_row(self, row_widget):
        """Remove a row both from the layout and our list."""
        self.rows_layout.removeWidget(row_widget)
        row_widget.setParent(None)
        self.rows.remove(row_widget)

    def get_selection(self):
        """Returns all selection rows in the table rows as a list of dicts."""
        return [row.get_row_select() for row in self.rows]
    
    def toggle_hidden(self, state: bool=True):
        """Toggles between hidden options and showing all options

        :param state: whether to hide the options or not
        :type state: bool
        """

        self.hide = state
        for idx, row in enumerate(self.rows):
            row._show_hide(self.hide)

    def save_select(self):
        """Callback when save button is clicked
        """

        try:
            header = ['dispenseWell'] + self.features
            rows = self.get_selection()
            df = pd.DataFrame(rows)[header]
            df.to_csv(self.pickable_path, index=False)
            QMessageBox.information(self, 'Saved', f'Selection saved to {self.pickable_path}. \n\nReady to Pick!')
            logging.info(f'Selection saved to {self.pickable_path}. Ready to Pick.')
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save: {e}")
            logging.info('Did not save the pickable file')


class AddRow(QWidget):
    """Widget helper to add a row to the pick selection GUI
    """

    def __init__(self, wells, features, deselect, hide=bool, on_delete=None):
        """
        :param wells: well names passed from dispense plate well names
        :type wells: list
        :param features: featue columns
        :type features: list
        :param deselect: columns not to include in pick selection
        :type deselect: list
        :param hide: parameter whether to hide deselect columns or not
        :type hide: bool
        :param on_delete: delete callback
        :type on_delete: function callback
        """

        super().__init__()

        self.cols = features
        self.deselect_cols = deselect
        self.on_delete = on_delete
        
        self.layout = QHBoxLayout(self)
        self.well_dropdown = QComboBox()
        self.well_dropdown.addItems(wells)
        self.layout.addWidget(self.well_dropdown)

        self.checkboxes = {}
        for col in features:
            cb = QCheckBox(col)
            if col == 'singlet':
                cb.setChecked(True)
            if col in self.deselect_cols and hide:
                cb.hide()
            self.checkboxes[col] = cb
            self.layout.addWidget(cb)
        self.setLayout(self.layout)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._delete_self)
        self.layout.addWidget(self.delete_btn)
        self.setLayout(self.layout)

    def get_row_select(self):
        """Return results from selection
        """

        well = self.well_dropdown.currentText()
        selection = {col: int(self.checkboxes[col].isChecked()) for col in self.cols}
        return {'dispenseWell': well, **selection}

    def _show_hide(self, hide: bool=True):
        """Determine whether to show the full list of selection features
        :param hide: whether to show or hide
        :type hide: bool
        """

        for col in self.deselect_cols:
            if col in self.checkboxes:
                self.checkboxes[col].setHidden(hide)
                if hide:
                    self.checkboxes[col].setChecked(False)        
        self.checkboxes['singlet'].setChecked(hide)

    def _delete_self(self):
        """Calls the parent callback to remove this row from the parent layout/list."""
        if self.on_delete:
            self.on_delete(self)