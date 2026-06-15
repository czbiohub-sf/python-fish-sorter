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
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton, 
    QSizePolicy, 
    QScrollArea, 
    QVBoxLayout, 
    QWidget
)

from fish_sorter.GUI.picking import Pick, latest_classifications_csv

COLOR_TYPES = Union[
    QColor,
    int,
    str,
    Qt.GlobalColor,
    "tuple[int, int, int, int]",
    "tuple[int, int, int]"
]

log = logging.getLogger(__name__)


def discover_pick_features_and_combos(class_df, well_class_features):
    """Derive Pick Selection feature columns + prepopulation combos from a classification.

    ``feature_cols`` = the standard well-class checkboxes (from config) followed by the
    *dynamic* feature columns found in ``class_df`` — i.e. every column beyond the
    identifier/global set and the well-class set. For a Finding Dory wide CSV those are
    the per-channel ``{channel}_{group}`` columns; for a classical CSV they are the
    ``feature_class`` columns (gEye, …). Sourcing them from the file guarantees the
    names line up with ``picking.match_pick``'s column-intersection join.

    ``combos`` = one prepopulation row per distinct combination of the dynamic columns
    among ``singlet`` wells (including the all-zero "singlet, no group" catch-all),
    sorted by descending well-count (most-assigned first). Each combo is
    ``{'checks': {col: 1, ...}, 'count': n}`` with ``singlet`` always checked.
    empty/multiple/deformed wells never form combos (they are not picked).

    :returns: ``(feature_cols, combos)``
    """
    standard_ids = {'slotName', 'well_name', 'lHead'}
    wc = list(well_class_features)
    dynamic = [c for c in class_df.columns if c not in wc and c not in standard_ids]
    feature_cols = wc + dynamic

    combos = []
    if 'singlet' in class_df.columns:
        singlets = class_df[class_df['singlet'] == 1]
    else:
        singlets = class_df.iloc[0:0]

    if len(singlets) and dynamic:
        grouped = singlets.groupby(dynamic, sort=False).size().reset_index(name='count')
        for _, grow in grouped.iterrows():
            checks = {'singlet': 1}
            for col in dynamic:
                if int(grow[col]) == 1:
                    checks[col] = 1
            combos.append({'checks': checks, 'count': int(grow['count'])})
        combos.sort(key=lambda c: c['count'], reverse=True)
    elif len(singlets):
        # No dynamic feature columns at all → a single catch-all singlet combo.
        combos.append({'checks': {'singlet': 1}, 'count': int(len(singlets))})

    return feature_cols, combos


def map_combos_to_wells(combos, wells):
    """Pair combos with dispense wells in order, capped to the number of wells.

    :returns: ``(mapped, dropped)`` — ``mapped`` is a list of ``(well, combo)`` for
        the first ``len(wells)`` combos (the most-assigned ones, since combos arrive
        sorted by descending count) and ``dropped`` is how many combos had no well.
    """
    mapped = [(wells[i], combo) for i, combo in enumerate(combos[:len(wells)])]
    dropped = max(0, len(combos) - len(wells))
    return mapped, dropped


def group_features_for_display(features, well_class):
    """Partition a row's features into display lines.

    :returns: ``(well_class_feats, channel_groups, ungrouped)`` where ``channel_groups``
        is an ordered list of ``[channel, [feature, ...]]`` for the per-channel
        ``{channel}_{group}`` columns (channel = text before the first ``_``). Well-class
        columns are separated out first so e.g. ``wrong_o`` is never mistaken for a
        channel; columns with no ``_`` (classical feature_class) go to ``ungrouped``.
        Every feature lands in exactly one bucket.
    """
    wc_set = set(well_class)
    well_class_feats = [f for f in features if f in wc_set]
    channel_groups = []
    pos = {}
    ungrouped = []
    for f in features:
        if f in wc_set:
            continue
        if '_' in f:
            channel = f.split('_', 1)[0]
            if channel not in pos:
                pos[channel] = len(channel_groups)
                channel_groups.append([channel, []])
            channel_groups[pos[channel]][1].append(f)
        else:
            ungrouped.append(f)
    return well_class_feats, channel_groups, ungrouped


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
        self.features = []
        self.combos = []
        self.layout = QVBoxLayout(self)
        self.rows_layout = QVBoxLayout()
        self.rows_container = QWidget()
        self.rows_container.setLayout(self.rows_layout)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.rows_container)
        self.layout.addWidget(self.scroll)

        self.add_row_btn = QPushButton('Add Row')
        self.add_row_btn.clicked.connect(lambda: self.add_row())

        self.refresh_btn = QPushButton('Refresh')
        self.refresh_btn.setToolTip('Reload classes + combos from the latest saved classification')
        self.refresh_btn.clicked.connect(lambda: self.refresh())

        self.save_btn = QPushButton('Save')
        self.save_btn.clicked.connect(self.save_select)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.add_row_btn)
        btn_layout.addWidget(self.refresh_btn)
        btn_layout.addWidget(self.save_btn)
        self.layout.addLayout(btn_layout)

        self.refresh()

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
        # Split config classes: well_class are the standard per-well checkboxes; the
        # feature_class is the *fallback* feature set used only when no classification
        # CSV exists yet (otherwise features are discovered from the CSV in _discover).
        self.well_class = [
            feat for feat in feat_data['well_class'].keys() if feat != 'deselect'
        ]
        self.feature_class = [
            feat for feat in feat_data['feature_class'].keys() if feat != 'lHead'
        ]

    def _discover(self):
        """Discover pick features + cross-channel combos from the latest classification.

        Decoupled from the live LabelStore: reads the newest ``*classifications.csv``
        in the pick directory so Pick Selection reflects whatever was saved last
        (classical or Finding Dory). Falls back to the config feature set when no
        classification exists yet.
        """
        class_path = latest_classifications_csv(self.pick.pick_dir)
        if class_path is not None:
            try:
                class_df = pd.read_csv(class_path)
                self.features, self.combos = discover_pick_features_and_combos(
                    class_df, self.well_class
                )
                logging.info(
                    f'Pick Selection discovered {len(self.features)} features and '
                    f'{len(self.combos)} combo(s) from {os.path.basename(class_path)}'
                )
                return
            except Exception as e:
                logging.warning(
                    f'Could not read {class_path}: {e}; falling back to config features.'
                )
        # Fallback: no classification (or unreadable) → config well_class + feature_class.
        self.features = list(self.well_class) + list(self.feature_class)
        self.combos = []

    def refresh(self):
        """Reload features/combos from the latest classification and rebuild rows.

        Lets the user switch classifiers (classical <-> Finding Dory) and re-pick
        without restarting the app. Existing rows are discarded because the feature
        columns differ between modes.
        """
        self._discover()

        for row in list(self.rows):
            self.rows_layout.removeWidget(row)
            row.setParent(None)
        self.rows = []

        if self.combos:
            mapped, dropped = map_combos_to_wells(self.combos, self.well)
            for well, combo in mapped:
                self.add_row(preset_well=well, preset_checks=combo['checks'])
            if dropped > 0:
                msg = (
                    f"{dropped} combo(s) were not auto-mapped: only {len(self.well)} dispense "
                    f"well(s) available for {len(self.combos)} combo(s). The most-assigned "
                    f"combos were kept; add rows manually for the rest if needed."
                )
                logging.warning(msg)
                QMessageBox.warning(self, 'Not enough dispense wells', msg)
        else:
            self.add_row()

    def add_row(self, preset_well=None, preset_checks=None):
        """Adds a row to the selection GUI

        :param preset_well: dispense well to pre-select in the dropdown
        :type preset_well: str | None
        :param preset_checks: {feature: bool/int} to pre-check; when given it fully
            determines checkbox state (otherwise `singlet` defaults to checked)
        :type preset_checks: dict | None
        """

        row = AddRow(self.well, self.features, self.deselect, on_delete=self.delete_row,
                     preset_well=preset_well, preset_checks=preset_checks,
                     well_class=self.well_class)
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

    def __init__(self, wells, features, deselect, on_delete=None,
                 preset_well=None, preset_checks=None, well_class=None):
        """
        :param wells: well names passed from dispense plate well names
        :type wells: list
        :param features: featue columns
        :type features: list
        :param deselect: columns not to include in pick selection
        :type deselect: list
        :param on_delete: delete callback
        :type on_delete: function callback
        :param preset_well: dispense well to pre-select (else first well)
        :type preset_well: str | None
        :param preset_checks: {feature: bool/int} to pre-check; when given it fully
            determines checkbox state (otherwise `singlet` defaults to checked)
        :type preset_checks: dict | None
        :param well_class: standard well-class feature names (kept on the top line);
            remaining features are grouped onto one line per channel for readability
        :type well_class: list | None
        """

        super().__init__()

        self.cols = features
        self.deselect_cols = deselect
        self.on_delete = on_delete

        well_class_feats, channel_groups, ungrouped = group_features_for_display(
            features, well_class or []
        )

        self.checkboxes = {}

        def _make_checkbox(col):
            cb = QCheckBox(col)
            if preset_checks is not None:
                cb.setChecked(bool(preset_checks.get(col, False)))
            elif col == 'singlet':
                cb.setChecked(True)
            self.checkboxes[col] = cb
            return cb

        outer = QVBoxLayout(self)

        # Top line: dispense-well dropdown + standard well-class checkboxes + Delete.
        top = QHBoxLayout()
        self.well_dropdown = QComboBox()
        self.well_dropdown.addItems(wells)
        if preset_well is not None and preset_well in wells:
            self.well_dropdown.setCurrentText(preset_well)
        top.addWidget(self.well_dropdown)
        for col in well_class_feats:
            top.addWidget(_make_checkbox(col))
        top.addStretch(1)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._delete_self)
        top.addWidget(self.delete_btn)
        outer.addLayout(top)

        # Classical feature_class columns (no channel prefix) share one line.
        if ungrouped:
            line = QHBoxLayout()
            line.addWidget(QLabel("features:"))
            for col in ungrouped:
                line.addWidget(_make_checkbox(col))
            line.addStretch(1)
            outer.addLayout(line)

        # One line per channel for the per-channel `{channel}_{group}` cluster columns.
        for channel, feats in channel_groups:
            line = QHBoxLayout()
            line.addWidget(QLabel(f"{channel}:"))
            for col in feats:
                line.addWidget(_make_checkbox(col))
            line.addStretch(1)
            outer.addLayout(line)

        # Visual separator so stacked multi-line rows stay distinct.
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        outer.addWidget(sep)

    def get_row_select(self):
        """Return results from selection
        """

        well = self.well_dropdown.currentText()
        selection = {col: int(self.checkboxes[col].isChecked()) for col in self.cols}
        return {'dispenseWell': well, **selection}

    def _delete_self(self):
        """Calls the parent callback to remove this row from the parent layout/list."""
        if self.on_delete:
            self.on_delete(self)