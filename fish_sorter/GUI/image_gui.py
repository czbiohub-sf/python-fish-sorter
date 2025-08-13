import logging
import numpy as np
import re
from typing import List, Optional, Union

from pymmcore_plus import CMMCorePlus
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QGridLayout, 
    QPushButton, 
    QSizePolicy,
    QHBoxLayout,
    QWidget
)

from fish_sorter.constants import CAM_PX_UM, CAM_X_PX, CAM_Y_PX


class ImageWidget(QWidget):

    def __init__(self, viewer, parent: QWidget | None=None):
        """
        :param viewer: napari viewer to use
        :type viewer: napari.Viewer
        """

        super().__init__(parent=parent)
        self.mmc = CMMCorePlus().instance()
        self.viewer = viewer

        self.mosaic_btn = QPushButton("Stitch mosaic")
        self.class_btn = QPushButton("Classify")
        self.cross_btn = QPushButton('Crosshairs')
       
        self.crosshair_layer = 'crosshairs'
        self.cross_btn.setToolTip('Toggle crosshairs')
        self.cross_btn.clicked.connect(self._toggle_crosshairs)

        layout = QHBoxLayout()
        layout.addWidget(self.mosaic_btn)
        layout.addWidget(self.class_btn)
        layout.addWidget(self.cross_btn)
        self.setLayout(layout)
        
    def _create_crosshairs(self):
        """Adds image center crosshairs to the napari viewer
        """
        
        self.viewer.reset_view()
        preview_layer = None
        for layer in self.viewer.layers:
            if layer.name == 'preview':
                preview_layer = layer
                break

        if preview_layer is None:
            return

        lines = [
            [[0, self.fov_h / 2], [self.fov_w, self.fov_h / 2]],
            [[self.fov_w / 2, 0], [self.fov_w / 2, self.fov_h]]
        ]
        
        if self.crosshair_layer in self.viewer.layers:
            self.viewer.layers.remove(self.crosshair_layer)

        layer = self.viewer.add_shapes(
            lines,
            shape_type='line',
            edge_color='yellow',
            edge_width=25,
            name=self.crosshair_layer,
            blending='translucent'
        )
        layer.editable = False
        layer.selectable = False

    def get_mag(self):
        """Helper function to get the current magnification of the microscope
        """

        logging.info('Getting the magnification')
        obj_dev = self.mmc.guessObjectiveDevices()[0]
        obj_label = self.mmc.getStateLabel(obj_dev)

        match = re.search(r'([\d.]+)x', obj_label)
        if match:
            mag = float(match.group(1))
            logging.info(f'Magnification: {mag}')
        else:
            mag = 1.0
            logging.warning(f'Could not parse magnfication from label {obj_label}')

        self.pixel_size_um = CAM_PX_UM / mag
        self.fov_h = CAM_Y_PX * self.pixel_size_um
        self.fov_w = CAM_X_PX * self.pixel_size_um

    def _toggle_crosshairs(self):
        """Toggles the crosshairs on the button press
        """

        if self.crosshair_layer in self.viewer.layers:
            self.viewer.layers.remove(self.crosshair_layer)
        else:
            self.get_mag()
            self._create_crosshairs()