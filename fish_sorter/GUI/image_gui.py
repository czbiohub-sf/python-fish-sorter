import numpy as np
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

from fish_sorter.constants import FOV_WIDTH, FOV_HEIGHT


class ImageWidget(QWidget):

    def __init__(self, viewer, parent: QWidget | None=None):
        """
        :param viewer: napari viewer to use
        :type viewer: napari.Viewer
        """

        super().__init__(parent=parent)
        self.mmc = CMMCorePlus().instance()
        self.viewer = viewer

        self.btn = QPushButton("Stitch mosaic")
        self.class_btn = QPushButton("Classify")
        self.cross_btn = QPushButton('Crosshairs')
       
        self.crosshair_layer = 'crosshairs'
        self.cross_btn.setToolTip('Toggle crosshairs')
        self.cross_btn.clicked.connect(self._toggle_crosshairs)

        layout = QHBoxLayout()
        layout.addWidget(self.btn)
        layout.addWidget(self.class_btn)
        layout.addWidget(self.cross_btn)
        self.setLayout(layout)
        
    def _create_crosshairs(self):

        preview_layer = None
        for layer in self.viewer.layers:
            if layer.name == 'preview':
                preview_layer = layer
                break

        if preview_layer is None:
            return

        lines = [
            [[0, FOV_HEIGHT / 2], [FOV_WIDTH, FOV_HEIGHT / 2]],
            [[FOV_WIDTH / 2, 0], [FOV_WIDTH / 2, FOV_HEIGHT]]
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

    def _toggle_crosshairs(self):
        """Toggles the crosshairs on the button press
        """

        if self.crosshair_layer in self.viewer.layers:
            self.viewer.layers.remove(self.crosshair_layer)
        else:
            self._create_crosshairs()