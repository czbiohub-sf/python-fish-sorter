import csv
import json
import logging
import napari
import numpy as np
import os
import pandas as pd
import sys
from datetime import datetime
from napari.components.viewer_model import ViewerModel
from napari.qt import QtViewer
from pathlib import Path
from pymmcore_plus import CMMCorePlus
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel, 
    QPushButton, 
    QSizePolicy, 
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from skimage import data, draw
from tifffile import imread
from typing import List, Optional, Tuple, Callable




#TODO check which class to import
from  fish_sorter.helpers.mapping import Mapping




class NapariPts():
    """Add points layer of the well locations to the image mosaic in napari.
    """
    
    def __init__(self, parent_dir, total_wells, viewer=None, mapping=None):
        """Load well array file of well location points and load fish features. 
        
        :param parent_dir: parent directory for array config files
        :type parent_dir: str
        :param total_wells: descriptor for the agarose array
        :type total_wells: int
        :param viewer: Optional napari viewer to use, otherwise a new one is created
        :type viewer: napari.Viewer, optional
        :param mapping: instance of mapping class
        :type mapping: class instance

        :raises FileNotFoundError: loggings critical if the array config file not found
        """

        array_dir = Path(parent_dir) / "configs/arrays"
        self.array_data = {}
        self.total_wells = total_wells
        num_wells = str(self.total_wells)
        for filename in os.listdir(array_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(array_dir, filename)
                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        var_name = os.path.splitext(filename)[0]
                        if num_wells in var_name:
                            self.var_name = var_name
                            self.array_data[var_name] = data
                            logging.info('Loaded {} config file'.format(var_name))
                except FileNotFoundError:
                    logging.critical("Config file not found")

        feat_dir = Path(parent_dir) / "configs/pick"
        self.feat_data = {}
        self.features = {}
        self.well_feat = {}
        self.fish_feat = {}
        for filename in os.listdir(feat_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(feat_dir, filename)
                try:
                    with open(file_path, 'r') as file:
                       data = json.load(file)
                       self.feat_data = data['larvae']
                       logging.info('Loaded {} config file'.format(filename))


                       #TODO change to handle whether larvae / embryos / something else
                             
                except FileNotFoundError:
                    logging.critical("Config file not found")
        self._feat()
        self.mask = {}

        if viewer is None:
            self.viewer = napari.Viewer()
        else:
            self.viewer = viewer
        self.viewer.window._qt_window.setFocusPolicy(Qt.StrongFocus)


        #Temporary for testing
        #TODO decide how to handle preloading mosaics
        FITC_path = '/Users/diane.wiener/Documents/fish-sorter-data/2024-02-15-cldnb-mScarlet_she-GFP/51hpf-pick1_FITC_2.5x_mosaic.tif'
        FITC_mosaic = np.array(imread(FITC_path))
        TXR_path = '/Users/diane.wiener/Documents/fish-sorter-data/2024-02-15-cldnb-mScarlet_she-GFP/51hpf-pick1_TXR_2.5x_mosaic.tif'
        TXR_mosaic = np.array(imread(TXR_path))

        #TODO add in mapping / imaging_plate class to go between image and real coordinates, not hard code these

        self.viewer.add_image(FITC_mosaic, colormap='green', contrast_limits = (FITC_mosaic.min(), FITC_mosaic.max()), opacity=0.5, name='FITC')
        self.viewer.add_image(TXR_mosaic, colormap='red', contrast_limits = (TXR_mosaic.min(), TXR_mosaic.max()), opacity=0.5, name='TXR')
        
            
        pts = self._points()
        self.points_layer, points = self.load_points(pts)
        self.points_layer.mode = 'select'
        
        self._key_binding()
        for key, feature in self.key_feature_map.items():
            self.viewer.bind_key(key)(self._toggle_feature(feature))
        
        self.extract_fish(points)
        
        
        #TODO handle preselecting fish
        #TODO add in zoom in of fish classified as singlets for classification


        #TODO pre select and then cycle though preselected fish but for now index at 0
        self.current_well = 0
        
        self.classify_widget = self._create_classify()
        # self.classify_widget = QWidget()
        # layout = QVBoxLayout()
        # layout.addWidget(self._create_well_display(2, self.well_extract))
        # layout.addWidget(self._display_key_bindings())
        # self.classify_widget.setLayout(layout)
        self.viewer.window.add_dock_widget(self.classify_widget, name= 'Classification')
         
        self.viewer.bind_key("Right", self._next_well)
        self.viewer.bind_key("Left", self._previous_well)

      
        

        self.save_data()

        #TODO will need to add in loading previous classifications
            
        napari.run()

    def _feat(self):
        """Loads the feature list
        """

        self.features['Well'] = np.array(self.array_data[self.var_name]['wells']['well_names'])

        self.well_feat = self.feat_data['well_class']
        self.deselect_rules = self.feat_data['well_class']['deselect']
        self.fish_feat = self.feat_data['feature_class']

        for feature, feature_data in self.well_feat.items():
            if feature != 'deselect':
                self.features[feature] = np.full(self.total_wells, feature_data['preset'])

        for feature, feature_data in self.fish_feat.items():
            self.features[feature] = np.full(self.total_wells, feature_data['preset'])

    def _key_binding(self):
        """Binds hot keys for classification based on the feature config keys
        """

        self.key_feature_map = {}

        for feature, feature_data in self.fish_feat.items():
            key = feature_data['key']
            self.key_feature_map[key] = feature

        for feature, feature_data in self.well_feat.items():
            if feature != 'deselect':
                key = feature_data['key']
                self.key_feature_map[key] = feature

    def _points(self) -> List[Tuple[float, float]]:
        """Open array with well coordinates and transform to napari viewer coordinates
        and scale to the image

        :return: x, y coordinates of the center point location of the wells 
        :rtype: numpy array
        """

        well_coords = self.array_data[self.var_name]['wells']['well_coordinates']
        points_coords = np.array(well_coords).reshape(-1,2)
        # img_points = self.mapping.
        points_coords = points_coords[:, ::-1]


        #TODO convert real coordinates to image coordinates
        #points = STAGE_2_IMG(points_coords)


        return points_coords

    def _toggle_feature(self, feature_name)-> Callable[[napari.utils.events.Event], None]:
        """Creates a callback toggle for a specific feature defined by feature_name

        :param feature_name: feature name loaded from the feature data in the config file
        :type feature_name: str

        :returns: callback function that toggles a feature on keypress
        :rtype: Callable[[Event], None]
        """ 

        def _toggle_annotation(event):
            """Callback for the key press event
            Used to classify features
            For some features defined in the config, when it is selected True, the others
            are deselected to False

            :param event: key press of hotkey binding from key binding function
            :type event: Event of Napari Qt Event loop
            """

            selected_points = list(self.points_layer.selected_data)
            if len(selected_points) > 0:
                feature_values = self.points_layer.features[feature_name]
                feature_values[selected_points] = ~feature_values[selected_points]
                self.points_layer.features[feature_name] = feature_values

                if feature_name in self.deselect_rules and feature_values[selected_points].iloc[0]:
                    for feat in self.deselect_rules[feature_name]:
                        feature_values = self.points_layer.features[feat]
                        feature_values[selected_points] = False
                        self.points_layer.features[feat] = feature_values
                                            
            self.refresh()
            self.points_layer.mode = 'select'

        return _toggle_annotation

    def save_data(self):
        """Saves the classification once the user pushes the button
        """

        self.class_btn = QPushButton("Save Classification")

        container_widget = QWidget()
        layout = QGridLayout(container_widget)  
        layout.addWidget(self.class_btn, 1, 0)
    
        self.viewer.window.add_dock_widget(container_widget, area='right')

        def _save_it():
            """Saves the classification data from the points layer to csv on button press

            Saves the classification and a template for pickable features as csv files
            """

            class_df = pd.DataFrame(self.points_layer.features)

            if 'Well' in class_df.columns:
                class_df.rename(columns={'Well': 'slotName'}, inplace=True)

            prefix = "fish_test"
            file_path = '/Users/diane.wiener/Documents/fish-sorter-data/2024-02-15-cldnb-mScarlet_she-GFP/'
            
            #TODO load in the prefix name and file path
            #Do we want timestamp, or simply overwrite?

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{timestamp}_{prefix}_classifications.csv"
            classified = os.path.join(file_path, file_name)
            class_df.to_csv(classified, index=False)
            logging.info(f'Classification saved as {classified}')

            pickable_file_name = f"{timestamp}_{prefix}_pickable.csv"
            pick = os.path.join(file_path, pickable_file_name)
            headers_df = pd.DataFrame(columns=class_df.columns)
            headers_df.rename(columns={'slotName': 'dispenseWell'}, inplace=True) 
            headers_df.to_csv(pick, index=False)
            logging.info(f'Pickable template saved as {pick}')
            
        self.class_btn.clicked.connect(_save_it)
    
    def load_points(self, points_coords) -> napari.layers.Points:
        """Load the points layer into the napari viewer

        :param points_coords: x, y coordinates for points defining the well locations
        :type points_coords: numpy array

        :return: napari points layer with points at each of the points provided by points_coords
        :rtype: napari.points.Layer
        """

        face_color_cycle = ['white', 'red']

        #TODO when mapping function is working remove these, and load in that and calibration
        scale_factor = 1 / 2.6
        translation_offset = (1120, 1000)
        rotation_angle = 0.5

        self.points_layer = self.viewer.add_points(
            points_coords,
            name = 'Well Locations',
            features = self.features,
            size = 100,
            scale = (scale_factor, scale_factor),
            translate = translation_offset,
            rotate = rotation_angle,
            face_color = 'empty',
            face_color_cycle = face_color_cycle
        )

        rotation_matrix = np.array([
            [np.cos(np.deg2rad(rotation_angle)), -np.sin(np.deg2rad(rotation_angle))],
            [np.sin(np.deg2rad(rotation_angle)), np.cos(np.deg2rad(rotation_angle))]
        ])

        scaled_pts = points_coords * scale_factor
        rot_pts = scaled_pts @ rotation_matrix.T
        points = rot_pts + translation_offset

        return self.points_layer, points

    def refresh(self):
        """Refresh the points layer after event
        """

        self.points_layer.data = self._points()
        self.points_layer.refresh_colors(update_color_mapping=False)

    def extract_fish(self, points):
        """Finds the locations of positive wells

        :param points: x, y coords for center location defining the well locations in the points layer
        :type points: numpy points
        """

        #TODO make sure padding makes sense for fish detection / viewing for classification
        self._well_mask()
        self.well_extract = self._extract_wells(points)



    def _well_mask(self, padding: int=100):
        """Create a mask of the well shape

        :param padding: extra pixels from the edge around the well shape to include in the mask
        :type padding: int
        """

        #TODO make sure it is correct with real to image conversion with mapping
        #Do not want a hard coded scale factor
        scale_factor = 1 / 2.6
        width = int(round(self.array_data[self.var_name]['array_design']['slot_length'] * scale_factor))
        height = int(round(self.array_data[self.var_name]['array_design']['slot_width'] * scale_factor))

        padded_height = height + 2 * padding
        padded_width = width + 2 * padding

        #TODO will this need scaling by the image conversion? probably

        self.mask = np.zeros((padded_height, padded_width), dtype=bool)
        
        if self.array_data[self.var_name]['array_design']['well_shape'] == "rectangular_array":
            start_row, start_col = padding, padding
            rr, cc = draw.rectangle(start=(start_row, start_col), extent=(height, width))
        else:
            center = (padded_height // 2, padded_width // 2)
            radius = min(height, width) // 2  
            rr, cc = draw.disk(center, radius)

        self.mask[rr, cc] = True
    
    def _extract_wells(self, points) -> dict:
        """Cuts a well centered around the points in the points layer of the image
        of the size defined in the array and displays the image layer

        :param points: x, y coordinates for center location points defining the well locations
        :type points: numpy points

        :return: layer name, extracted region for each layer for each point
        :rtype: dict 
        """

        mask_height, mask_width = self.mask.shape
        half_mask_height, half_mask_width = mask_height // 2, mask_width // 2

        image_layers = [layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
        extracted_regions = []

        for i, point in enumerate(points):
            width_center, height_center  = int(point[1]), int(point[0])
            region_by_layer = {}

            for layer in image_layers:
                img_data = layer.data
                width_min = max(width_center - half_mask_width, 0)
                width_max = min(width_center + half_mask_width, img_data.shape[1])
                height_min = max(height_center - half_mask_height, 0)
                height_max = min(height_center + half_mask_height, img_data.shape[0])
                region = img_data[height_min : height_max, width_min : width_max]

                mask_height_min = max(0, height_min - (height_center - half_mask_height))
                mask_height_max = mask_height_min + region.shape[0]
                mask_width_min = max(0, width_min - (width_center - half_mask_width))
                mask_width_max = mask_width_min + region.shape[1]

                mask_width_max = min(mask_width, mask_width_max)
                mask_height_max = min(mask_height, mask_height_max)
            
                overlap_width = mask_width_max - mask_width_min
                overlap_height = mask_height_max - mask_height_min
            
                if overlap_width > 0 and overlap_height > 0:
                    masked_region = np.zeros_like(region)
                    masked_region[:overlap_height, :overlap_width] = (
                        region[:overlap_height, :overlap_width] * self.mask[mask_height_min:mask_height_max, mask_width_min:mask_width_max]
                    )
                    region_by_layer[layer.name] = masked_region
                    # self.viewer.add_image(
                    #     masked_region,
                    #     name=f'Well {i} -- {layer.name}',
                    #     translate=(height_center - region.shape[0] // 2, width_center - region.shape[1] // 2)
                    # )
                else:
                    logging.info(f'Skipping point at ({point[0]}, {point[1]}) due to zero overlap dimensions')
            extracted_regions.append(region_by_layer)
   
        return extracted_regions

    def _create_classify(self):
        """Classification side widget with key bindings and well viewer windows

        :return: A QWidget with key binding map
        :rtype: QWidget
        """

        widget = QWidget()
        layout = QVBoxLayout()

        self.well_disp = self._create_well_display(self.current_well, self.well_extract)
        layout.addWidget(self.well_disp)
        key_binding = self._display_key_bindings()
        layout.addWidget(key_binding)
        widget.setLayout(layout)
        
        return widget
    
    def _display_key_bindings(self, columns: int=3):
        """Show the custom key bindings for classification

        :param columns: number of columns to list key bindings
        :type columns: int

        :return: A QWidget with key binding map
        :rtype: QWidget
        """

        widget = QWidget()
        layout = QGridLayout()
        layout.addWidget(QLabel('Key Bindings:'), 0, 0, 1, columns)
        
        row, col = 1, 0
        for key, feature in self.key_feature_map.items():
            layout.addWidget(QLabel(f'{key}: {feature}'), row, col)
            col += 1
            if col >= columns:
                col = 0
                row +=1
        widget.setLayout(layout)

        return widget

    def _create_well_display(self, well: int, extracted_regions):
        """Show the custom key bindings for classification

        Displays each layer in a different subimage viewer for a specific well

        :param well: the well id, currently 0 indexed
        :type well: int
        :param extracted_regions: extracted regions by point and layer 
        :type extracted_regions: dict

        :return: A QWidget with the multiviewers
        :rtype: QWidget
        """

        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel(f'Well {well}'))
        splitter = QSplitter()
        splitter.setOrientation(Qt.Vertical)
        point_region = extracted_regions[well]

        for layer_name, masked_region in point_region.items():
            if masked_region is not None:
                logging.info(f'Processing extracted region from {layer_name} for well {well}')
                container = QWidget()
                container_layout = QVBoxLayout()
                container_layout.addWidget(QLabel(layer_name))
                viewer_model = ViewerModel(title=layer_name)
                qt_viewer = QtViewerWrap(self.viewer, viewer_model)

                #TODO make sure that contrast limits are set according to overall contrast limits
                viewer_model.add_image(masked_region, contrast_limits=(masked_region.min(), masked_region.max()), name=f'{layer_name}')
                container_layout.addWidget(qt_viewer)
                container.setLayout(container_layout)
                splitter.addWidget(container)
            else:
                logging.info(f'No data for {layer_name} at well {well}')
        layout.addWidget(splitter)
        widget.setLayout(layout)

        return widget

    def _next_well(self, event=None):
        """Updates the viewer window with the next well when the right arrow key is pressed

        :param event: key press of right arrow key
        :type event: Event of Napari Qt Event loop
        """

        if self.current_well < self.total_wells - 1:
            self.current_well += 1
        else:
            self.current_well = 0    
        self._update_well_display()
        self._refocus_viewer()
        self._select_current_point()
    
    def _previous_well(self, event=None):
        """Updates the viewer window with the previous well when the left arrow key is pressed

        :param event: key press of left arrow key
        :type event: Event of Napari Qt Event loop
        """

        if self.current_well > 0:
            self.current_well -= 1
        else:
            self.current_well = self.total_wells - 1    
        self._update_well_display()
        self._select_current_point()
        self._refocus_viewer()

    def _update_well_display(self):
        """Updates the side Classify viewer with a new well
        """

        self.classify_widget.layout().removeWidget(self.well_disp)
        self.well_disp.deleteLater()
        self.well_disp = self._create_well_display(self.current_well, self.well_extract)
        self.classify_widget.layout().insertWidget(0, self.well_disp)

    def _select_current_point(self):
        """Select the point for the well currently in view for classification
        """

        if self.points_layer is not None:
            self.points_layer.selected_data = {self.current_well}
            self.points_layer.refresh()

    def _refocus_viewer(self):
        """Needed so that classification can be done in the main window
        """

        self.viewer.layers.selection.active = self.points_layer
        self.viewer.window._qt_window.setFocus()
       
class QtViewerWrap(QtViewer):
    def __init__(self, main_viewer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.main_viewer = main_viewer

    def _qt_open(
        self,
        filenames: list,
        stack: bool,
        plugin: Optional[str] = None,
        layer_type: Optional[str] = None,
        **kwargs,
    ):
        """for drag and drop open files"""
        self.main_viewer.window._qt_viewer._qt_open(
            filenames, stack, plugin, layer_type, **kwargs
        )

if __name__ == '__main__':
    NapariPts('/Users/diane.wiener/Documents/GitHub/python-fish-sorter/fish_sorter', 595)