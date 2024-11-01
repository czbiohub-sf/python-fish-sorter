import csv
import json
import logging
import napari
import numpy as np
import os
import pandas as pd
import sys
from datetime import datetime
from pathlib import Path
from pymmcore_plus import CMMCorePlus
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QLabel, QPushButton, QSizePolicy, QWidget, QGridLayout
from skimage import data, draw
from tifffile import imread
from typing import List, Optional, Tuple




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

        
        #Temporary for testing
        #TODO decide how to handle preloading mosaics
        im_path = '/Users/diane.wiener/Documents/fish-sorter-data/2024-02-15-cldnb-mScarlet_she-GFP/51hpf-pick1_FITC_2.5x_mosaic.tif'
        mosaic = np.array(imread(im_path))

        if viewer is None:
            self.viewer = napari.Viewer()
            img_layer = self.viewer.add_image(mosaic, name='FITC')

            #TODO add in mapping / imaging_plate class to go between image and real coordinates, not hard code these
            img_layer.translate = [-2600,-2600]
            img_layer.scale = [2.6, 2.6]
        else:
            self.viewer = viewer
            
        pts = self._points()
        self.points_layer = self.load_points(pts)
        self.points_layer.mode = 'select'
        
        self._key_binding()
        for key, feature in self.key_feature_map.items():
            self.viewer.bind_key(key)(self._toggle_feature(feature))
        self.display_key_bindings()

        self.extract_fish()

        self.save_data()


        #TODO handle preselecting fish
        #TODO add in zoom in of fish classified as singlets for classification

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
    
    def display_key_bindings(self):
        """Show the custom key bindings for classification
        """

        label = QLabel()
        label.setText("Key Bindings:\n" + "\n".join(f"{key}: {self.key_feature_map[key]}" for key in self.key_feature_map))
        self.viewer.window.add_dock_widget(label, area='right')

    def save_data(self):
        """Saves the classification once the user pushes the button
        """

        self.class_btn = QPushButton("Save Classification")

        container_widget = QWidget()
        layout = QGridLayout(container_widget)  
        layout.addWidget(self.class_btn, 1, 0)
    
        self.viewer.window.add_dock_widget(container_widget, area='right')

        def _save_it():
            """Saves the data

            :param event: key press of hotkey binding from key binding function
            :type event: Event of Napari Qt Event loop
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
    
    def load_points(self, points_coords) -> napari.points.Layer:
        """Load the points layer into the napari viewer

        :param points_coords: x, y coordinates for points defining the well locations
        :type points_coords: numpy array

        :return: napari points layer with points at each of the points provided by points_coords
        :rtype: napari.points.Layer
        """

        face_color_cycle = ['white', 'red']

        self.points_layer = self.viewer.add_points(
            points_coords,
            features = self.features,
            size = 100,
            face_color = 'empty',
            face_color_cycle = face_color_cycle
        )

        return self.points_layer

    def refresh(self):
        """Refresh the points layer after event
        """

        self.points_layer.data = self._points()
        self.points_layer.refresh_colors(update_color_mapping=False)

    def extract_fish(self):
        """Finds the locations of positive wells
        """

        self._well_mask()
        self._extract_well

        
    

    def _well_mask(self):
        """Create a mask of the well shape
        """

        length = self.array_data[self.var_name]['array_design']['slot_length']
        width = self.array_data[self.var_name]['array_design']['slot_width']

        self.mask = np.zeros((length, width), dtype=bool)
        
        if self.array_data[self.var_name]['array_design']['well_shape'] == "rectangular_array":
            rr, cc = draw.rectangle(start=(0,0), end=(length, width))
        else:
            rr, cc = draw.disk((0,0), length)

        self.mask[rr, cc] = True
    
    def _extract_well(self, points):
        """Cuts a well centered around the points in the points layer of the image
        of the size defined in the array and displays the image layer

        :param points: x, y coordinates for selected points defining the well locations
        :type points: numpy points
        """

        self.array_data[self.var_name]['wells']['well_coordinates']

if __name__ == '__main__':
    NapariPts('/Users/diane.wiener/Documents/GitHub/python-fish-sorter/fish_sorter', 595)