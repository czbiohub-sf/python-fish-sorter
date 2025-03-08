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
from napari.layers import Image
from napari.qt import QtViewer
from pathlib import Path
from pymmcore_plus import CMMCorePlus
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel, 
    QPushButton, 
    QSizePolicy, 
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from skimage import data, draw
from tifffile import imread
from typing import List, Optional, Tuple, Callable

from fish_sorter.hardware.imaging_plate import ImagingPlate

class Classify():
    """Add points layer of the well locations to the image mosaic in napari.
    """
    
    def __init__(self, cfg_dir, array_cfg, mmc, mda, pick_type, prefix, expt_dir, viewer=None):
        """Load pymmcore-plus core, acquisition engine and napari viewer, and load classification features
        
        :param cfg_dir: parent path directory for all of the config files
        :type cfg_dir: Path
        :param array_cfg: array config file
        :type array_cfg: filename with path
        :param mmc: pymmcore-plus core
        :type mmc: pymmcore-plus  core instance
        :param mda: pymmcore-plus multidimensial acquisition engine
        :type mda: pymmcore-plus mda instance
        :param pick_type: user-input pick type from pick type config options
        :type pick_type: str
        :param prefix: experiment name prefix
        :type prefix: str
        :param expt_dir: experiment directory
        :type expt_dir: str
        :param viewer: Optional napari viewer to use, otherwise a new one is created
        :type viewer: napari.Viewer, optional

        :raises FileNotFoundError: loggings critical if the array config file not found
        """

        #TODO will need to add in loading previous classifications
        
        img_array = cfg_dir / 'arrays' / array_cfg
        logging.info(f'Imgaing array file path {img_array}')
        self.iplate = ImagingPlate(mmc, mda, img_array)
        self.iplate.load_wells()
        self.total_wells = self.iplate.wells["array_design"]["rows"] * self.iplate.wells["array_design"]["columns"]
        
        feat_dir = cfg_dir / "pick"
        self.feat_data = {}
        self.features = {}
        self.well_feat = {}
        self.fish_feat = {}
        self.picking = pick_type

        for filename in os.listdir(feat_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(feat_dir, filename)
                try:
                    with open(file_path, 'r') as file:
                       data = json.load(file)
                       self.feat_data = data[self.picking]
                       logging.info('Loaded {} config file'.format(filename))
                except FileNotFoundError:
                    logging.critical("Config file not found")
        self._feat()
        self.mask = {}
        self.fish = []

        if viewer is None:
            self.viewer = napari.Viewer()
        else:
            self.viewer = viewer
        self.viewer.window._qt_window.setFocusPolicy(Qt.StrongFocus)


        #Temporary for testing
        #TODO decide how to handle preloading mosaics
        #stich_mosaic class returns the mosaic as a numpy array
        #probs level higher than this stich mosaic call and then call napari points
        #use the self.viewer to load those layers
        
        # FITC_path = '/Users/diane.wiener/Documents/fish-sorter-data/2024-02-15-cldnb-mScarlet_she-GFP/51hpf-pick1_FITC_2.5x_mosaic.tif'
        # FITC_mosaic = np.array(imread(FITC_path))
        # TXR_path = '/Users/diane.wiener/Documents/fish-sorter-data/2024-02-15-cldnb-mScarlet_she-GFP/51hpf-pick1_TXR_2.5x_mosaic.tif'
        # TXR_mosaic = np.array(imread(TXR_path))

        # self.viewer.add_image(FITC_mosaic, colormap='green', opacity=0.5, name='FITC')
        # self.viewer.add_image(TXR_mosaic, colormap='red', opacity=0.5, name='TXR')

        self.contrast_callbacks = {}
        self.contrast_widget = ContrastWidget(self.viewer)
        self.viewer.window.add_dock_widget(self.contrast_widget, name= 'Contrast', area='left')
    
        pts = self._points()
        self.points_layer = self.load_points(pts)
        self.points_layer.mode = 'select'
        self.points_layer.events.highlight.connect(self._selected_pt)
        
        self._key_binding()
        for key, feature in self.key_feature_map.items():
            self.viewer.bind_key(key)(self._toggle_feature(feature))
        
        self.feature_widget = None
        self.extract_fish(pts)
        self._find_fish_widget(pts)

        if self.picking == 'larvae':
            self.find_orientation()
    
        self.current_well = 0
        self.classify_widget = self._create_classify()
        self.viewer.window.add_dock_widget(self.classify_widget, name= 'Classification')
        self.viewer.bind_key("Right", self._next_well)
        self.viewer.bind_key("Left", self._previous_well)

        self.prefix = prefix
        self.expt_dir = expt_dir
        self.save_data()
            
        napari.run()

    def _feat(self):
        """Loads the feature list
        """

        self.features['Well'] = np.array(self.iplate.wells["names"])
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

        well_coords = self.iplate.wells["calib_px"]
        points_coords = np.array(well_coords).reshape(-1,2)
        points_coords = points_coords[:, ::-1]

        return points_coords

    def _selected_pt(self, event):
        """Callback when a point is selected
        """

        selected_data = self.points_layer.selected_data
        if selected_data:
            selected_idxs = list(selected_data)
            selected_idx = selected_idxs[0]
            self.current_well = selected_idx
            self._update_well_display()
            self._update_feature_display(self.current_well)
            self.refresh()

    def _selected_current_pt(self):
        """Select the point for the current well in the points layer
        """
        
        if self.points_layer is not None:
            self.points_layer.selected_data = {self.current_well}
            self.refresh()
    
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

            boolean_columns = class_df.select_dtypes(include='bool').columns
            class_df[boolean_columns] = class_df[boolean_columns].astype(int)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{timestamp}_{self.prefix}_classifications.csv"
            classified = os.path.join(self.expt_dir, file_name)
            class_df.to_csv(classified, index=False)
            logging.info(f'Classification saved as {classified}')

            pickable_file_name = f"{timestamp}_{prefix}_pickable.csv"
            pick = os.path.join(self.expt_dir, pickable_file_name)
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

        face_color_cycle = ['black', 'white']

        self.points_layer = self.viewer.add_points(
            points_coords,
            name = 'Well Locations',
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
        
        if self.feature_widget is not None:
            self._update_feature_display(self.current_well)

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

        width = int(round(self.iplate.wells["array_design"]["slot_length"] / self.iplate.px2um))
        height = int(round(self.iplate.wells["array_design"]["slot_width"] / self.iplate.px2um))

        padded_height = height + 2 * padding
        padded_width = width + 2 * padding
        self.mask = np.zeros((padded_height, padded_width), dtype=bool)
        
        if self.iplate.wells["array_design"]["well_shape"] == "rectangular_array":
            start_row, start_col = padding, padding
            rr, cc = draw.rectangle(start=(start_row, start_col), extent=(height, width))
        else:
            center = (padded_height // 2, padded_width // 2)
            radius = min(height, width) // 2  
            rr, cc = draw.disk(center, radius)

        self.mask[rr, cc] = True
    
    def _extract_wells(self, points, img_flag: bool=True, mask_layer: str=None, sigma: float=0.25) -> dict:
        """Cuts a well centered around the points in the points layer of the image
        of the size defined in the array and displays the image layer

        :param points: x, y coordinates for center location points defining the well locations
        :type points: numpy points

        :param img_flag: option whether to use the image layers or to create the binary image mask 
        automatic fish detection, True to use the image layers, False to use the binary mask
        :type img_flag: bool 

        :param mask_layer: layer to use to find fish, default None will use all layers
        :type layer_name: str

        :param sigma: number of standard deviations from mean for thresholding the mask
        :type sigma: float

        :return: layer name, extracted region for each layer for each point
        :rtype: dict 
        """

        mask_height, mask_width = self.mask.shape
        half_mask_height, half_mask_width = mask_height // 2, mask_width // 2

        if img_flag:
            image_layers = [{'data': layer.data, 'name': layer.name} for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
        else:
            if mask_layer:
                raw_layers = [layer for layer in self.viewer.layers if layer.name == mask_layer]
                raw_data = raw_layers[0].data
                layer_name = raw_layers[0].name
            else:
                raw_layers = [layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
                raw_data = np.zeros_like(raw_layers[0].data, dtype=np.float32)
                for layer in raw_layers:
                    raw_data += layer.data
                layer_name = 'sum'
            mask_mean = raw_data.mean()
            mask_std = raw_data.std()
            thresh = mask_mean + sigma * mask_std
            binary_mask = raw_data > thresh
            image_layers = [{'data': binary_mask.astype(np.uint8), 'name': layer_name}]

        extracted_regions = []

        for i, point in enumerate(points):
            width_center, height_center  = int(point[1]), int(point[0])
            region_by_layer = {}

            for layer in image_layers:
                img_data = layer['data']
                layer_name = layer['name']
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
                    region_by_layer[layer['name']] = masked_region
                else:
                    logging.info(f'Skipping point at ({point[0]}, {point[1]}) due to zero overlap dimensions')
            extracted_regions.append(region_by_layer)
   
        return extracted_regions
    
    def find_fish(self, points, layer_name=None, sigma=0.25, reset=True, drop=True):
        """Automatically detects fish and fish orientation.

        :param points: x, y coordinates for center location points defining the well locations
        :type points: numpy points

        :param layer_name: layer to use to find fish, default None will use all layers
        :type layer_name: str

        :param sigma: threshold value to compare well intensities to background
        :type simga: float

        :param reset: whether to reset the detected fish prior to the fish detection
        :type reset: bool

        :param drop: whether to drop the four corners from the detected fish list
        :type drop: bool
        """

        if reset:
            empty = self.points_layer.features['empty']
            empty[:] = True
            self.points_layer.features['empty'] = empty
            for feat in self.deselect_rules['empty']:
                self.points_layer.features.loc[empty, feat] = False
        img_data = self._extract_wells(points, False, layer_name, sigma)
        wells_data = [list(region.values())[0].mean() for region in img_data]
        well_mean = np.mean(wells_data)
        well_class = [region_mean > well_mean for region_mean in wells_data]

        if drop:
            corners = []
            tl = 0
            tr = self.iplate.wells["array_design"]["columns"] - 1
            bl = self.total_wells - self.iplate.wells["array_design"]["columns"]
            br = self.total_wells - 1
            corners = [tl, tr, bl, br]
            for idx in corners:
                well_class[idx] = False

        self._update_found_fish(well_class)

    def _update_found_fish(self, wells: List[int]):
        """Updates the feature classification for the found fish and orientation

        :param wells: well locations with fish
        :type wells: List[int]
        """

        singlets = self.points_layer.features['singlet']
        singlets[wells] = True
        singlets_idxs = np.where(singlets)[0]
        self.points_layer.features['singlet'] = singlets
        for feat in self.deselect_rules['singlet']:
            self.points_layer.features.loc[singlets, feat] = False                                 
        self.refresh()
        self.points_layer.mode = 'select'
        self.current_wells = singlets_idxs[0]

    def find_orientation(self):
        """Determines orientation to select side of well for picking
        """

        single = [i for i, val in enumerate(self.points_layer.features['singlet']) if val]
        orientation = {}

        for fish in single:

            if fish >= len(self.well_extract):
                continue
            well_data = self.well_extract[fish]
            channels = list(well_data.values())
            well_total = np.zeros_like(channels[0], dtype=np.float64)
            for channel in channels:
                well_total += channel
            well_total /= len(channels)

            height, width = well_total.shape
            half_width = width // 2
            left = well_total[:height, :half_width]
            right = well_total[:height, half_width:width]
            orientation[fish] = np.sum(left) >= np.sum(right)    

        self._update_orientation(orientation)
    
    def _update_orientation(self, orientation=List[int]):
        """Updates the feature classification for the found fish and orientation

        :param orientation: well location where fish orientation is True
        :type orientation: List[int]
        """

        for idx, orient in orientation.items():
            self.points_layer.features['lHead'][idx] = orient
        self.refresh()
        self.points_layer.mode = 'select'

    def _find_fish_widget(self, points):
        """Call back widget to use to detect wells with fish

        :param points: x, y coordinates for center location points defining the well locations
        :type points: numpy points
        """

        def find_fish_callback(layer_name: str, sigma: float):
            self.find_fish(points, layer_name, sigma, drop=False)
        fish_widget = FishFinderWidget(self.viewer, find_fish_callback)
        self.viewer.window.add_dock_widget(fish_widget, name= 'Finding Nemo', area='left')
    
    def _create_classify(self):
        """Classification side widget with key bindings and well viewer windows

        :return: A QWidget with key binding map
        :rtype: QWidget
        """

        widget = QWidget()
        layout = QVBoxLayout()

        self.well_disp = self._create_well_display(self.current_well, self.well_extract)
        layout.addWidget(self.well_disp)
        
        widget.setLayout(layout)
        return widget
    
    def _display_feature_key_widget(self, well: int, columns: int=3):
        """Shows the features and key bindings for the specific well 
        in the side planel for classification

        :param well: the well id, currently 0 indexed
        :type well: int

        :param columns: number of columns to list key bindings
        :type columns: int

        :return: A QWidget with the multiviewers
        :rtype: QWidget
        """

        widget = QWidget()
        layout = QGridLayout()
        layout.addWidget(QLabel('Feature:'), 0, 0)
        layout.addWidget(QLabel('Value:'), 0, 1)
        layout.addWidget(QLabel('Key Binding:'), 0, 2)
        
        row = 1
        features = self.points_layer.features
        key_map = {feature: key for key, feature in self.key_feature_map.items()}
        
        if features is not None:
            for feature_name, values in features.items():
                feature_value = values[well]
                key_binding = key_map.get(feature_name, " ")
                value_label = QLabel(str(feature_value))
                if isinstance(feature_value, (bool, np.bool_)):
                    color = "green" if feature_value else "red"
                else:
                    color = "white"
                value_label.setStyleSheet(f"color: {color}; font-weight: bold;")
                layout.addWidget(QLabel(feature_name), row, 0)
                layout.addWidget(value_label, row, 1)
                layout.addWidget(QLabel(key_binding), row, 2)

                row += 1

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
        image_layers = {layer.name: layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)}
        self.well_display_layers = {}

        for layer_name, masked_region in point_region.items():
            if masked_region is not None:
                logging.info(f'Processing extracted region from {layer_name} for well {well}')
                container = QWidget()
                container_layout = QVBoxLayout()
                container_layout.addWidget(QLabel(layer_name))
                viewer_model = ViewerModel(title=layer_name)
                qt_viewer = QtViewerWrap(self.viewer, viewer_model)

                if layer_name == 'GFP':
                    color = 'green'
                elif layer_name == 'TXR':
                    color = 'red'
                else:
                    color = 'grey'

                main_layer = image_layers.get(layer_name)
                if main_layer is not None:
                    contrast_limits = main_layer.contrast_limits
                else:
                    contrast_limits = (masked_region.min(), masked_region.max())
                well_layer = viewer_model.add_image(
                    masked_region,
                    colormap=color,
                    contrast_limits=contrast_limits,
                    name=f'{layer_name}'
                )
                container_layout.addWidget(qt_viewer)
                container.setLayout(container_layout)
                splitter.addWidget(container)
                self.well_display_layers[layer_name] = well_layer

                if main_layer is not None:
                    def mini_contrast_callback(main_layer, well_layer):
                        def update_contrast(event):
                            well_layer.contrast_limits = main_layer.contrast_limits
                        return update_contrast
                    mini_contrast = mini_contrast_callback(main_layer, well_layer)
                    main_layer.events.contrast_limits.connect(mini_contrast)
                self.contrast_callbacks[layer_name] = mini_contrast_callback
            else:
                logging.info(f'No data for {layer_name} at well {well}')
        layout.addWidget(splitter)

        self.feature_widget = self._display_feature_key_widget(well)
        layout.addWidget(self.feature_widget)

        widget.setLayout(layout)
        return widget

    def _next_well(self, event=None):
        """Updates the viewer window with the next well when the right arrow key is pressed

        :param event: key press of right arrow key
        :type event: Event of Napari Qt Event loop
        """

        singlets = self.points_layer.features['singlet']
        singlet_idxs = np.where(singlets)[0]

        if len(singlets) == 0:
            logging.warning('There are no singlets.')
            return
        try:
            current_well = np.where(singlex_idxs == self.current_well)[0][0]
            next_idx = (current_well + 1) % len(singlet_idxs)
        except:
            next_idxs = singlet_idxs[singlet_idxs > self.current_well]
            if len(next_idxs) == 0:
                next_idx = 0
            else:
                next_idx = np.where(singlet_idxs == next_idxs[0])[0][0]
        
        self.current_well = singlet_idxs[next_idx]
        self._update_well_display()
        self._refocus_viewer()
        self._select_current_point()
        self._update_feature_display(self.current_well)
        self._selected_current_pt()
    
    def _previous_well(self, event=None):
        """Updates the viewer window with the previous well when the left arrow key is pressed

        :param event: key press of left arrow key
        :type event: Event of Napari Qt Event loop
        """

        singlets = self.points_layer.features['singlet']
        singlet_idxs = np.where(singlets)[0]

        if len(singlets) == 0:
            logging.warning('There are no singlets.')
            return
        try:
            current_well = np.where(singlex_idxs == self.current_well)[0][0]
            prev_idx = (current_well - 1) % len(singlet_idxs)
        except:
            prev_idxs = singlet_idxs[singlet_idxs < self.current_well]
            if len(prev_idxs) == 0:
                prev_idx = len(singlet_idxs) - 1
            else:
                prev_idx = np.where(singlet_idxs == prev_idxs[-1])[0][0]
        self.current_well = singlet_idxs[prev_idx]
        self._update_well_display()
        self._refocus_viewer()
        self._select_current_point()
        self._update_feature_display(self.current_well)
        self._selected_current_pt()
        
    def _update_well_display(self):
        """Updates the side Classify viewer with a new well
        """

        self.classify_widget.layout().removeWidget(self.well_disp)
        self.well_disp.deleteLater()
        self.well_display_layers.clear()
        self.contrast_callbacks.clear()
        self.well_disp = self._create_well_display(self.current_well, self.well_extract)
        self.classify_widget.layout().insertWidget(0, self.well_disp)

    def _update_feature_display(self, well: int, columns: int=3):
        """Updates the feature classifications with key press actions

        :param well: the well id, currently 0 indexed
        :type well: int

        :param columns: number of columns to list key bindings
        :type columns: int
        """

        layout = self.feature_widget.layout()

        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
            else:
                del item
        row =  0
        layout.addWidget(QLabel('Feature'), row, 0)
        layout.addWidget(QLabel('Value'), row, 1)
        layout.addWidget(QLabel('Key Binding'), row, 2)

        row += 1
        features = self.points_layer.features
        key_map = {feature: key for key, feature in self.key_feature_map.items()}
        
        if features is not None:
            for feature_name, values in features.items():
                feature_value = values[well]
                key_binding = key_map.get(feature_name, " ")
                value_label = QLabel(str(feature_value))
                if isinstance(feature_value, (bool, np.bool_)):
                    color = "green" if feature_value else "red"
                else:
                    color = "white"
                value_label.setStyleSheet(f"color: {color}; font-weight: bold;")
                layout.addWidget(QLabel(feature_name), row, 0)
                layout.addWidget(value_label, row, 1)
                layout.addWidget(QLabel(key_binding), row, 2)

                row += 1
        self.feature_widget.update()
        self.feature_widget.repaint()

    def _select_current_point(self):
        """Select the point for the well currently in view for classification
        """

        if self.points_layer is not None:
            self.points_layer.selected_data = {self.current_well}
            self.refresh()

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

class FishFinderWidget(QWidget):
    """Widget to setup the fish finding algorithm and run it to determine
    the well locations with fish
    """

    def __init__(self, viewer, find_fish_callback):
        super().__init__()
        self.viewer = viewer
        self.find_fish_callback = find_fish_callback
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.layer_label = QLabel('Layer for fish detection')
        self.layer_combo = QComboBox()
        layout.addWidget(self.layer_label)
        layout.addWidget(self.layer_combo)
        self.sigma_label = QLabel('Sigma')
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setMinimum(0.1)
        self.sigma_spin.setMaximum(10)
        self.sigma_spin.setSingleStep(0.1)
        self.sigma_spin.setValue(0.25)
        layout.addWidget(self.sigma_label)
        layout.addWidget(self.sigma_spin)
        self.run_button = QPushButton('Find Fish')
        self.run_button.clicked.connect(self.run_find_fish)
        layout.addWidget(self.run_button)
        self.update_layers()

    def update_layers(self):
        """Create list of layers for the dropdown from the napari
        list of layers and include the sum of the layers
        """

        layer_names = [layer.name for layer in self.viewer.layers if isinstance(layer, Image)]
        layer_names.append('sum')
        self.layer_combo.addItems(layer_names)

    def run_find_fish(self):
        """Callback for the fish finding widget
        """

        layer_name = self.layer_combo.currentText()
        if layer_name == "sum":
            layer_name = None
        sigma = self.sigma_spin.value()
        self.find_fish_callback(layer_name, sigma)


class ContrastWidget(QWidget):
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.layer_controls = {}
        self.contrast_settings = {}
        self.init_ui()
        self.viewer.layers.events.inserted.connect(self.on_layer_inserted)
        self.viewer.layers.events.removed.connect(self.on_layer_removed)

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self.add_layer_control(layer)

    def add_layer_control(self, layer, setpoint=30.0, scale=100.0):
        layer_layout = QHBoxLayout()
        label = QLabel(f'{layer.name} sigma:')
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(0.01*scale))
        slider.setMaximum(int(50*scale))
        slider.setSingleStep(1)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(int(10*scale))

        n_sigma = self.contrast_settings.get(layer.name, setpoint)
        slider.setValue(int(n_sigma * scale))
        value_label = QLabel(f'{n_sigma:.1f}')

        slider.valueChanged.connect(lambda value, l=layer, vl=value_label: self.on_slider_change(l, value, vl, scale))

        layer_layout.addWidget(label)
        layer_layout.addWidget(slider)
        layer_layout.addWidget(value_label)
        self.layout.addLayout(layer_layout)
        self.layer_controls[layer] = (slider, value_label)
        self.update_contrast(layer, slider.value(), scale)

    def remove_layer_control(self, layer):
        if layer in self.layer_controls:
            slider, value_label = self.layer_controls.pop(layer)
            for i in reversed(range(self.layout.count())):
                item = self.layout.itemAt(i)
                if item.layout():
                    layout = item.layout()
                    widget = layout.itemAt(0).widget()
                    if widget and widget.text().startswith(layer.name):
                        while layout.count():
                            child = layout.takeAt(0)
                            if child.widget():
                                child.widget().deleteLater()
                        self.layout.removeItem(item)
                        break
            self.contrast_settings.pop(layer.name, None)
    
    def on_slider_change(self, layer, value, value_label, scale):
        n_sigma = value / scale
        value_label.setText(f'{n_sigma:.1f}')
        self.update_contrast(layer, value, scale)

    def update_contrast(self, layer, slider_value, scale):
        n_sigma = slider_value / scale
        data = layer.data
        mean = np.mean(data)
        std = np.std(data)
        lower = mean - n_sigma * std
        upper = mean + n_sigma * std
        layer.contrast_limits = (lower, upper)
        self.contrast_settings[layer.name] = n_sigma    

    def on_layer_inserted(self, event):
        layer = event.value
        if isinstance(layer, napari.layers.Image):
            self.add_layer_control(layer)

    def on_layer_removed(self, event):
        layer = event.value
        self.remove_layer_control(layer)


if __name__ == '__main__':
    Classify()