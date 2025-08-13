import concurrent.futures
import json
import logging
import napari
import numpy as np
import os
import pandas as pd
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from napari.components.viewer_model import ViewerModel
from napari.layers import Image
from napari.qt import QtViewer
from napari.utils.colormaps import Colormap
from pathlib import Path
from pymmcore_plus import CMMCorePlus
from PyQt6.QtCore import (
    QObject,
    QSize,
    Qt,
    QThread,
    QTimer
)
from qtpy.QtGui import QColor, QScreen
from qtpy.QtWidgets import (
    QApplication,
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

class Classify(QObject):
    """Add points layer of the well locations to the image mosaic in napari.
    """
    
    def __init__(self, cfg_dir, pick_type, prefix, expt_dir, iplate, viewer=None):
        """Load pymmcore-plus core, acquisition engine and napari viewer, and load classification features
        
        :param cfg_dir: parent path directory for all of the config files
        :type cfg_dir: Path
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
        super().__init__()
        CMMCorePlus.instance()

        self.iplate = iplate
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

        self.contrast_callbacks = {}

        # Independent contrast control
        # Because users are more comfortable using the native napari layer controls contrast limits, not implemented
        # self.contrast_widget = ContrastWidget(self.viewer)
        # self.viewer.window.add_dock_widget(self.contrast_widget, name= 'Contrast', area='left')
    
        pts = self._points()
        self.pts = np.array(pts)
        self.points_layer = self.load_points(self.pts.copy())
        self.points_layer.mode = 'select'
        self.points_layer.events.set_data.connect(self.refresh)
        self.points_layer.events.highlight.connect(self._selected_pt)

        
        self._key_binding()
        for key, feature in self.key_feature_map.items():
            self.viewer.bind_key(key, overwrite=True)(self._toggle_feature(feature)) # Note: set to overwrite standard shortcuts in napari
            self.points_layer.bind_key(key, overwrite=True)(self._toggle_feature(feature)) # Note: set to overwrite points layer specific shortcuts
        
        self.navigate_all = True
        
        # Prevents points from being deleted
        self.viewer.bind_key('Backspace', self._blank, overwrite=True)
        self.viewer.bind_key('Delete', self._blank, overwrite=True)
        self.points_layer.bind_key('Backspace', self._blank, overwrite=True)
        self.points_layer.bind_key('Delete', self._blank, overwrite=True)

        
        self.well_viewers = {}
        self.well_display_layers = {}
        self.viewer_containers = {}
        self.feature_labels = {}
        self.current_well = 0
        self.counter = None

        self._create_classify()
        self.extract_fish(self.pts)
        self._find_fish_widget(self.pts)
        self._start_async_extraction()

        self.prefix = prefix
        self.expt_dir = expt_dir
        self.save_data()

    def _blank(self, event):
        """Empty function used to overwrite hotkeys
        """

        pass

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

        well_coords = self.iplate.wells["actual_px"]
        points_coords = np.array(well_coords).reshape(-1,2)
        points_coords = points_coords[:, ::-1]

        return points_coords

    def _selected_pt(self, event):
        """Callback when a point is selected
        """

        self.refresh()
        selected_data = self.points_layer.selected_data
        if selected_data:
            selected_idxs = list(selected_data)
            selected_idx = selected_idxs[0]
            self.current_well = selected_idx
            self._well_disp()
            self._update_feature_display(self.current_well)

    def _selected_current_pt(self):
        """Select the point for the current well in the points layer
        """
        
        if self.points_layer is not None:
            self.points_layer.selected_data = {self.current_well}
    
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
                feature_values.loc[selected_points] = ~feature_values.loc[selected_points]
                self.points_layer.features.loc[:, feature_name] = feature_values

                if feature_name in self.deselect_rules and feature_values[selected_points].iloc[0]:
                    for feat in self.deselect_rules[feature_name]:
                        feature_values = self.points_layer.features[feat]
                        feature_values.loc[selected_points] = False
                        self.points_layer.features[feat] = feature_values
                    self.points_layer.refresh_colors(update_color_mapping=False)
                    self._update_feature_display(self.current_well)
                    self._update_counter()
                    self._well_disp()
                    self.points_layer.mode = 'select'
                    return
                                            
            self.points_layer.refresh_colors(update_color_mapping=False)
            self._update_feature_display(self.current_well)
            self.points_layer.mode = 'select'

        return _toggle_annotation

    def save_data(self):
        """Saves the classification once the user pushes the button
        """

        self.class_btn = QPushButton("Save Classification")

        self.save_widget = QWidget()
        layout = QGridLayout(self.save_widget)  
        layout.addWidget(self.class_btn, 1, 0)
        self.viewer.window.add_dock_widget(self.save_widget, name= 'Save', area='left', tabify=True)

        if hasattr(self, 'fish_widget'):
            QTimer.singleShot(50, lambda: self.viewer.window._dock_widgets[self.fish_widget_name].raise_())

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
            classified = os.path.normpath(os.path.join(self.expt_dir, file_name))
            class_df.to_csv(classified, index=False)
            logging.info(f'Classification saved as {classified}')
            
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

        if not hasattr(self, "_refreshing"):
            self._refreshing = False

        if self._refreshing:
            return

        self._refreshing = True

        try:
            delta = self.points_layer.data - self.pts
            max_diff = np.abs(delta).max()
            if max_diff > 1e-6:
                self.points_layer.data = self.pts.copy()
            self.points_layer.mode = 'select'
            self.points_layer.refresh_colors(update_color_mapping=False)
            if self.feature_widget is not None:
                self._update_feature_display(self.current_well)
        finally:
            self._refreshing = False

    def extract_fish(self, points):
        """Finds the locations of positive wells

        :param points: x, y coords for center location defining the well locations in the points layer
        :type points: numpy points
        """
        
        self._well_mask()
        self.well_extract = self._extract_wells(points)
    
    def _well_mask(self, padding: int=100):
        """Create a mask of the well shape

        :param padding: extra pixels from the edge around the well shape to include in the mask
        :type padding: int
        """

        w_h = np.array([self.iplate.wells['array_design']['slot_length'], self.iplate.wells['array_design']['slot_width']])
        origin = self.iplate.px_to_rel_um(np.array([0, 0]))
        abs_w_h = w_h + origin
        convert_w_h = self.iplate.rel_um_to_px(abs_w_h)
        width = int(round(convert_w_h[0]))
        height = int(round(convert_w_h[1]))

        padded_width = width + 2 * padding
        padded_height = height + 2 * padding
        self.mask = np.zeros((padded_height, padded_width), dtype=bool)
        
        if self.iplate.wells["array_design"]["well_shape"] == "rectangular_array":
            start_row, start_col = padding, padding
            rr, cc = draw.rectangle(start=(start_row, start_col), extent=(height, width), shape=self.mask.shape)
        else:
            center = (padded_height // 2, padded_width // 2)
            radius = min(height, width) // 2  
            rr, cc = draw.disk(center, radius, shape=self.mask.shape)

        self.mask[rr, cc] = True
    
    def _start_async_extraction(self):
        """Thread for well extraction
        """

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future = self.executor.submit(self._extract_wells_threaded)
        self.future.add_done_callback(self._extract_done)

    def _extract_done(self, future):
        """Callback to main thread after background well extraction is done
        """

        try:
            self.well_extract = future.result()
            self._well_disp()
        except Exception as e:
            logging.error(f'Well extraction failed: {e}')

    def _extract_wells_threaded(self):
        """Background thread for well extractraion
        """

        points = self._points()
        self._well_mask()
        return self._extract_wells(points, img_flag=True, parallel=True)
    
    def _extract_wells(self, points, img_flag: bool=True, mask_layer: str=None, parallel: bool=False, sigma: float=0.25) -> dict:
        """Cuts a well centered around the points in the points layer of the image
        of the size defined in the array and displays the image layer

        :param points: array of (y, x) coordinates (row, col) for well centers
        :type points: numpy points
        :param img_flag: option whether to use the image layers or to create the binary image mask 
            automatic fish detection, True to use the image layers, False to use the binary mask
        :type img_flag: bool 
        :param mask_layer: layer to use to find fish, default None will use all layers
        :type layer_name: str
        :param parallel: whether to use parallel processing
        :type parallel: bool
        :param sigma: number of standard deviations from mean for thresholding the mask
        :type sigma: float

        :return: layer name, extracted region for each layer for each point
        :rtype: dict 
        """

        mask_height, mask_width = self.mask.shape
        half_mask_height, half_mask_width = mask_height // 2, mask_width // 2

        if img_flag:
            image_layers = [
                {'data': layer.data, 'name': layer.name} 
                for layer in self.viewer.layers 
                if isinstance(layer, napari.layers.Image)
            ]
        else:
            if mask_layer:
                raw_layers = [layer for layer in self.viewer.layers if layer.name == mask_layer]
                raw_data = raw_layers[0].data
                layer_name = raw_layers[0].name
            else:
                raw_layers = [
                    layer for layer in self.viewer.layers 
                    if isinstance(layer, napari.layers.Image) and layer.name != 'BF'
                ]
                raw_data = np.zeros_like(raw_layers[0].data, dtype=np.uint16)
                for layer in raw_layers:
                    logging.info(f'layer name {layer}')
                    raw_data += layer.data
                layer_name = 'sum'
            mask_mean = raw_data.mean()
            mask_std = raw_data.std()
            thresh = mask_mean + (sigma * mask_std)
            binary_mask = raw_data > thresh
            image_layers = [{'data': binary_mask.astype(np.uint8), 'name': layer_name}]
            logging.info(f'image layers {layer_name}')
    
        def _extract_point(point):
            """Creates extracted image around points

            :param point: (y, x) coordinate (row, col) for well center
            :type point: float

            :return region_by_layer, extracted region for each layer for each point
            :rtype: dict
            """

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
   
            return region_by_layer

        if parallel:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                results = list(pool.map(_extract_point, points))
            return results
        else:
            return [_extract_point(p) for p in points]
    
    def find_fish(self, points, layer_name=None, sigma=0.25):
        """Automatically detects fish and fish orientation.

        :param points: x, y coordinates for center location points defining the well locations
        :type points: numpy points

        :param layer_name: layer to use to find fish, default None will use all layers
        :type layer_name: str

        :param sigma: threshold value to compare well intensities to background
        :type simga: float
        """

        img_data = self._extract_wells(points, img_flag=False, mask_layer=layer_name, parallel=True, sigma=sigma)
        wells_data = [list(region.values())[0].mean() for region in img_data]
        well_mean = np.mean(wells_data)
        if layer_name == 'BF':
            well_class = [region_mean < well_mean for region_mean in wells_data]
        else:
            well_class = [region_mean > well_mean for region_mean in wells_data]

        self.navigate_all = False
        self._update_found_fish(well_class)

    def _reset_fish(self):
        """Resets the found fish to the empty state and deselects all classifications
        """

        self.navigate_all = True
        self._feat()
        for feat in self.features:
            self.points_layer.features[feat] = self.features[feat]
        self.refresh()
        self._update_counter()
        self._update_nav_mode()
    
    def _update_found_fish(self, wells: List[int]):
        """Updates the feature classification for the found fish and orientation

        :param wells: well locations with fish
        :type wells: List[int]
        """

        singlets = self.points_layer.features['singlet']
        singlets.loc[wells] = True
        singlets_idxs = np.where(singlets)[0]
        self.points_layer.features.loc[:, 'singlet'] = singlets
        for feat in self.deselect_rules['singlet']:
            self.points_layer.features.loc[singlets, feat] = False                                 
        self.refresh()
        self.points_layer.mode = 'select'
        self.current_wells = singlets_idxs[0]

        if self.picking == 'larvae':
            logging.info('Determining fish orientation for picking larvae')
            self.find_orientation()
        
        self._update_counter()
        self._update_nav_mode()

    def find_orientation(self):
        """Determines orientation to select side of well for picking
        """

        single = [i for i, val in enumerate(self.points_layer.features['singlet']) if val]

        def comp_orientation(fish):
            """Helper to parallelize orienataion computation
            """

            if fish >= len(self.well_extract):
                logging.info(f'Skipping fish {fish}: out of bounds for well extraction')
                return fish, None
            well_data = self.well_extract[fish]
            channels = list(well_data.values())
            well_total = np.zeros_like(channels[0], dtype=np.float32)
            for channel in channels:
                well_total += channel
            well_total /= len(channels)

            height, width = well_total.shape
            half_width = width // 2
            left = well_total[:height, :half_width]
            right = well_total[:height, half_width:width]

            left_sum = np.sum(left)
            right_sum = np.sum(right)

            return fish, left_sum >= right_sum

        with concurrent.futures.ThreadPoolExecutor() as pool:
            results = list(pool.map(comp_orientation, single))
        orientation = {fish: head for fish, head in results if head is not None}
        self._update_orientation(orientation)
        # self.plot_crop() #Toggle for crop debugging
    
    def plot_crop(self):
        """Plots the cropped images for debugging
        """

        single = [i for i, val in enumerate(self.points_layer.features['singlet']) if val]

        for idx in single:
            if idx >= len(self.well_extract):
                continue
            well_data = self.well_extract[idx]
            channels = list(well_data.values())

            if not channels:
                continue

            avg_img = np.mean(np.stack(channels), axis=0)
            height, width = avg_img.shape
            half_width = width //2
            left = avg_img[:, :half_width]
            right = avg_img[:, half_width:]

            fig, axs = plt.subplots(1, 3, figsize=(12,4))
            fig.suptitle(f'Well #{idx}')

            axs[0].imshow(avg_img, cmap='gray')
            axs[0].set_title('Avg well image')
            axs[1].imshow(left, cmap='grey')
            axs[1].set_title('Left half')
            axs[2].imshow(right, cmap='grey')
            axs[2].set_title('Right half')

            for ax in axs:
                ax.axis('off')
            plt.show()
    
    def _update_orientation(self, orientation=List[int]):
        """Updates the feature classification for the found fish and orientation

        :param orientation: well location where fish orientation is True
        :type orientation: List[int]
        """

        for idx, head in orientation.items():
            self.points_layer.features.loc[idx, 'lHead'] = head
        self.refresh()
        self.points_layer.mode = 'select'
        logging.info('Ready for individual fish classification')

    def _find_fish_widget(self, points):
        """Call back widget to use to detect wells with fish

        :param points: x, y coordinates for center location points defining the well locations
        :type points: numpy points
        """

        def find_fish_callback(layer_name: str, sigma: float):
            self.find_fish(points, layer_name, sigma)
        self.fish_widget = FishFinderWidget(self.viewer, find_fish_callback, self)
        self.fish_widget_name = 'Finding Nemo'
        self.viewer.window.add_dock_widget(self.fish_widget, name=self.fish_widget_name, area='left')
        minmax_wgt = self.viewer.window._dock_widgets['MinMax']
        self.viewer.window._qt_window.tabifyDockWidget(self.viewer.window._dock_widgets[self.fish_widget_name], minmax_wgt)
    
    def _create_classify(self):
        """Classification side widget with key bindings and well viewer windows

        :return: A QWidget with key binding map
        :rtype: QWidget
        """

        self.classify_widget = QWidget()
        self.classify_layout = QVBoxLayout(self.classify_widget)

        header_widget = QWidget()
        header_layout =QHBoxLayout()
        header_widget.setLayout(header_layout)
        self.counter = QLabel('')
        self.counter.setStyleSheet('font-size: 16pt; font-weight: bold; color: #6a617a;')
        header_layout.addWidget(self.counter)
        self.nav_mode_label = QLabel()
        self.nav_mode_label.setStyleSheet('font-size: 10 pt; color: #6a617a;')
        header_layout.addWidget(self.nav_mode_label)
        header_layout.addStretch()
        self.classify_layout.addWidget(header_widget)

        self.well_disp_container = QWidget()
        self.well_disp_layout = QVBoxLayout(self.well_disp_container) 
        self.classify_layout.addWidget(self.well_disp_container)

        self.feature_widget = QWidget()
        feature_layout = QGridLayout()
        feature_layout.addWidget(QLabel('Feature'), 0, 0)
        feature_layout.addWidget(QLabel('Classification'), 0, 1)
        feature_layout.addWidget(QLabel('Key Binding'), 0, 2)

        self.feature_labels = {}
        row = 1
        key_map = {feature: key for key, feature in self.key_feature_map.items()}

        for feature_name in self.points_layer.features.columns:
            value_label = QLabel()
            key_label = QLabel(key_map.get(feature_name, " "))

            feature_layout.addWidget(QLabel(feature_name), row, 0)
            feature_layout.addWidget(value_label, row, 1)
            feature_layout.addWidget(key_label, row, 2)

            self.feature_labels[feature_name] = (value_label, key_label)
            row += 1
        self.feature_widget.setLayout(feature_layout)
        self.classify_layout.addWidget(self.feature_widget)

        self.viewer.window.add_dock_widget(self.classify_widget, name= 'Classification', area='right', tabify = True)
        self.viewer.bind_key("Right", self._next_well, overwrite=True)
        self.viewer.bind_key("Left", self._previous_well, overwrite=True)
        self.viewer.bind_key("T", self._toggle_navigation, overwrite=True)

        self._update_counter()
        self._update_nav_mode()
    
    def _create_viewer(self, layer_name, masked_region):
        """Make side viewer on main thread

        :param layer_name: name of layer
        :type layer_name: str
        :param masked_region: well region for display
        :type masked_region: numpy array
        """

        def _create():

            viewer_model = ViewerModel(title = layer_name)
            viewer = QtViewerWrap(self.viewer, viewer_model)
            color = self._get_color(layer_name)
            well_layer = viewer_model.add_image(
                masked_region,
                colormap=color,
                contrast_limits=(masked_region.min(), masked_region.max()),
                name=layer_name,
            )
            main_layer = self._get_main_layer(layer_name)
            if main_layer is not None:
                def update_contrast(event, m=main_layer, w=well_layer):
                    w.contrast_limits = m.contrast_limits
                self.contrast_callbacks[layer_name] = update_contrast
                main_layer.events.contrast_limits.connect(update_contrast)    
            viewer_model.camera.zoom = 0.95
            self.well_viewers[layer_name] = viewer
            self.well_display_layers[layer_name] = well_layer
            
            container = QWidget()
            container_layout = QVBoxLayout()
            container_layout.addWidget(QLabel(layer_name))
            container_layout.addWidget(viewer)
            container.setLayout(container_layout)
            self.viewer_containers[layer_name] = container
            viewer.setParent(container)
            self.well_disp_layout.addWidget(container)
        
        QTimer.singleShot(0, _create)

    def _get_color(self, layer):
        """Helper to select the color from the layer

        :param layer: layer name
        :type layer: str
        """

        if layer == 'DAPI':
            return Colormap([[0, 0, 0], [0.16, 0.82, 0.79]], name='DAPI-cyan')
        elif layer == 'GFP':
            return Colormap([[0, 0, 0], [0, 1, 0]], name='GFP-green')
        elif layer == 'TXR':
            return Colormap([[0, 0, 0], [1, 0.25, 0]], name='tiger-orange')
        elif layer == 'CIT':
            return Colormap([[0, 0, 0], [1, 1, 0]], name='CIT-yellow')
        elif layer == 'CY5':
            return Colormap([[0, 0, 0], [0.93, 0.13, 0.53]], name='CY5-plasma')
        else:
            return Colormap([[0, 0, 0], [0.5, 0.5, 0.5]], name='gray')
    
    def _next_well(self, event=None):
        """Updates the viewer window with the next well when the right arrow key is pressed

        :param event: key press of right arrow key
        :type event: Event of Napari Qt Event loop
        """

        idxs = np.arange(len(self.points_layer.data)) if self.navigate_all else np.where(self.points_layer.features['singlet'])[0]

        if len(idxs) == 0:
            logging.info(f'No wells were found under the current navigation mode')
            return

        try:
            current_idx = np.where(idxs == self.current_well)[0][0]
            next_idx = (current_idx + 1) % len(idxs)
        except IndexError:
            next_idx = 0
            
        self.current_well = idxs[next_idx]
        self._well_disp()
        self._refocus_viewer()
        self._select_current_point()
        self._update_feature_display(self.current_well)
        self._selected_current_pt()
        self._update_counter()
    
    def _previous_well(self, event=None):
        """Updates the viewer window with the previous well when the left arrow key is pressed

        :param event: key press of left arrow key
        :type event: Event of Napari Qt Event loop
        """

        idxs = np.arange(len(self.points_layer.data)) if self.navigate_all else np.where(self.points_layer.features['singlet'])[0]

        if len(idxs) == 0:
            logging.info(f'No wells were found under the current navigation mode')
            return

        try:
            current_idx = np.where(idxs == self.current_well)[0][0]
            prev_idx = (current_idx - 1 + len(idxs)) % len(idxs)
        except IndexError:
            prev_idx = len(idxs) - 1

        self.current_well = idxs[prev_idx]
        self._well_disp()
        self._refocus_viewer()
        self._select_current_point()
        self._update_feature_display(self.current_well)
        self._selected_current_pt()
        self._update_counter()

    def _toggle_navigation(self, event=None):
        """Changes left right navigation from singlets to all fish

        :param event: key press of T
        :type event: Event of Napari Qt Event loop
        """

        self.navigate_all = not self.navigate_all
        mode = 'All' if self.navigate_all else 'Singlets'
        logging.info(f'Well navigation set to {mode}')
        self._update_counter()
        self._update_nav_mode()

    def _update_counter(self):
        """Updates the fish counter for the user to know which fish they are on and the total
        """

        self._singlet_nav()

        if self.navigate_all:
            total = len(self.points_layer.data)
            pos = self.current_well + 1
            self.counter.setText(f'Well {pos} of {total} (All Wells)')

        else:
        
            singlets = self.points_layer.features['singlet']
            singlet_idxs = np.where(singlets)[0]

            if len(singlet_idxs) == 0:
                self.counter.setText('No Fish')
                return

            try:
                pos = np.where(singlet_idxs == self.current_well)[0][0] + 1

            except IndexError:
                pos = 1
        
            total = len(singlet_idxs)
            self.counter.setText(f'Fish {pos} of {total}')
    
    def _update_nav_mode(self):
        """GUI navigate mode indicator
        """

        mode = 'All Wells' if self.navigate_all else 'Singlets'
        self.nav_mode_label.setText(f'Nav Mode <b>{mode} </b/> &nbsp; (press <b>T</b> to toggle)')

    def _singlet_nav(self):
        """Helper function during singlet navigation mode when a fish is reclassified as not a singlet
        Does not reset the curret well to 0
        """

        if self.navigate_all:
            return

        singlet_idxs = np.where(self.points_layer.features['singlet'])[0]

        if len(singlet_idxs) == 0:
            return

        if self.current_well not in singlet_idxs:
            next_well = singlet_idxs[singlet_idxs > self.current_well]
            self.current_well = next_well[0] if len(next_well) > 0 else singlet_idxs[0]
    
    def _well_disp(self):
        """Force _update_well_display to run on main thread
        """

        QTimer.singleShot(0, self._update_well_display)

    def _update_well_display(self):
        """Updates the side Classify viewer with a new well
        """

        if not hasattr(self, 'well_extract'):
            return

        point_region = self.well_extract[self.current_well]

        for i in reversed(range(self.well_disp_layout.count())):
            item = self.well_disp_layout.takeAt(i)
            widget = item.widget()
            if widget:
                widget.setParent(None)

        for layer_name, masked_region in point_region.items():
            if masked_region is None:
                continue

            if (layer_name in self.well_viewers and 
                layer_name in self.well_display_layers and
                layer_name in self.viewer_containers):

                well_layer = self.well_display_layers[layer_name] 
                viewer = self.well_viewers[layer_name]
                well_layer.data = masked_region
                well_layer.refresh()
                well_layer.events.data()
                
                container = self.viewer_containers[layer_name]
                self.well_disp_layout.addWidget(container)             
            else:
                self._create_viewer(layer_name, masked_region)

        self._update_feature_display(self.current_well)

    def _get_main_layer(self, name):
        return next((layer for layer in self.viewer.layers if layer.name == name), None)
    
    def _update_feature_display(self, well: int):
        """Updates the feature classifications with key press actions

        :param well: the well id, currently 0 indexed
        :type well: int
        """

        for feature_name, (value_label, key_label) in self.feature_labels.items():
            feature_value = self.points_layer.features[feature_name][well]
            value_label.setText(str(feature_value))
            if isinstance(feature_value, (bool, np.bool_)):
                color = "green" if feature_value else "red"
            else:
                color = "white"
            value_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _select_current_point(self):
        """Select the point for the well currently in view for classification
        """

        if self.points_layer is not None:
            self.points_layer.selected_data = {self.current_well}

    def _refocus_viewer(self):
        """Needed so that classification can be done in the main window
        """

        self.viewer.layers.selection.active = self.points_layer
        self.viewer.window._qt_window.setFocus()
       
class QtViewerWrap(QtViewer):
    
    def __init__(self, main_viewer, *args, **kwargs) -> None:
        
        app = QApplication.instance()
        if not app:
            app = QApplication([])

        super().__init__(*args, **kwargs)
        self.main_viewer = main_viewer

        screen = QApplication.primaryScreen().availableGeometry()
        max_width, max_height = screen.width(), screen.height()
        self.setMaximumSize(max_width, max_height)
        self.resize(max_width, max_height)

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

    def __init__(self, viewer, find_fish_callback, classify):
        super().__init__()
        self.viewer = viewer
        self.classify = classify
        self.find_fish_callback = find_fish_callback
        
        layout = QGridLayout()
        self.setLayout(layout)
        self.layer_label = QLabel('Fish detection channel')
        self.layer_combo = QComboBox()
        layout.addWidget(self.layer_label, 1, 0)
        layout.addWidget(self.layer_combo, 1, 1)
        self.sigma_label = QLabel('Sigma')
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setDecimals(4)
        self.sigma_spin.setMinimum(0.0001)
        self.sigma_spin.setMaximum(100)
        self.sigma_spin.setSingleStep(0.0001)
        self.sigma_spin.setValue(1)
        layout.addWidget(self.sigma_label, 2, 0)
        layout.addWidget(self.sigma_spin, 2, 1)
        self.run_button = QPushButton('Find Fish')
        self.run_button.clicked.connect(self.run_find_fish)
        layout.addWidget(self.run_button, 3, 0)
        self.reset_fish = QPushButton('Reset Fish')
        self.reset_fish.clicked.connect(self.classify._reset_fish)
        layout.addWidget(self.reset_fish, 3, 1)
        self.update_layers()

    def update_layers(self):
        """Create list of layers for the dropdown from the napari
        list of layers and include the sum of the layers
        """

        layer_names = [layer.name for layer in self.viewer.layers if isinstance(layer, Image)]
        if len(layer_names) >1 and 'sum' not in layer_names:
            layer_names.append('sum')
        self.layer_combo.clear()
        self.layer_combo.addItems(layer_names)
        if 'sum' in layer_names:
            sum_idx = self.layer_combo.findText('sum')
            self.layer_combo.setCurrentIndex(sum_idx)
        elif layer_names:
            self.layer_combo.setCurrentIndex(0)

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

    def add_layer_control(self, layer, setpoint=20.0, scale=100.0):
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