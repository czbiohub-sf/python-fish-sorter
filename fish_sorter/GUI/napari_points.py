import json
import logging
import napari
import numpy as np
import os
import sys
from pathlib import Path
from skimage import data
from typing import List, Optional

class NapariPts():
    """Add points layer of the well locations to the image mosaic in napari.
    """
    
    def __init__(self, parent_dir, total_wells, viewer=None):
        """Load well array file of well location points and load fish features. 
        
        :param parent_dir: parent directory for array config files
        :type parent_dir: str
        :param total_wells: descriptor for the agarose array
        :type total_wells: int
        :param viewer: Optional napari viewer to use, otherwise a new one is created
        :type viewer: napari.Viewer, optional
        :raises FileNotFoundError: loggings critical if the array config file not found
        """

        parent_dir = Path(parent_dir) / "configs/arrays"
        self.array_data = {}
        self.total_wells = total_wells
        num_wells = str(self.total_wells)

        for filename in os.listdir(parent_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(parent_dir, filename)
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
        
        if viewer is None:
            viewer = napari.Viewer()
        self._feat()
        points_layer = self.load_points(viewer)

        @viewer.bind_key('f')
        def toggle_point_annotation(event):
            selected_points = list(points_layer.selected_data)
            if len(selected_points) > 0:
                fish_point = points_layer.features['fish']
                fish_point[selected_points] = ~fish_point[selected_points]
                points_layer.features['fish'] = fish_point
                points_layer.refresh_colors(update_color_mapping=False)
        napari.run()

    def _feat(self):
        """Loads the feature list
        """
            
        self.features = {
            'Well': np.array(self.array_data[self.var_name]['wells']['well_names']),
            'fish': np.full(self.total_wells, False),
            'deformed': np.full(self.total_wells, False),
            'wrong_o': np.full(self.total_wells, False),
            'unknown': np.full(self.total_wells, False),
            'multiple': np.full(self.total_wells, False),
            'singlet': np.full(self.total_wells, False),
            'gEye': np.full(self.total_wells, False),
            'gHeart': np.full(self.total_wells, False),
            'gBody': np.full(self.total_wells, False),
            'gSpeckles': np.full(self.total_wells, False),
            'rEye': np.full(self.total_wells, False),
            'rHeart': np.full(self.total_wells, False),
            'rSpeckles': np.full(self.total_wells, False),
            'lHead': np.full(self.total_wells, False),
        }

    def load_points(self, viewer):
        """Load the points layer into the napari viewer
        """

        well_coords = self.array_data[self.var_name]['wells']['well_coordinates']
        points_coords = np.array(well_coords).reshape(-1,2)

        #TODO convert real coordinates to image coordinates
        #points = STAGE_2_IMG(points_coords)

        face_color_cycle = ['white', 'red']

        #TODO change points_coords to points once conversion in place
        points_layer = viewer.add_points(
            points_coords,
            features = self.features,
            size = 100,
            face_color = 'fish',
            face_color_cycle = face_color_cycle
        )
        return points_layer

if __name__ == '__main__':
    NapariPts('/Users/diane.wiener/Documents/GitHub/python-fish-sorter/fish_sorter', 595)