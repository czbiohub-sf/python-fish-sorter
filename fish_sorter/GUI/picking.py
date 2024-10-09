import json
import logging
import numpy as np
import os
import sys
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import List, Optional

from fish_sorter.hardware.picking_pipette import PickingPipette
from fish_sorter.helpers.mapping import Mapping

class Pick():
    """Loads files of classifications and pick parameters, iterates through pick parameters,
    and coordiates all hardware operations to pick from the source to the destination locations
    It uses the PickingPipette class and the Mapping class
    """

    def __init__(self, pick_dir, dest_array, source_array):
        """Loads the files for classification and initializes PickingPipette class
        
        :param pick_dir: directory for classification and pick files
        :type pick_dir: str
        :param dest_array: name of the array config for the destination plate
        :param dest_array: str
        :param source_array: name of the array config for the source plate
        :param source_array: str
        :raises FileNotFoundError: loggings critical if any of the files are not found
        """

        logging.info('Initializing Picking Pipette hardware controller')
        cfg_dir = Path().absolute().parent
        try:
            self.pp = PickingPipette(cfg_dir)
        except Exception as e:
            logging.info("Could not initialize and connect hardware controller")
        
        logging.info('Load classification and picking files')
        self.class_file = None
        self.pick_param_file = None
        self.picked_file = None
        self.pick_dir = pick_dir



    def connect_hardware(self):
        """Connects to hardware
        """

        self.pp.connect(env='prod')

    def disconnect_hardware(self):
        """Disconnects from hardware
        """

        self.pp.disconnect()

    def check_calib(self, calibrated: bool=False, pick: bool=True, well: Optional [str]):
        """Checks for calibration of pipette tip height

        :param calibrated: check if pipette tip is calibrated
        :type calibrated: bool
        :param pick: pick location is True
        :type pick: bool
        :param well: well ID
        :type well: str
        """

        if calibrated:
            if pick:
                logging.info('Calbrating Pick Height')
                self.pp.move_for_calib(pick)
            else:
                logging.info('Calibrating Dispense Height')
                dest_loc = self.get_dest_xy(well)
                self.pp.move_for_calib(pick, dest_loc)
        else:
            logging.info('Already calibrated')
        
    def set_calib(self, pick: bool=True)
        """Sets pipette calibration once user acknowledges location
                
        :param pick: pick location is True
        :type pick: bool
        """
            
        self.pp.set_calib(pick)

    def get_dest_xy(self, well: str) -> Tuple[float, float]:
        """"Uses the Mapping class to get the well x, y coordinates from the well ID

        :param well: well ID
        :type well: str
        :return: The coordinates for the specific well ID location: (x, y)
        :rtype: Tuple[float, float]
        """

        #TODO use helpers.mapping to go between well ID and x, y coords
        return (0.0, 0.0)

    def move_to_pick(self, well: str, offset: Tuple[float, float]=(0.0, 0.0))
        """Moves the microscope stage to the source well for picking
        The offset is for any needed offset from the well center coordinates
        to the pick location
        
        :param well: well ID
        :type well: str
        :param offset: offset from the well center coordinates to the pick location
        :type offset: Tuple[float, float]
        """
    
        #TODO properly call / initialize self.mapping
        self.mapping.go_to_well(well, offset)



        #TODO use helpers.mapping load_wells and go_to_well for control of the microscope stage

    def pick_me(self, src_well: str, dest_well: str, offset: Optional [float]):
        """Performs all actions to pick from the source plate to the destination plate

        :param src_well: well ID of the source plate on the microscope stage
        :type src_well: str
        :param dest_well: well ID of the source plate on the microscope stage
        :type dest_well: str

        