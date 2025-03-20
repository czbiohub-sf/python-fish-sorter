import csv
import json
import logging
import numpy as np
import os
import pandas as pd
import sys
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import List, Optional, Tuple

from fish_sorter.hardware.picking_pipette import PickingPipette
from fish_sorter.hardware.imaging_plate import ImagingPlate
from fish_sorter.hardware.dispense_plate import DispensePlate

class Pick():
    """Loads files of classifications and pick parameters, iterates through pick parameters,
    and coordiates all hardware operations to pick from the source to the destination locations
    It uses the PickingPipette class and the Mapping class
    """

    def __init__(self, cfg_dir, pick_dir, prefix, offset, mmc, mda, img_array, dp_array):
        """Loads the files for classification and initializes PickingPipette class
        
        :param cfg_dir: parent path directory for all of the config files
        :type cfg_dir: path
        :param pick_dir: experiment directory for classification and pick files
        :type pick_dir: str
        :param prefix: prefix name details
        :type prefix: str
        :param offset: offset value from center points for picking
        :type offset: float
        :param mmc: pymmcore-plus core
        :type mmc: pymmcore-plus  core instance
        :param mda: pymmcore-plus multidimensial acquisition engine
        :type mda: pymmcore-plus mda instance
        :param img_array: path to image plate array in config folder
        :type: path
        :param dp_file: path to dispense plate array in config folder
        :type: path

        :raises FileNotFoundError: loggings critical if any of the files are not found
        """

        logging.info(f'cfg dir {cfg_dir}')
        logging.info('Initializing Picking Pipette hardware controller')
        dplate_array = cfg_dir / 'arrays' / dp_array
        try:
            self.pp = PickingPipette(cfg_dir, mmc, dplate_array)
        except Exception as e:
            logging.info("Could not initialize and connect hardware controller")
        
        self.pick_dir = pick_dir
        self.prefix = prefix
        self.class_file = None
        self.pick_param_file = None

        array = cfg_dir / 'arrays' / img_array
        logging.info(f'Imgaing array file path {array}')

        self.iplate = ImagingPlate(mmc, mda, array)
        
        self.matches = None
        self.pick_offset = offset

    def connect_hardware(self):
        """Connects to hardware
        """

        self.pp.connect(env='prod')

    def disconnect_hardware(self):
        """Disconnects from hardware
        """

        self.pp.disconnect()

    def check_calib(self, calibrated: bool=False, pick: bool=True, well: Optional[str]=None):
        """Checks for calibration of pipette tip height

        :param calibrated: check if pipette tip is calibrated
        :type calibrated: bool
        :param pick: pick location is True
        :type pick: bool
        :param well: well ID
        :type well: str
        """

        if not calibrated:
            if pick:
                logging.info('Calbrating Pick Height')
                self.pp.move_for_calib(pick)
            else:
                logging.info('Calibrating Dispense Height')
                dest_loc = self.get_dest_xy(well)
                self.pp.move_for_calib(pick, dest_loc)
        else:
            logging.info('Already calibrated')
        
    def set_calib(self, pick: bool=True):
        """Sets pipette calibration once user acknowledges location
                
        :param pick: pick location is True
        :type pick: bool
        """
            
        self.pp.set_calib(pick)

    def get_classified(self):
        """Opens classification and pick parameter files
        """

        logging.info('Load classification and picking files')
        for filename in os.listdir(self.pick_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.pick_dir, filename)
                try:
                    if 'classifications.csv' in filename:
                        self.class_file = pd.read_csv(file_path)
                        logging.info('Loaded {}'.format(filename))
                    elif 'pickable.csv' in filename:
                        self.pick_param_file = pd.read_csv(file_path)
                        logging.info('Loaded {}'.format(filename))
                except FileNotFoundError:
                    logging.critical("File not found")

        picked_filename = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + self.prefix + '_picked.csv'
        self.picked_file = os.path.normpath(os.path.join(self.pick_dir, picked_filename))
    
    def get_dest_xy(self, well: str) -> Tuple[float, float]:
        """"Uses the Mapping class to get the well x, y coordinates from the well ID

        :param well: well ID
        :type well: str

        :return: The coordinates for the specific well ID location: (x, y)
        :rtype: Tuple[float, float]
        """

        # MK TODO ensure array formats are compatible
        return self.pp.dplate.get_abs_um_from_well_name(well)

    def pick_me(self):
        """Performs all actions to pick from the source plate to the destination plate using
        the match list created by match_pick
        """

        logging.info('Begin iterating through pick list')
        self.matches.drop(columns=['lHead']).head(0).to_csv(self.picked_file, index=False)
        self.pp.move_pipette('clearance')
        self.pp.dest_home()
        
        for match in self.matches.index:
            if self.matches['lHead'][match]:
                offset = - self.offset
            else:
                offset = self.offset
            
            self.iplate.go_to_well(self.matches['slotName'][match], offset)
            self.pp.move_pipette('pick')
            sleep(1)
            self.pp.draw()
            self.pp.move_pipette('clearance')
            
            self.pp.dplate.go_to_well(self.matches['dispenseWell'][match])
            self.pp.move_pipette('dispense')
            self.pp.expel()
            sleep(1)
            self.pp.expel()
            self.pp.move_pipette('clearance')
            self.pp.dest_home()
            logging.info('Picked fish in {} to {}'.format(self.matches['slotName'][match], self.matches['dispenseWell'][match]))
            pd.DataFrame([self.matches.drop(columns=['lHead']).iloc[match].values], columns=self.matches.drop(columns=['lHead']).columns)\
                .to_csv(self.picked_file, mode='a', header=False, index=False)

        #TODO how to more elegantly handle lHead, rightHead, none, etc
        #call to mapping?
        #call mapping for the dest plate at init?      
        #Better handle 'lHead' column in csv??? or rather abstract at some point to also include embryos
        #Use pick_type_config.json better to determine what columns are needed


    def match_pick(self):
        """Matches the desired pick parameters to the classification
        """

        class_drop = self.class_file.reset_index(drop=True)
        pick_param_drop = self.pick_param_file.reset_index(drop=True)
        matching = class_drop.columns.intersection(pick_param_drop.columns).difference(['slotName', 'dispenseWell'])
        merge = pd.merge(class_drop, pick_param_drop, on=list(matching), how='inner')
        merge_sorted = pd.merge(pick_param_file[['dispenseWell']], merged, on='dispenseWell', how='inner')
        self.matches = pd.DataFrame({'slotName': merge_sorted['slotName'], 'dispenseWell': merge_sorted['dispenseWell'], 'lHead': merge_sorted['lHead']})
        logging.info('Created pick list')

        #TODO future feature: save time and snake through position list?