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
from typing import List, Optional

from fish_sorter.hardware.picking_pipette import PickingPipette
from fish_sorter.helpers.mapping import Mapping

class Pick():
    """Loads files of classifications and pick parameters, iterates through pick parameters,
    and coordiates all hardware operations to pick from the source to the destination locations
    It uses the PickingPipette class and the Mapping class
    """

    def __init__(self, pick_dir, prefix, mapping):
        """Loads the files for classification and initializes PickingPipette class
        
        :param pick_dir: directory for classification and pick files
        :type pick_dir: str
        :param prefix: prefix name details
        :type prefix: str
        :param mapping: instance of mapping class
        :type mapping: class instance

        :raises FileNotFoundError: loggings critical if any of the files are not found
        """

        logging.info('Initializing Picking Pipette hardware controller')
        cfg_dir = Path().absolute().parent
        try:
            self.pp = PickingPipette(cfg_dir)
        except Exception as e:
            logging.info("Could not initialize and connect hardware controller")
        
        self.pick_dir = pick_dir
        self.prefix = prefix
        self.class_file = None
        self.pick_param_file = None
        
        #TODO handle offset and pick type in Mapping or here?
        #TODO handle dest array and source array in Mapping or here?
        #Make it efficient so not repeating calcs
        self.mapping = mapping
        
        
        self.matches = None



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
        self.picked_file = os.path.join(self.pick_dir, picked_filename)
    
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
    
        self.mapping.go_to_well(well, offset)

        #TODO use helpers.mapping load_wells and go_to_well for control of the microscope stage



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
                #TODO handle this is mapping? 
                #If not load the config
                offset = - pick_type_config['larvae']['offset']
            else:
                offset = pick_type_config['larvae']['offset']
            
            self.move_to_pick(self.matches['slotName'][match], offset)
            self.pp.move_pipette('pick')
            sleep(1)
            self.pp.draw()
            self.pp.move_pipette('clearance')

            #TODO better handle dispense well position calling mapping
            
            (x, y) = self.get_dest_xy(self.matches['dispenseWell'][match])
            self.pp.move_to_dest((x,y))
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