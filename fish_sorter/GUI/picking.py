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

    def __init__(self, cfg_dir, pick_dir, prefix, offset, dtime, iplate, dp_array, phc=None):
        """Loads the files for classification and initializes PickingPipette class
        
        :param cfg_dir: parent path directory for all of the config files
        :type cfg_dir: path
        :param pick_dir: experiment directory for classification and pick files
        :type pick_dir: str
        :param prefix: prefix name details
        :type prefix: str
        :param offset: offset value from center points for picking
        :type offset: np array
        :param dtime: delay time in s to be used between pipette actions from config
        :type dtime: float
        :param iplate: image plate class
        :type: image plate class instance
        :param dp_file: path to dispense plate array in config folder
        :type: path
        :param phc: PickingPipette class
        :type phc: Picking pipette class instance

        :raises FileNotFoundError: loggings critical if any of the files are not found
        """

        logging.info('Initializing Pick class')
        dplate_array = cfg_dir / 'arrays' / dp_array
        self.phc = phc
        self.phc.define_dp(dplate_array)

        self.pick_dir = pick_dir
        self.prefix = prefix
        self.class_file = None
        self.pick_param_file = None

        self.iplate = iplate
        
        self.matches = None
        self.pick_offset = offset
        self.dtime = dtime

    def connect_hardware(self):
        """Connects to hardware
        """

        self.phc.connect(env='prod')

    def disconnect_hardware(self):
        """Disconnects from hardware
        """

        self.phc.disconnect()

    def reset_hardware(self):
        """Reset the hardware connection
        """
        
        self.phc.reset()

    def move_calib(self, pick: bool=True, well: Optional[str]=None):
        """Checks for calibration of pipette tip height

        :param pick: pick location is True
        :type pick: bool
        :param well: well ID
        :type well: str
        """

        if pick:
            logging.info('Move for Picking Calib Height')
            self.phc.move_for_calib(pick)
        else:
            logging.info('Move for Dispense Calib Height')
            logging.info(f'well passed: {well}')
            logging.info(f'pick is {pick}')
            self.phc.move_for_calib(pick, well)

    def set_calib(self, pick: bool=True):
        """Sets pipette calibration once user acknowledges location
                
        :param pick: pick location is True
        :type pick: bool
        """
            
        self.phc.set_calib(pick)

    def get_classified(self):
        """Opens classification and pick parameter files
        """

        logging.info('Load classification and picking files')

        pickable_files = []

        for filename in os.listdir(self.pick_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.pick_dir, filename)
                try:
                    if 'classifications.csv' in filename:
                        self.class_file = pd.read_csv(file_path)
                        logging.info('Loaded {}'.format(filename))
                    elif 'pickable.csv' in filename:
                        pickable_files.append(file_path)
                except FileNotFoundError:
                    logging.critical("File not found")

        if pickable_files:
            pickable_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_pickable_file = pickable_files[0]
            self.pick_param_file = pd.read_csv(latest_pickable_file)
            logging.info('Loaded latest pickable file: {os.path.basename(latest_pickable_file)}')
            logging.info(f'{self.pick_param_file}')
        else:
            logging.info('No Pickable files founds')

        picked_filename = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + self.prefix + '_picked.csv'
        self.picked_file = os.path.normpath(os.path.join(self.pick_dir, picked_filename))

        logging.info('Load image plate calibration and wells')

    def pick_me(self):
        """Performs all actions to pick from the source plate to the destination plate using
        the match list created by match_pick
        """

        logging.info('Begin iterating through pick list')
        self.matches.drop(columns=['lHead']).head(0).to_csv(self.picked_file, index=False)
        self.phc.move_pipette('clearance')
        self.phc.dest_home()
        
        for match in self.matches.index:
            if self.matches['lHead'][match]:
                offset = np.array([-self.pick_offset[0], self.pick_offset[1]])
                logging.info(f'Offset left head:{offset}')
            else:
                offset = self.pick_offset
                logging.info(f'Offset right head:{offset}')
            
            self.iplate.go_to_well(self.matches['slotName'][match], offset)
            self.phc.move_pipette('pick')
            sleep(self.dtime)
            self.phc.draw()
            sleep(self.dtime)
            self.phc.move_pipette('clearance')
            
            self.phc.dplate.go_to_well(self.matches['dispenseWell'][match])
            self.phc.move_pipette('dispense')
            self.phc.expel()
            sleep(self.dtime)
            self.phc.expel()
            sleep(self.dtime)
            self.phc.move_pipette('clearance')
            self.phc.dest_home()
            logging.info('Picked fish in {} to {}'.format(self.matches['slotName'][match], self.matches['dispenseWell'][match]))
            pd.DataFrame([self.matches.drop(columns=['lHead']).iloc[match].values], columns=self.matches.drop(columns=['lHead']).columns)\
                .to_csv(self.picked_file, mode='a', header=False, index=False)
        
        #TODO how to more elegantly handle lHead, rightHead, none, etc
        #call to mapping?
        #call mapping for the dest plate at init?      
        #Better handle 'lHead' column in csv??? or rather abstract at some point to also include embryos
        #Use pick_type_config.json better to determine what columns are needed
        #Should it snake through the list of options?


        self.done()

    def done(self):
        """Helper to call when picking is complete
        """

        logging.info('Finished Picking!')

    def match_pick(self):
        """Matches the desired pick parameters to the classification
        """

        class_drop = self.class_file.reset_index(drop=True)
        pick_param_drop = self.pick_param_file.reset_index(drop=True)
        matching = class_drop.columns.intersection(pick_param_drop.columns).difference(['slotName', 'dispenseWell'])
        merge = pd.merge(class_drop, pick_param_drop, on=list(matching), how='inner')
        merge_sorted = pd.merge(self.pick_param_file[['dispenseWell']], merge, on='dispenseWell', how='inner')
        self.matches = pd.DataFrame({'slotName': merge_sorted['slotName'], 'dispenseWell': merge_sorted['dispenseWell'], 'lHead': merge_sorted['lHead']})
        logging.info('Created pick list')

    def single_pick(self, dtime: float=1.00):
        """Perform a single pick at the current position

        :param dtime: delay time in seconds between pipette stage movement and valve control function calls
        :type dtime: float
        """

        logging.info(f'Begin single pick with delay time {dtime} seconds')
        self.phc.move_pipette('clearance')
        self.phc.dest_home()
        
        self.phc.move_pipette('pick')
        sleep(dtime)
        self.phc.draw()
        sleep(dtime)
        self.phc.move_pipette('clearance')
            
        self.phc.dplate.go_to_well(self.matches['dispenseWell'][match])
        self.phc.move_pipette('dispense')
        self.phc.expel()
        sleep(dtime)
        self.phc.expel()
        sleep(dtime)

        self.phc.move_pipette('clearance')
        self.phc.dest_home()
        logging.info('Finished Pick')