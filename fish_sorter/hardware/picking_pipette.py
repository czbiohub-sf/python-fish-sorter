import json
import logging
import os
import sys
from pathlib import Path
from time import sleep
from zaber_controller import ZaberController

class PickingPipette():
    """Coordinated control of Pipette movement, pneumatics, and dispense plate
        It uses the ZaberController and PneumaticController classes
    """

    def __init__(self, parent_dir):
        """Runs pipette hardware setup and passes config parameters to each hardware
        
        :param parent_dir: parent directory for config files
        :type parent_dir: str
        :raises FileNotFoundError: loggings critical if the hardware config file not found
        """
        
        parent_dir = Path(parent_dir) / config / hardware
        self.json_data = {}

        for filename in os.listdir(parent_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(parent_dir, filename)
                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        var_name = os.path.splitext(filename)[0]
                        json_data[var_name] = data
                        logging.info('Loaded {} config file'.format(var_name))
                except FileNotFoundError:
                    logging.critical("Config file not found")       

    def connect_hardware(self, env='prod'):
        """Connect to hardware
        
        :param env: environment as to whether in production or test mode
        :type env: str
        """
        
        if env == 'test':
            # Change this depending on computer
            logging.info("Loaded test environment")
            
        self.zc = ZaberController(self.json_data['zaber_config'], env)
        self.pc = PneumaticController(self.json_data['pneumatic_config'], env)

    def run_complete(self):
        """Does all the connection close up after the full run is complete
        """

        logging.info("Homing zaber arms and closing connection")
        self.zc.home_arm(['p','x','y'])
        self.zc.disconnect()