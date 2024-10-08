import json
import logging
import os
import sys
from pathlib import Path
from time import sleep
from zaber_controller import ZaberController
from valve_controller import ValveController

class PickingPipette():
    """Coordinated control of Pipette movement, pneumatics, and dispense plate
        It uses the ZaberController and ValveController classes
    """

    def __init__(self, parent_dir):
        """Runs pipette hardware setup and passes config parameters to each hardware
        
        :param parent_dir: parent directory for config files
        :type parent_dir: str
        :raises FileNotFoundError: loggings critical if the hardware config file not found
        """
        
        parent_dir = Path(parent_dir) / config / hardware
        self.hardware_data = {}

        for filename in os.listdir(parent_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(parent_dir, filename)
                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        var_name = os.path.splitext(filename)[0]
                        hardware_data[var_name] = data
                        logging.info('Loaded {} config file'.format(var_name))
                except FileNotFoundError:
                    logging.critical("Config file not found")       

    def connect(self, env='prod'):
        """Connect to hardware
        
        :param env: environment as to whether in production or test mode
        :type env: str
        """
        
        if env == 'test':
            # Change this depending on computer
            logging.info("Loaded test environment")
            
        self.zc = ZaberController(self.hardware_data['zaber_config'], env)
        self.vc = ValveController(self.hardware_data['pneumatic_config'], env)

        logging.info('Setting pneumatics idle to Atmospheric')
        self._valve_cmd(self.hardware_data['pneumatic_config']['register']['func_idle_type'], self.hardware_data['pneumatic_config']['function_codes']['idle_atm'])

    def disconnect(self):
        """Does all the connection shutdown 
        """

        logging.info("Homing zaber arms and closing connection")
        self.zc.home_arm(['p','x','y'])
        self.zc.disconnect()
        self.vc.disconnect()

    def reset(self):
        """Reset hardware connection
        """

        self.disconnect()
        self.connect()

    def draw(self):
        """Sends Draw function command to valve controller
        This is use to aspirate with pipette
        """

        address_offset = self.hardware_data['pneumatic_config']['register']['func_add_offset']
        func_code = self.hardware_data['pneumatic_config']['function_codes']['draw']
        logging.info(f'Sending draw command with function code {func_code}')
        self._valve_cmd(address_offset, func_code, self.draw_time)

    def expel(self):
        """Sends Expel function command to valve controller
        This is use to dispense from pipette
        """

        address_offset = self.hardware_data['pneumatic_config']['register']['func_add_offset']
        func_code = self.hardware_data['pneumatic_config']['function_codes']['expel']
        logging.info(f'Sending expel command with function code {func_code}')
        self._valve_cmd(address_offset, func_code, self.expel_time)

    def idle(self):
        """Toggles to idle atmospheric pressure 
        """

        address_offset = self.hardware_data['pneumatic_config']['register']['func_add_offset']
        func_code = self.hardware_data['pneumatic_config']['function_codes']['func_idle']
        logging.info(f'Setting to Atmospheric Idle with function code {func_code}')
        self._valve_cmd(address_offset, func_code)

    def pressure(self, state: bool=False):
        """Toggles the pressure valve according to state

        :param state: On/Off state of pressure valve
        :type state: bool
        """

        address_offset = self.hardware_data['pneumatic_config']['register']['func_add_offset']
        
        if state:
            func_code = self.hardware_data['pneumatic_config']['function_codes']['press_on']
            logging.info(f'Setting Pressure On with function code {func_code}')
        else:
            func_code = self.hardware_data['pneumatic_config']['function_codes']['press_off']
            logging.info(f'Setting Pressure Off with function code {func_code}')
        
        self._valve_cmd(address_offset, func_code)

    def vacuum(self, state: bool=False):
        """Toggles the vacuum valve according to state

        :param state: On/Off state of vacuum valve
        :type state: bool
        """

        address_offset = self.hardware_data['pneumatic_config']['register']['func_add_offset']
        
        if state:
            func_code = self.hardware_data['pneumatic_config']['function_codes']['vac_on']
            logging.info(f'Setting Vacuum On with function code {func_code}')
        else:
            func_code = self.hardware_data['pneumatic_config']['function_codes']['vac_off']
            logging.info(f'Setting Vacuum Off with function code {func_code}')
        
        self._valve_cmd(address_offset, func_code)

    def draw_time(self, time: int):
        """Updates the draw time setting in the valve controller

        :param time: time in [ms]
        :type time: int
        """

        address_offset = self.hardware_data['pneumatic_config']['register']['draw_time_add_offset']
        self._valve_cmd(address_offset, time)
        logging.info(f'Update draw time to {time} ms')
        self.draw_time = time
    
    def expel_time(self, time: int):
        """Updates the expel setting in the valve controller

        :param time: time in [ms]
        :type time: int
        """

        address_offset = self.hardware_data['pneumatic_config']['register']['expel_time_add_offset']
        self._valve_cmd(address_offset, time)
        logging.info(f'Update expel time to {time} ms')
        self.expel_time = time

    def _pipette_wait(self, time: int):
        """Time to wait for valve controller to finish execution

        :param time: time in [ms]
        :type time: int
        """
        t = double(time)/1000
        pause = max([0.05, t/3])
        n = max([1, round(t/pause)])

        for i in n:
            sleep(pause)
            if self.vc.read_register(address_offset, 1) == 0:
                break

    def _valve_cmd(self, address_offset: int, value: int, time: int=0):
        """Sends write command to valve controller and reads state after call

        :address_offset: register address offset from the start_address
        :type address_offset: int

        :param value: state controller function code or setting
        :type value: int
        """

        self.vc.write_register(address_offset, value)
        self._pipette_wait(time)