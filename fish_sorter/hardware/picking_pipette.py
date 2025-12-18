import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional
from time import sleep

from fish_sorter.hardware.zaber_controller import ZaberController
from fish_sorter.hardware.valve_controller import ValveController
from fish_sorter.hardware.dispense_plate import DispensePlate

log = logging.getLogger(__name__)

class PickingPipette():
    """Coordinated control of Pipette movement, pneumatics, and dispense plate
        It uses the ZaberController and ValveController classes
    """

    def __init__(self, parent_dir, zc=None):
        """Runs pipette hardware setup and passes config parameters to each hardware
        
        :param parent_dir: parent directory for config files
        :type parent_dir: path
        :param zc: zaber controller class 
        :type zc: zaber controller instance
        :raises FileNotFoundError: loggings critical if the hardware config file not found
        """
        
        self.hardware_data = {}
        hardware_dir = parent_dir / 'hardware'
        logging.info(f'Picking Pipette hardware dir {hardware_dir}')
        for filename in os.listdir(hardware_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(hardware_dir, filename)
                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        var_name = os.path.splitext(filename)[0]
                        if var_name in data and isinstance(data[var_name], dict):
                            self.hardware_data[var_name] = data[var_name]
                        else:    
                            self.hardware_data[var_name] = data
                        logging.info('Loaded {} config file'.format(var_name))
                except FileNotFoundError:
                    logging.critical("Config file not found")

        self.drw_t = self.hardware_data['picker_config']['pipette']['time']['draw']
        self.exp_t = self.hardware_data['picker_config']['pipette']['time']['expel']
        self.pick_h = self.hardware_data['picker_config']['pipette']['stage']['pick']['p']
        self.disp_h = self.hardware_data['picker_config']['pipette']['stage']['dispense']['p']

        self.pipettor_cfg = hardware_dir / 'picker_config.json'

        self.connect(zc=zc)

    def connect(self, zc=None, env='prod'):
        """Connect to hardware
        
        :param env: environment as to whether in production or test mode
        :type env: str
        """
        
        if env == 'test':
            # Change this depending on computer
            logging.info("Loaded test environment")

        if zc is None:
            self.zc = ZaberController(self.hardware_data['zaber_config'], env)
        else:
            self.zc = zc
        self.vc = ValveController(self.hardware_data['pneumatic_config'], env)

        logging.info('Setting pneumatics idle to Atmospheric')
        self._valve_cmd(self.hardware_data['pneumatic_config']['register']['func_idle_type'], self.hardware_data['pneumatic_config']['function_codes']['idle_atm'])

    def disconnect(self):
        """Does all the connection shutdown 
        """

        logging.info("Homing stage arms and turning off pressure and vacuum")

        self.move_pipette(pos='clearance')
        self.zc.home_arm(['x','y'])
        self.move_pipette(pos='pipette_swing')
        self._valve_cmd(self.hardware_data['pneumatic_config']['register']['func_idle_type'], self.hardware_data['pneumatic_config']['function_codes']['idle_atm'])
        self.zc.disconnect()
        logging.info('Closed stage connnection')
        self.vc.disconnect()
        logging.info('Closed valve connection')

    def reset(self):
        """Reset hardware connection
        """

        self.disconnect()
        self.connect()
        self.define_dp(self.current_dp, self.pixel_size_um)

    def define_dp(self, array_file, pixel_size_um):
        """Define the dispense plate and create an instance of the
        Dispense plate class

        :param array_file: path to dispense plate pick type array in config folder
        :type: path
        :param pixel_size_um: calculation of pixel size in image based on magnification
        :type pixel_size_um: float 
        """

        self.current_dp = array_file
        self.pixel_size_um = pixel_size_um
        self.dplate = DispensePlate(self.zc, self.current_dp, self.pixel_size_um)
        self.dplate.set_calib_pts(pipettor_cfg=self.pipettor_cfg)
        self.dplate.load_wells(xflip=True)
    
    def draw(self):
        """Sends Draw function command to valve controller
        This is use to aspirate with pipette
        """

        address_offset = self.hardware_data['pneumatic_config']['register']['func_add_offset']
        func_code = self.hardware_data['pneumatic_config']['function_codes']['draw']
        logging.info(f'Sending draw command with function code {func_code}')
        self._valve_cmd(address_offset, func_code, self.drw_t)

    def expel(self):
        """Sends Expel function command to valve controller
        This is use to dispense from pipette
        """

        address_offset = self.hardware_data['pneumatic_config']['register']['func_add_offset']
        func_code = self.hardware_data['pneumatic_config']['function_codes']['expel']
        logging.info(f'Sending expel command with function code {func_code}')
        self._valve_cmd(address_offset, func_code, self.exp_t)

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

        logging.info(f'Pressure state requested: {state}')

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

        logging.info(f'Vacuum state requested: {state}')
        
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
        self.drw_t = time
    
    def expel_time(self, time: int):
        """Updates the expel setting in the valve controller

        :param time: time in [ms]
        :type time: int
        """

        address_offset = self.hardware_data['pneumatic_config']['register']['expel_time_add_offset']
        self._valve_cmd(address_offset, time)
        logging.info(f'Update expel time to {time} ms')
        self.exp_t = time

    def _pipette_wait(self, address_offset: int, time: int):
        """Time to wait for valve controller to finish execution

        :param address_offset: register address offset from the start_address
        :type address_offset: int
        :param time: time in [ms]
        :type time: int
        """

        t = float(time)/1000
        pause = max([0.05, t/3])
        n = max([1, round(t/pause)])

        for i in range(n):
            sleep(pause)
            if self.vc.read_register(address_offset) == 0:
                break

    def _valve_cmd(self, address_offset: int, value: int, time: int=0):
        """Sends write command to valve controller and reads state after call

        :param address_offset: register address offset from the start_address
        :type address_offset: int
        :param value: state controller function code or setting
        :type value: int
        :param time: time in [ms]
        :type time: int        
        """

        self.vc.write_register(address_offset, value)
        self._pipette_wait(address_offset, time)

    def move_for_calib(self, pick: bool=True, well: Optional[str]=None):
        """Moves destination stage for pipette calibration

        :param pick: pick or dispense location
        :type pick: bool
        :type well: well location
        :type well: str
        """

        if pick:
            self.dest_home()
        else:
            self.zc.move_arm('p', self.hardware_data['picker_config']['pipette']['stage']['clearance']['p'])
            self.dplate.go_to_well(well)

    def set_calib(self, pick: bool=True):
        """Sets pipette calibration location and saves it to the config

        :param pick: pick or dispense location
        :type pick: bool
        """
        
        if pick:
            self.pick_h = self.zc.get_pos('p')
            logging.info(f'Set pick height to: {self.pick_h}')
        else:
            self.disp_h = self.zc.get_pos('p')
            logging.info(f'Set dispense height to: {self.disp_h}')

        self.save_calib()

    def save_calib(self):
        """Saves the calibration for the pipette
        """

        logging.info('Updating picker config file with pipette calibration')

        with open(self.pipettor_cfg, 'r') as pc:
            pick_cfg = json.load(pc)
            pick_cfg['pipette']['stage']['pick']['p'] = self.pick_h
            pick_cfg['pipette']['stage']['dispense']['p'] = self.disp_h
            pc.close()
        with open(self.pipettor_cfg, 'w') as pc:
            pick_update = json.dump(pick_cfg, pc, indent = 4, separators= (',',': '))
            pc.close()

        logging.info('Saved picker_config.json with updated values')
        
    def dest_home(self):
        """Convenience function to move destination plate to home position
        """

        self.zc.move_arm('p', self.hardware_data['picker_config']['pipette']['stage']['clearance']['p'])
        self.zc.move_arm('x', self.hardware_data['zaber_config']['home']['x'])
        self.zc.move_arm('y', self.hardware_data['zaber_config']['home']['y'])
        logging.info('Moved destination plate to home')

    def move_pipette(self, pos: str):
        """Moves pipette to swing, clearance, pick, or dispense

        :param pos: location to move pipette arm
        :type pos: str
        """

        if pos == 'pick':
            self.zc.move_arm(arm='p', dist=self.pick_h)
        elif pos == 'dispense':
            self.zc.move_arm(arm='p', dist=self.disp_h)
        elif pos == 'pipette_swing':
            self.dest_home()
            self.zc.move_arm(arm='p', dist=self.hardware_data['picker_config']['pipette']['stage'][pos]['p'])
        else:
            self.zc.move_arm(arm='p', dist=self.hardware_data['picker_config']['pipette']['stage'][pos]['p'])
        logging.info(f'Moved pipette to: {pos}')

    def move_pipette_increment(self, dist: float, units: bool=True):
        """Moves pipette a specified distance and unit

        :param dist: distance to move the stage
        :type dist: float
        :param units: units for distance 
        :type units: bool, True for mm / False for um
        """

        if units is False:
            dist =  dist / 1000
        self.zc.move_arm('p', dist, is_relative=True)
        logging.info(f'Moved pipette {dist} mm')

    def move_fluor_img(self):
        """Moves the destination stages to image with fluorescent channels
        The destination stages are moved between the objective and pipettte to prevent reflections
        off of the pipette tip
        """

        self.zc.move_arm('p', self.hardware_data['picker_config']['fluor_img']['stage']['p'])
        self.zc.move_arm('y', self.hardware_data['picker_config']['fluor_img']['stage']['y'])
        self.zc.move_arm('x', self.hardware_data['picker_config']['fluor_img']['stage']['x'])
        logging.info('Move for fluorecent imaging')