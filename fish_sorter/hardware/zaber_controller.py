import json
import logging
from time import sleep
from typing import Optional, Tuple
from zaber_motion import Library, Units
from zaber_motion.binary import Connection, Device, CommandCode
from zaber_motion.exceptions.connection_failed_exception import ConnectionFailedException
from zaber_motion.exceptions.movement_failed_exception import MovementFailedException

class ZaberController():
    """Communicate with Zaber devices over serial to move the stages
        Note that this class is using the zaber_motion.binary library instead of 
        zaber_motion.ascii because of older T-series devices that do not support the ASCII Protocol
    """

    def __init__(self, config: dict, env='prod'):
        """Setup the serial connection between with the zaber device

        :param config: The zaber specific parameters defined in the 
                    zaber_config.json file
        :type config: dict {'port': <name of the serial port>,
                    'location': <x, y, p>, ...} 
        :param env: The environment to run the Zaber Controller.
        :type env: string, either 'prod' or 'dev'
        """

        self.zaber = None
        self.stage_alias = {}
        self.config = config
        self.env = env
        self._connect()

    def _connect(self):
        """Create a serial communication with the zaber devices

        :raises ConnectionFailedException: Logs critical if the connection fails
        """

        try:
            if self.env == 'prod':
                logging.info('Establishing connection with Zaber devices')
                self.zaber = Connection.open_serial_port(self.config['port'])
                logging.info('Zaber devices successfully connected')
                # Set the names and velocities for each axis
                self._set_axis()
                logging.info('Homing all')
                self.home_arm()
            elif self.env == 'dev':
                logging.info('Establishing connection with mock Zaber devices')
                self.zaber = Zaber(self.config['port'])
                logging.info('Zaber devices successfully connected')
                # Set the names for each axis
                self._set_axis(self.zaber.detect_devices())
                logging.info('Homing all')
                self.home_arm()                
        except ConnectionFailedException:
            logging.critical("Could not make connection to zaber stage")
            raise

    def disconnect(self):
        """Closes the serial Connection
        """

        self.zaber.close()
        logging.info('Closed Zaber device connection')

    def _set_axis(self):
        """Set the x, y, p stage dictionary variables based off the peripheral name

        :param stage: zaber x, y, p stage
        :type stage: tuple of zaber device objects
        """
        
        self.stages = self.zaber.detect_devices()
        logging.info('Stage list {} '.format(self.stages))
        for stage in self.stages:
            name = stage.name
            if name == 'T-LSQ150D':
                self.stage_alias[stage] = 'x'
                stage.generic_command_with_units(CommandCode.SET_TARGET_SPEED, data = self.config['max_speed']['x'], from_unit = Units.NATIVE, to_unit = Units.NATIVE, timeout = 0.0)
            elif name == 'A-LSQ150A-E01':
                self.stage_alias[stage] = 'y'
                stage.generic_command_with_units(CommandCode.SET_TARGET_SPEED, data = self.config['max_speed']['y'], from_unit = Units.NATIVE, to_unit = Units.NATIVE, timeout = 0.0)
            elif name == 'T-LSQ075B':
                self.stage_alias[stage] = 'p'
                stage.generic_command_with_units(CommandCode.SET_TARGET_SPEED, data = self.config['max_speed']['p'], from_unit = Units.NATIVE, to_unit = Units.NATIVE, timeout = 0.0)       


    def home_arm(self, arm: Optional[list]=None):
        """Home either all or a subset of the devices

        The devices include the x, y, p stages. The order in which
        it homes is dependent on the list passed. The order is important 
        to ensure the device does not crash while homing.

        :param arm: list of the devices to home in the desired sequence,
                    defaults to None, if None homes everything
        :type arm: list of str, optional
        :raises: Any Zaber exception requires restart and reinitialization of Zaber connection
        """
        
        home = ['p','x','y'] if arm == None else arm
        for h in home:
            try:
                self.move_arm(h)
            except:
                raise
    
    def move_arm(self, arm: str, dist: Optional[float]=None, is_relative: bool=False):
        """Move any arm 'x','y','p' by a fixed amount

        :param arm: The arm to move x' or 'y' or 'p'
        :type arm: str
        :param dist: The distance to move in mm, if None: home arm, defaults to None
        :type dist: float, optional
        :param is_relative: True: move a relative distance, False: move an absolute distance,
                    defaults to False
        :type is_relative: bool, optional
        :raises MovementFailedException: Logs if the desired position is not reached
        :raises ConnectionFailedException: Logs if the zaber connection fails
        """

        for key, value in self.stage_alias.items():
            if value == arm:
                device_arm = key
        
        
        try:
            if dist is None:
                device_arm.home()
            elif is_relative:
                device_arm.move_relative(dist, Units.LENGTH_MILLIMETRES, timeout = 60)
            else:
                device_arm.move_absolute(dist, Units.LENGTH_MILLIMETRES, timeout = 60)
        except MovementFailedException:
            cur_pos = device_arm.get_position(unit=Units.LENGTH_MILLIMETRES)
            logging.critical('Failed to move {} arm'.format(device_arm))
            logging.critical('Stuck At: {}, Desired Pos: {}'.format(cur_pos, dist))
            raise
        except ConnectionFailedException:
            logging.critical('Zaber Connection Failed')

    def get_pos(self, arm: str) -> float:
        """returns the positon of the zaber stage

        :param arm: The arm to move x' or 'y' or 'p'
        :type arm: str
        :return: The stage location position in mm
        :rtype: float
        """
        
        for key, value in self.stage_alias.items():
            if value == arm:
                device_arm = key
        try:
            return device_arm.get_position(unit=Units.LENGTH_MILLIMETRES)
        except ConnectionFailedException:
            logging.critical('Zaber Connection Failed')