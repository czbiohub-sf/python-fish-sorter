import json
import logging
from time import sleep
from typing import Optional, Tuple
from zaber_motion import Library, Units
from zaber_motion.ascii import Connection, Axis
from zaber_motion.exceptions.connection_failed_exception import ConnectionFailedException
from zaber_motion.exceptions.movement_failed_exception import MovementFailedException

class ZaberController():
    """Communicate with Zaber devices over serial to move the stage or the gripper
    """

    def __init__(self, config: dict, env='prod'):
        """Setup the serial connection between with the zaber device

        :param config: The zaber specific parameters defined in the 
                    zaber_config.json file
        :type config: dict {'port': <name of the serial port>,
                    'location': <x,y,z>, ...} 
        :param env: The environment to run the Zaber Controller.
        :type env: string, either 'prod' or 'dev'
        """

        self.zaber = None
        self.stage_axes = {}
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
                self._set_axis(self.zaber.detect_devices()[0])
                logging.info('Homing all')
                self.home_arm()
            elif self.env == 'dev':
                logging.info('Establishing connection with mock Zaber devices')
                self.zaber = Zaber(self.config['port'])
                logging.info('Zaber devices successfully connected')
                # Set the names for each axis
                self._set_axis(self.zaber.detect_devices()[0])
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

    def _set_axis(self, stage):
        """Set the x y z stage dictionary variables based off the peripheral name

        :param stage: zaber x,y,z stage
        :type stage: tuple of zaber device objects
        """

        for i in range(3):
            name = stage.get_axis(i+1).peripheral_name
            if name == 'LSQ450D-E01T3A':
                self.stage_axes.update({'x': stage.get_axis(i+1)})
                self.stage_axes['x'].settings.set("maxspeed", self.config['max_speed']['x'], Units.VELOCITY_MILLIMETRES_PER_SECOND)
            elif name == 'LSQ075B-T4A-ENG2690':
                self.stage_axes.update({'y': stage.get_axis(i+1)})
                self.stage_axes['y'].settings.set("maxspeed", self.config['max_speed']['y'], Units.VELOCITY_MILLIMETRES_PER_SECOND)
            elif name == 'LSQ150B-T3A':
                self.stage_axes.update({'z': stage.get_axis(i+1)})
                self.stage_axes['z'].settings.set("maxspeed", self.config['max_speed']['z'], Units.VELOCITY_MILLIMETRES_PER_SECOND)

    def home_arm(self, arm: Optional[list]=None):
        """Home either all or a subset of the devices

        The devices include the xyz stages. The order in which
        it homes is dependent on the list passed. The order is important 
        to ensure the device does not crash while homing.

        :param arm: list of the devices to home in the desired sequence,
                    defaults to None, if None homes everything
        :type arm: list of str, optional
        :raises: Any Zaber exception requires restart and reinitialization of Zaber connection
        """
        
        home = ['z','x','y'] if arm == None else arm
        for h in home:
            try:
                self._move_arm(h)
            except:
                raise

    def _move_arm(self, arm: str, dist: Optional[float]=None, is_relative: bool=False):
        """Move any arm 'x','y','z' by a fixed amount

        :param arm: The arm to move x' or 'y' or 'z'
        :type arm: str
        :param dist: The distance to move in mm, if None: home arm, defaults to None
        :type dist: float, optional
        :param is_relative: True: move a relative distance, False: move an absolute distance,
                    defaults to False
        :type is_relative: bool, optional
        :raises MovementFailedException: Logs if the desired position is not reached
        :raises ConnectionFailedException: Logs if the zaber connection fails
        """

        device_arm = self.stage_axes[arm]
        try:
            if dist is None:
                device_arm.home()
            elif is_relative:
                device_arm.move_relative(dist, Units.LENGTH_MILLIMETRES)
            else:
                device_arm.move_absolute(dist, Units.LENGTH_MILLIMETRES)
        except MovementFailedException:
            cur_pos = device_arm.get_position(unit=Units.LENGTH_MILLIMETRES)
            logging.critical('Failed to move {} arm'.format(device_arm))
            logging.critical('Stuck At: {}, Desired Pos: {}'.format(cur_pos, dist))
            raise
        except ConnectionFailedException:
            logging.critical('Zaber Connection Failed')