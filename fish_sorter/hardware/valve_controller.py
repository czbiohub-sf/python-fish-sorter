import json
import logging
from pymodbus.client import ModbusTcpClient

log = logging.getLogger(__name__)

class ValveController():
    """Communicate with Wago valve controller to actuate the pressure and vacuum valves
    Note that the Wago controller was programmed with CoDeSys to receive specific function codes
    to perform actions
    """

    def __init__(self, config: dict, env='prod'):
        """Open config file and setup the TCP connection with the Wago controller

        :param config: The pneumatics box specific parameters defined in the 
                    pneumatic_config.json file
        :type config: dict {'connect': <TCP connection parameters>,
                    'registers': <starting address, address offset>, ...} 
        :param env: The environment to run the Valve Controller.
        :type env: string, either 'prod' or 'dev'
        """

        self.valve = None
        self.config = config
        self.env = env
        logging.info('Start connection to Valve Controller')
        self._connect()

    def _connect(self):
        """Create TCP communication with the Wago controller

        :raises ConnectionError: Logs critical if the connection fails
        """

        host = self.config['connect']['host']
        port = self.config['connect']['port']

        self.valve = ModbusTcpClient(
            host=host,
            port=port,
            timeout=5,
            retries=3,
        )
        logging.info(f'{self.valve}')
        try:
            self.valve.connect()
            logging.info('Valve modbus client connected')
        except Exception as e:
            logging.critical('Could not make connection to Wago valve controller')
            raise

    def _check_connect(self):
        """Reconnects to modbus client if the socket is not open"""

        if not self.valve.is_socket_open():
            logging.warning('Valve modbus socket closed. Reconnecting...')
            self.valve.close()
            self._connect()

    def disconnect(self):
        """Closes the TCP Connection
        """
        if self.valve:
            self.valve.close()
            logging.info('Closed valve controller connection')

    def read_register(self, add_offset: int, count: int=1):
        """Reads the state of register specified by the function code

        :param add_offset: register address offset from the start_address
        :type add_offset: int
        :param count: state controller function code or setting
        :type count: int

        :raises ModbusException: Logs critical if the connection fails
        :raise ValueError: Logs critical if the response contains an error in the Modbus library 
        :raises ExceptionResponse: Logs critical if Modbus protocol exception response
        """

        self._check_connect()
        address = self.config['register']['start_address'] + add_offset

        try:
            return self.valve.read_holding_registers(address=address, count=count)
        except Exception as e:
            logging.exception(f"Received exception {e}")  
            raise 
    
    def write_register(self, add_offset: int, value: int):
        """Writes to the register specified by the function code

        :param add_offset: register address offset from the start_address
        :type add_offset: int

        :param value: state controller function code or setting
        :type value: int

        :raises ModbusException: Logs critical if the connection fails
        :raise ValueError: Logs critical if the response contains an error in the Modbus library 
        :raises ExceptionResponse: Logs critical if Modbus protocol exception response
        """

        self._check_connect()
        address = self.config['register']['start_address'] + add_offset
        try:
            logging.info(f'Writing {value} to register {address}')
            write = self.valve.write_register(address=address, value=value)
            return write
        except Exception as e:
            logging.exception('Valve modbus write failed. Attempting one reconnect.')
            self.valve.close()
            self._connect()
            return self.valve.write_register(address, value)