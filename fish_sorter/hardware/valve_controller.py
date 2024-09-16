import json
import logging
import pymodbus.client as ModbusClient
from pymodbus import (
    ExceptionResponse,
    FramerType,
    ModbusException,
    pymodbus_apply_logging_config,
)

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
        self._connect()

    def _connect(self):
        """Create TCP communication with the Wago controller

        :raises ConnectionError: Logs critical if the connection fails
        """
        
        framer = FramerType.SOCKET
        pymodbus_apply_logging_config("DEBUG")

        self.valve = ModbusClient.ModbusTcpClient(
            host=self.config['connect']['host'],
            port=self.config['connect']['port'],
            framer=framer,
            # timeout=10,
            # retries=3,
            # retry_on_empty=False,
            # source_address=("localhost", 0),
        )
        
        try:
            self.valve.connect()
        
            if not self.valve.connected():
                raise ConnectionError('Failed to connect TCP device')
            else:
                logging.info(f'Connected to Wago valve controller {self.valve}')
        except ConnectionError as e:
            logging.critical('Could not make connection to Wago valve controller')
            raise

    def disconnect(self):
        """Closes the TCP Connection
        """

        self.valve.close()
        logging.info('Closed valve controller connection')

    def read_register(self, add_offset: int, value: int):
        """Reads the state of register specified by the function code

        :add_offset: register address offset from the start_address
        :type add_offset: int

        :param value: state controller function code or setting
        :type value: int

        :raises ModbusException: Logs critical if the connection fails
        :raise ValueError: Logs critical if the response contains an error in the Modbus library 
        :raises ExceptionResponse: Logs critical if Modbus protocol exception response
        """

        try:
            rr = self.valve.read_holding_registers(self.config['register']['start_address'] + add_offset, value)
            logging.info(f'Reading register {rr}')
            logging.info(f'Register state: {rr.registers[0]:016b}')

        except ModbusException as exc:
            logging.critical(f"Received ModbusException({exc}) from library")  
            raise ModbusException(f"Received ModbusException({exc}) from library")
        if rr.isError():
            logging.critical(f"Received Modbus library error({rr})") 
            raise ValueError(f"Received Modbus library error({rr})")
        if isinstance(rr, ExceptionResponse):
            # THIS IS NOT A PYTHON EXCEPTION, but a valid modbus message
            logging.critical(f"Received Modbus library exception ({rr})")
            raise ExceptionResponse
    
    def write_register(self, add_offset: int, value: int):
        """Writes to the register specified by the function code

        :add_offset: register address offset from the start_address
        :type add_offset: int

        :param value: state controller function code or setting
        :type value: int

        :raises ModbusException: Logs critical if the connection fails
        :raise ValueError: Logs critical if the response contains an error in the Modbus library 
        :raises ExceptionResponse: Logs critical if Modbus protocol exception response
        """

        try:
            wr = self.valve.write_registers(self.config['register']['start_address'] + add_offset, value)
            logging.info(f'Writing register {wr}')

        except ModbusException as exc:
            logging.critical(f"Received ModbusException({exc}) from library")
            raise ModbusException(f"Received ModbusException({exc}) from library")
        if wr.isError():
            logging.critical(f"Received Modbus library error({wr})")
            raise ValueError(f"Received Modbus library error({wr})")
        if isinstance(wr, ExceptionResponse):
            # THIS IS NOT A PYTHON EXCEPTION, but a valid modbus message
            logging.critical(f"Received Modbus library exception ({wr})")
            raise ExceptionResponse