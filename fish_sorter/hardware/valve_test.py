#!/usr/bin/env python3
"""Pymodbus synchronous client example.

An example of a single threaded synchronous client.

usage: simple_sync_client.py

All options must be adapted in the code
The corresponding server must be started before e.g. as:
    python3 server_sync.py
"""

# --------------------------------------------------------------------------- #
# import the various client implementations
# --------------------------------------------------------------------------- #
import argparse
import pymodbus.client as ModbusClient
from pymodbus import (
    ExceptionResponse,
    FramerType,
    ModbusException,
    pymodbus_apply_logging_config,
)

ADDRESS_OFFSET = 4
COMM = "tcp"
HOST = "192.168.1.10"
PORT = 502

def run_sync_simple_client(func_code, framer=FramerType.SOCKET):
    """Run sync client."""
    # activate debugging
    pymodbus_apply_logging_config("DEBUG")

    print("get client")
    if COMM == "tcp":
        client = ModbusClient.ModbusTcpClient(
            HOST,
            port=PORT,
            framer=framer,
            # timeout=10,
            # retries=3,
            # retry_on_empty=False,
            # source_address=("localhost", 0),
        )
    elif COMM == "udp":
        client = ModbusClient.ModbusUdpClient(
            HOST,
            port=PORT,
            framer=framer,
            # timeout=10,
            # retries=3,
            # retry_on_empty=False,
            # source_address=None,
        )
    elif COMM == "serial":
        client = ModbusClient.ModbusSerialClient(
            PORT,
            framer=framer,
            # timeout=10,
            # retries=3,
            # retry_on_empty=False,
            baudrate=9600,
            bytesize=8,
            parity="N",
            stopbits=1,
            # handle_local_echo=False,
        )
    else:
        print(f"Unknown client {COMM} selected")
        return

    print("connect to server")
    client.connect()

    print("get and verify data")

    try:
        print(type(func_code))
        rr = client.read_holding_registers(12288 + ADDRESS_OFFSET, count = 1)
        print(rr)
        print(f'{rr.registers[0]:016b}')

    except ModbusException as exc:
        print(f"Received ModbusException({exc}) from library")
        client.close()
        return
    if rr.isError():
        print(f"Received Modbus library error({rr})")
        client.close()
        return
    if isinstance(rr, ExceptionResponse):
        print(f"Received Modbus library exception ({rr})")
        # THIS IS NOT A PYTHON EXCEPTION, but a valid modbus message
        client.close()
        return
    
    try:
        wr = client.write_register(12288 + ADDRESS_OFFSET, func_code)
        print(wr)
        rr = client.read_holding_registers(12288 + ADDRESS_OFFSET, count = 1)
        print(rr)
        print(f'{rr.registers[0]:016b}')

    except ModbusException as exc:
        print(f"Received ModbusException({exc}) from library")
        client.close()
        return
    if rr.isError():
        print(f"Received Modbus library error({rr})")
        client.close()
        return
    if isinstance(rr, ExceptionResponse):
        print(f"Received Modbus library exception ({rr})")
        # THIS IS NOT A PYTHON EXCEPTION, but a valid modbus message
        client.close()
        return

    print("close connection")
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test a Function Code. You must specify the Function Code Value")
    parser.add_argument(
        "func_code",
        type = int,
        help ="Function Code: 8, 9, 16, 17"
    )
    args = parser.parse_args()
    func_code = args.func_code
    run_sync_simple_client(func_code)