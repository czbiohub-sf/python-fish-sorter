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
import pymodbus.client as ModbusClient
from pymodbus import (
    ExceptionResponse,
    Framer,
    ModbusException,
    pymodbus_apply_logging_config,
)

import math
import numpy as np

ADDRESS_OFFSET = 2
TOTAL_VALVES = 3
TOTAL_BYTES = math.ceil(TOTAL_VALVES / 8)
TOTAL_WORDS = math.ceil(TOTAL_BYTES / 2)
# WORD_PAD = TOTAL_WORDS * 16 - TOTAL_VALVES - 1

# TODO load in valve polarities

def run_sync_simple_client(comm, host, port, framer=Framer.SOCKET):
    """Run sync client."""
    # activate debugging
    pymodbus_apply_logging_config("DEBUG")

    print("get client")
    if comm == "tcp":
        client = ModbusClient.ModbusTcpClient(
            host,
            port=port,
            framer=framer,
            # timeout=10,
            # retries=3,
            # retry_on_empty=False,
            # source_address=("localhost", 0),
        )
    elif comm == "udp":
        client = ModbusClient.ModbusUdpClient(
            host,
            port=port,
            framer=framer,
            # timeout=10,
            # retries=3,
            # retry_on_empty=False,
            # source_address=None,
        )
    elif comm == "serial":
        client = ModbusClient.ModbusSerialClient(
            port,
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
        print(f"Unknown client {comm} selected")
        return

    print("connect to server")
    client.connect()

    print("get and verify data")

    client.write_coil(0, 1)
    # rr = client.read_coils(1, 1)
    # print(rr.bits)
    try:
        # rr = client.read_coils(512, TOTAL_WORDS, slave=1)
        # rr = client.read_holding_registers(512, 1, unit=1)
        rr = client.read_holding_registers(512, 1)


        # If this gives 0, read registers instead
        # https://github.com/czbiohub-sf/Matlab-Wago/blob/14975edaec94fdf86dc066dd9ffa62f389cca297/wagoNModbus.m#L234-L236
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
    
    # words = np.zeros((1, TOTAL_WORDS)).astype(int)
    # for i in range(0, TOTAL_WORDS):
    #     print(client.read_coils(512+TOTAL_WORDS-i, 1, slave=1))


    print(rr)
    # print(bin(rr.registers[0]).zfill(16))
    print(f'{rr.registers[0]:016b}')
    # print(f"TOTAL WORDS {TOTAL_WORDS}")  

    print("close connection")
    client.close()


if __name__ == "__main__":
    run_sync_simple_client("tcp", "192.168.1.10", "502")