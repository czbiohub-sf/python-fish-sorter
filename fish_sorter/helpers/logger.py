import logging
import os
import sys
import datetime

from logging.handlers import RotatingFileHandler
from typing import List, Optional



"""
Usage

from helpers.logger import Logger

logger_manager = Logger(auto_detect=True)
logger = logger_manager.get_logger("custom")

or

logger_manager = Logger(logger_names=["napari", "napari_micromanager"], auto_detect=True or False)
logger = logger_manager.get_logger("custom")

"""



class Logger:
    """Custom logger class
    """

    def __init__(self, log_dir='../../logs', log_level=logging.INFO, logger_names: Optional[List[str]] = None, auto_detect: bool=True):
        """
        :param log_dir: the directory to store the log file
        :type log_dir: str
        :param log_level: logging level (logging.INFO, logging.DEBUG)
        :type logger object attribute
        :param logger_names: list of additional loggers to capture in logging (default: root, __name__)
        :type logger_names: List[str]
        :param auto_detect: select whether to detect all logs or use only a specified list
        :type auto_detect: bool
        """

        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strfrtime('%Y-%m-%d')
        base_log_file = os.path.join(self.log_dir, f'{timestamp}_log')

        file_handler = RotatingFileHandler(f'{base_log_file}.txt', maxBytes = 5*1024*1024)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            '%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)

        default_loggers = ['root', __name__]

        if auto_detect:
            detected_loggers = list(logging.root.manager.loggerDict.keys())
        else:
            detected_loggers = []

        self.loggers = {logging.getLogger(name) for name in set(default_loggers + (logger_names or []) + detected_loggers)}

        for logger in self.loggers:
            logger.setLevel(log_level)
            logger.addHandler(console_handler)

    def get_logger(self, name=__name__):
        """Returns a logger instance for a specific module or class

        :param name: script name
        :type name: str
        """

        return logging.getLogger(name)