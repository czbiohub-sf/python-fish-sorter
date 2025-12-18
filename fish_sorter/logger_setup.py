import logging
import datetime
import sys
from pathlib import Path

def setup_logger(name: str=None):
    """
    Creates and returns a logger configured to log to both file and stdout.
    """

    log_dir = Path(__file__).resolve().parent.parent / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{ts}_fish_sorter.log"
    
    # Get module-specific logger
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    curr_file_log = any(isinstance(h, logging.FileHandler) for h in root.handlers)
    curr_stream_log = any(isinstance(h, logging.StreamHandler) for h in root.handlers)

    if not curr_file_log:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        root.addHandler(file_handler)

    if not curr_stream_log:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )

        stream_handler.setFormatter(stream_formatter)
        root.addHandler(stream_handler)

    return logging.getLogger(name or __name__)