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
    log_file = log_dir / f"{ts}.log"
    
    # Get module-specific logger
    root = logging.getLogger(name or __name__)
    root.setLevel(logging.INFO)

    if not root.handlers:
        return logger

    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.FOrmatter(
        "%(asctime)s - %(levelnames)s - %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    rott.addHandler(file_handler)

    stream_handler.setFormatter(stream_formatter)
    root.addHandler(stream_handler)

    return logging.getLogger(name or __name__)