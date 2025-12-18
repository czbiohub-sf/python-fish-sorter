import logging
import datetime
import sys
import warnings
from pathlib import Path

def setup_logger(name: str=None):
    """
    Creates and returns a logger configured to log to both file and stdout.
    """
    
    # Get root logger
    root = logging.getLogger()
    if getattr(root, "_fish_sorter_configured", False):
        return logging.getLogger(name or __name__)


    log_dir = Path(__file__).resolve().parent.parent / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{ts}_fish_sorter.log"

    root.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    stream_handler.setFormatter(stream_formatter)
    root.addHandler(stream_handler)
    
    logging.captureWarnings(True)
    warnings.simplefilter("default")

    def handle_exception(exc_type, exc_value, exc_traceback):

        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
    
        logger.logging.getLogger(__name__)
        logger.error(
            "Uncaught Exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = handle_exception

    root.__fish_sorter_configured = True

    root.info(f"Logging initialized. Log file {log_file}")

    return logging.getLogger(name or __name__)