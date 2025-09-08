# from pydantic import Field
# from pydantic_settings import BaseSettings
import logging
import logging.config
import os
from pathlib import Path


# Determine Project Root
PROJECT_ROOT_DIR_ENV = os.getenv("PROJECT_ROOT_DIR")

if PROJECT_ROOT_DIR_ENV:
    PROJ_ROOT: Path = Path(PROJECT_ROOT_DIR_ENV)
    print(f"Using PROJECT_ROOT_DIR from env: {PROJ_ROOT}")
else:
    raise ValueError("PROJECT_ROOT_DIR environment variable not set")

# Paths
# for idx, x in enumerate(Path(__file__).resolve().parents):
#     print(f'i: {idx} {x}')

print(PROJ_ROOT)
DATA_DIR = PROJ_ROOT / "data"
LOGS_DIR = PROJ_ROOT / "logs"
SRC_DIR = PROJ_ROOT / "src"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def setup_logging():
    """Configures logging from the logging.conf file."""
    LOGGING_CONFIG_PATH = SRC_DIR / "logging.conf"
    if LOGGING_CONFIG_PATH.exists():
        logging.config.fileConfig(LOGGING_CONFIG_PATH, disable_existing_loggers=False)
    else:
        print(f"Warning: Logging configuration file not found at {LOGGING_CONFIG_PATH}")
        logging.basicConfig(level=logging.INFO)  # Fallback to basic logging
