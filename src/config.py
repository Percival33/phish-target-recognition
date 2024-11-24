from pydantic_settings import BaseSettings
from pathlib import Path

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]


class AppConfig(BaseSettings):
    DATA_DIR: Path = PROJ_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    PHISHIRIS_DL_DATASET_DIR: Path = RAW_DATA_DIR / "phishIRIS_DL_Dataset"


config = AppConfig()
