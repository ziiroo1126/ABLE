import os
from pathlib import Path

# Root
ROOT_DIR = Path(__file__).parent.parent.absolute()

# Data
MODELS_DIR = ROOT_DIR / "data" / "models"
TEXT_DATA_DIR = ROOT_DIR / "data" / "selected_data"
ABLE_DATA_DIR = ROOT_DIR / "able"

# logging
LOG_DIR = ROOT_DIR / "log"
LOG_ABLE_DIR = LOG_DIR / "able"

# makesure directories exist
os.makedirs(TEXT_DATA_DIR, exist_ok=True)
os.makedirs(ABLE_DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(LOG_ABLE_DIR, exist_ok=True)


def get_texts_path(dataset_name):
    return TEXT_DATA_DIR / f"{dataset_name}.jsonl"


def get_ABLE_dir(dataset_name):
    path = ABLE_DATA_DIR / dataset_name
    os.makedirs(path, exist_ok=True)
    return path


def get_log_ABLE_dir():
    return LOG_ABLE_DIR


def get_models_path(model_name: str):
    return MODELS_DIR / f"{model_name}.yaml"
