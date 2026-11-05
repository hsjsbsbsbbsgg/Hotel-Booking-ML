import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["quick_mode"] = os.environ.get("QUICK") == "1"
    if config["quick_mode"]:
        config.update(config["quick"])
    return config
