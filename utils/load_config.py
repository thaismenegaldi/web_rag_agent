import yaml
from pathlib import Path


def load_yaml_config(config_path: Path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        return config
