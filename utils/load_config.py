from pathlib import Path

import yaml


def load_yaml_config(config_path: Path):
    """
    Load the YAML configuration file from the specified path.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        The loaded configuration as a dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        return config
