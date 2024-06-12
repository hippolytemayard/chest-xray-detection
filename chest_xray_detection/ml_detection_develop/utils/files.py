import os

import yaml
from omegaconf import OmegaConf


def make_exists(file_path: str) -> None:
    """Create the directory if it does not exist."""
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def load_yaml(path: str) -> OmegaConf:
    """Load a YAML file and return an OmegaConf configuration."""
    yaml_file = OmegaConf.load(path)
    return yaml_file


def load_yaml_full_loader(path: str) -> OmegaConf:
    """Load a YAML file and return an OmegaConf configuration.
    Use this function if YAML file contains special characters.
    """
    yaml_file = open(path, "r")
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    config = OmegaConf.create(config)

    return config
