import yaml
from typing import Any, Dict


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Loads configuration data from a specified YAML file.

    Args:
        config_file (str): The file path to the YAML configuration.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters.
    """
    with open(config_file, "r", encoding="utf-8") as file:
        config: Dict[str, Any] = yaml.safe_load(file)
    return config
