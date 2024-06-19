"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import yaml
from easydict import EasyDict


def create_config(config_file: str, prefix: str = "") -> EasyDict:
    """
    Create a configuration dictionary from a YAML file with optional prefixing for directory paths.

    Args:
        config_file (str): Path to the YAML configuration file.
        prefix (str): Optional prefix to add to directory paths. Defaults to "".

    Returns:
        EasyDict: Configuration dictionary with optional prefixed directory paths.
    """
    # Load the YAML configuration file
    with open(config_file, "r") as stream:
        config = yaml.safe_load(stream)

    # Create an EasyDict from the loaded configuration
    cfg = EasyDict()
    for k, v in config.items():
        cfg[k] = v

        # Add prefix to directory paths if specified
        if "DIR" in k:
            if len(prefix) > 0:
                cfg[k] = prefix + cfg[k]
    return cfg


def load_config(config_file_exp: str, prefix: str = "") -> EasyDict:
    """
    Load and merge configuration files with optional prefixing for directory paths.

    Args:
        config_file_exp (str): Path to the experimental configuration file.
        prefix (str): Optional prefix to add to directory paths. Defaults to "".
        parent (bool): If True, use the parent directory of the current working directory.
            Defaults to False.

    Returns:
        EasyDict: Merged configuration dictionary with optional prefixed directory paths.
    """
    # Determine the current working directory or its parent
    cwd = os.getcwd()

    # Load the system configuration file
    sys_config_file = f"{cwd}/configs/config.yaml"
    sys_config = create_config(sys_config_file, prefix=prefix)

    # Load the experimental configuration file
    config = create_config(config_file_exp, prefix=prefix)

    # Set the configuration name from the experimental config file
    config["config_name"] = os.path.basename(config_file_exp).split(".")[0]

    # Identify keys to remove from the system configuration
    remove = []
    for key, val in sys_config.items():
        if key in config:
            remove.append(key)

    # Remove the identified keys from the system configuration
    for key in remove:
        sys_config.pop(key)

    # Update the experimental configuration
    config.update(sys_config)

    return config
