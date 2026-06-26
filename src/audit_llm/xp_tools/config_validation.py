"""Configuration validation and comparison utilities."""

import re
import warnings
from pathlib import Path

from audit_llm.data_transforms import revert_dictionary
from audit_llm.file_io import load_json, write_json


def compare_xp_configs(xp_config1, xp_config2, mode="warning", _path=""):
    """
    Compare two nested dict configs.

    Parameters
    ----------
    mode : {"error", "warning"}
        - "error": raise ValueError on mismatch
        - "warning": issue warnings instead
    """

    def handle(msg):
        if mode == "error":
            raise ValueError(msg)
        elif mode == "warning":
            warnings.warn(msg, stacklevel=2)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    for key in xp_config1:
        path = f"{_path}.{key}" if _path else key

        if key not in xp_config2:
            handle(f"Key '{path}' not found in both xp_configs.")
            continue

        v1, v2 = xp_config1[key], xp_config2[key]

        if isinstance(v1, dict) and isinstance(v2, dict):
            if key == "figures":
                continue
            compare_xp_configs(v1, v2, mode=mode, _path=path)
        else:
            if v1 != v2:
                handle(f"Value mismatch for key '{path}': {v1} != {v2}")


def check_xp_config_coherence(xp_config_path, current_xp_config):
    if xp_config_path.exists():
        xp_config_from_checkpoint = load_json(path=xp_config_path)
        compare_xp_configs(xp_config_from_checkpoint, current_xp_config)
    else:
        write_json(current_xp_config, xp_config_path)


def get_iter_idx_from_calculations_config(iterator_name, xp_config):
    """Return the calculation key for the given iterator name.

    With readable keys, the key IS the iterator name, so this is now
    an identity lookup that validates the key exists.
    """
    calculations_config = xp_config["calculations"]
    if iterator_name not in calculations_config:
        raise KeyError(f"The {iterator_name} iterator must be defined in calculations for classification.")
    return iterator_name


def is_pattern_strictly_in_string(s, pattern):
    return bool(re.search(rf"{re.escape(pattern)}($|[^0-9A-Za-z])", s))
