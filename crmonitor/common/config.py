from functools import lru_cache
from importlib import resources as pkg_resources
from pathlib import Path

from ruamel.yaml import YAML

import crmonitor


def load_yaml(file_name: Path | str) -> dict | None:
    """
    Loads configuration setup from a yaml file

    :param file_name: name of the yaml file
    """
    file_name = Path(file_name)
    config = YAML().load(file_name)
    return config


@lru_cache(maxsize=None)
def get_traffic_rule_config() -> dict:
    with pkg_resources.path(crmonitor, "traffic_rules_rtamt.yaml") as traffic_rules_path:
        traffic_rules_config = load_yaml(traffic_rules_path)

        if traffic_rules_config is None:
            raise RuntimeError(
                f"Failed to load traffic rule config from '{traffic_rules_path}': Due to an unkown reason the traffic rule config file could not be read."
            )

    return traffic_rules_config


def get_traffic_rule_from_config(rule_name: str) -> str | None:
    traffic_rules_config = get_traffic_rule_config()
    traffic_rules = traffic_rules_config["traffic_rules"]

    return traffic_rules.get(rule_name)
