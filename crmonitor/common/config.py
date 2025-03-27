from functools import lru_cache
from importlib import resources as pkg_resources
from typing import Dict

import crmonitor
from crmonitor.common.helper import load_yaml


@lru_cache(maxsize=None)
def get_traffic_rule_config() -> Dict:
    with pkg_resources.path(crmonitor, "traffic_rules_rtamt.yaml") as traffic_rules_path:
        traffic_rules_config = load_yaml(traffic_rules_path)

        if traffic_rules_config is None:
            raise RuntimeError(
                f"Failed to load traffic rule config from '{traffic_rules_path}': Due to an unkown reason the traffic rule config file could not be read."
            )

    return traffic_rules_config


def get_traffic_rule_from_config(rule_name: str) -> str:
    traffic_rules_config = get_traffic_rule_config()
    traffic_rules = traffic_rules_config["traffic_rules"]

    if rule_name not in traffic_rules:
        available_rules = ", ".join(traffic_rules.keys())
        raise RuntimeError(
            f"Failed to read traffic rule {rule_name} from traffic rule config: The rule does not exist! Available rules are: {available_rules}."
        )

    return traffic_rules[rule_name]


@lru_cache(maxsize=None)
def get_evaluation_config():
    with pkg_resources.path(crmonitor, "config.yaml") as traffic_rules_path:
        traffic_rules_config = load_yaml(traffic_rules_path)
    return traffic_rules_config
