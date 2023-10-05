from functools import lru_cache
from importlib import resources as pkg_resources

import crmonitor
from crmonitor.common.helper import load_yaml


@lru_cache(maxsize=None)
def get_traffic_rule_config():
    with pkg_resources.path(
        crmonitor, "traffic_rules_rtamt.yaml"
    ) as traffic_rules_path:
        traffic_rules_config = load_yaml(traffic_rules_path)
    return traffic_rules_config


@lru_cache(maxsize=None)
def get_evaluation_config():
    with pkg_resources.path(crmonitor, "config.yaml") as traffic_rules_path:
        traffic_rules_config = load_yaml(traffic_rules_path)
    return traffic_rules_config
