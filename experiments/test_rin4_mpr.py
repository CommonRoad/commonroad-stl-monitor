import os
import sys
import logging
import unittest
from pathlib import Path

import numpy as np

from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.helper import load_yaml
from crmonitor.common.world import World
from crmonitor.evaluation.proposition_evaluation import PropositionRuleEvaluator
from crmonitor.rule.rule_node import (
    RuleNode,
    PredicateNode,
    ExistNode,
    AllNode,
)

if __name__ == "__main__":
    root_path = Path(__file__).parents[1] / "crmonitor"
    config_path = root_path / "config.yaml"
    config = load_yaml(str(config_path))
    config["scale_rob"] = True
    config["d_sl"] = 1.0
    config["scenario"] = "intersection"
    config["intersection_road_network_param"]["map_type"] = "dataset"

    rules_path = root_path / "traffic_rules_rtamt.yaml"
    traffic_rules = load_yaml(str(rules_path))
    traffic_rules["traffic_rules_param"]["use_mpr"] = True
    traffic_rules["traffic_rules_param"]["mpr_scenario"] = "intersection"
    scenario_root_path = root_path.parent / "scenarios"

    # scenario_file = os.path.join(
    #     scenario_root_path, "test_intersection/DEU_AAH1-2_76900_T-7049.xml"
    # )
    # ego_id = 10065

    # scenario_file = os.path.join(
    #     scenario_root_path, "test_intersection/DEU_AAH1-2_109450_T-9599.xml"
    # )
    # ego_id = 10112

    scenario_file = os.path.join(
        scenario_root_path, "test_intersection/DEU_AAH-2_221950_T-2099.xml"
    )
    ego_id = 10068
    # scenario_file = os.path.join(
    #     scenario_root_path, "test_intersection/DEU_TestRIN1-3_1_T-1.xml"
    # )
    # ego_id = 31
    # scenario_file = os.path.join(
    #     scenario_root_path, "test_intersection/DEU_TestIntersectionInteract-2_1_T-1.xml"
    # )
    # ego_id = 30
    scenario, _ = CommonRoadFileReader(scenario_file).open(lanelet_assignment=True)
    world = World.create_from_scenario(scenario, config)
    ego_vehicle = world.vehicle_by_id(ego_id)
    rule_eval = PropositionRuleEvaluator.create_from_config(
        world, ego_vehicle, "R_IN4", traffic_rules_config=traffic_rules, use_boolean=False
    )
    rule = rule_eval._rule
    rule_robustness = list()
    prob_robs = list()
    with open('output.txt', 'w') as file:
        sys.stdout = file
        for i in range(
                rule_eval.ego_vehicle.start_time, rule_eval.ego_vehicle.end_time + 1
        ):
            rob = rule_eval.update()
            prob_rob = rule_eval.get_propositions()
            prob_robs.append(prob_rob)
            rule_robustness.append(rob)
            print("--------------------------------")
            print("time = ", i)
            print("rob = ", rob)
            for prob in prob_rob[0]:
                print(prob, ":", prob_rob[0][prob])
    sys.stdout = sys.__stdout__
    rule_robustness = np.array(rule_robustness)
