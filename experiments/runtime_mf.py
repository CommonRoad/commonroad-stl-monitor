from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import (
    Lanelet,
    LaneletNetwork,
    LaneletType,
    LineMarking,
)
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.state import CustomState
from commonroad.common.file_reader import CommonRoadFileReader

from crmonitor.common.vehicle import CurvilinearStateManager, Vehicle
from crmonitor.common.helper import load_yaml
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.world import World
from crmonitor.predicates.position import (
    PredOnLaneletWithTypeIntersection,
    PredInIntersectionConflictArea,
    PredStopLineInFront,
)

from crmonitor.predicates.general import (
    PredTurningLeft,
    PredGoingStraight,
    PredTurningRight,
)

from crmonitor.predicates.velocity import (
    PredInStandStill
)

from crmonitor.predicates.acceleration import (
    PredCausesBrakingIntersection
)

from pathlib import Path
import numpy as np
import os
import time

if __name__ == "__main__":
    root_path = Path(__file__).parents[1] / "crmonitor"
    config_path = root_path / "config.yaml"
    config = load_yaml(str(config_path))
    config["scale_rob"] = True
    config["d_sl"] = 1.0
    config["d_br"] = 15.0
    config["a_br"] = -1.0
    config["standstill_error"] = 0.1
    config["scenario"] = "intersection"
    config["intersection_road_network_param"]["map_type"] = "dataset"
    
    rules_path = root_path / "traffic_rules_rtamt.yaml"
    traffic_rules = load_yaml(str(rules_path))
    traffic_rules["traffic_rules_param"]["use_mpr"] = False
    config["use_mpr"] = False
    traffic_rules["traffic_rules_param"]["mpr_scenario"] = "intersection"
    scenario_root_path = root_path.parent / "scenarios"
    
    scenario_file = os.path.join(
        scenario_root_path, "test_intersection/DEU_AAH1-2_112100_T-2249.xml"
    )
    ego_id = 10037
    other_id = 10038

    time_step = 50

    scenario, _ = CommonRoadFileReader(scenario_file).open(lanelet_assignment=True)
    world = World.create_from_scenario(scenario, config)
    ego_vehicle = world.vehicle_by_id(ego_id)
    other_vehicle = world.vehicle_by_id(other_id)
    predicates = [
        "stop_line_in_front",
        "in_intersection_conflict_area",
        "turning_right",
        "turning_left",
        "going_straight",
        "on_lanelet_with_type_intersection",
        "causes_braking_intersection",
        "in_standstill",
    ]
    pred_eval_list = [PredStopLineInFront(config),
                      PredInIntersectionConflictArea(config),
                      PredTurningRight(config),
                      PredTurningLeft(config),
                      PredGoingStraight(config),
                      PredOnLaneletWithTypeIntersection(config),
                      PredCausesBrakingIntersection(config),
                      PredInStandStill(config)]

    print("evaluating one predicate at a time:")
    
    for i in range(len(predicates)):
        t = time.time()
        for _ in range(100):
            pred_eval_list[i].evaluate_robustness(world, time_step, [ego_id, other_id])
        elapsed = time.time() - t
        print(predicates[i] + f": {elapsed/100}")
    
    print("Done!")