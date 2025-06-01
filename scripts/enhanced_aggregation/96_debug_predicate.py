import logging
from pathlib import Path

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad_mpr.common.observation import World as WorldMPR
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as MprCfg
from crmonitor.common.world import World
from crmonitor.predicates.base import PredicateEvaluatorConfig, PredicateMprConfig
from crmonitor.predicates.predicate_factory import PredicateFactory

logging.basicConfig(level=logging.INFO)

scenario_name = "DEU_LocationALower26-1_261256_T-1506"
time_step = 0
ego_vehicle_id = 10186
other_vehicle_id = 10176
predicate = "lat_close_to_vehicle_left"

scenarios_load_path = (
    Path(__file__).parent.parent.parent / "scenarios-for-semantic-aware-stl" / "highD"
)
model_path = Path(__file__).parent.parent / "output" / "models"

MprCfg.build_configuration(
    config={
        "common": {
            "scenario": "interstate",
            "lane": {
                # Increased the default parameters to work around projection limit issues in MPR
                "lateral_projection_domain_limit": 500,
                "extend_length": 500,
                "large_resampling_step": 3.5,
                "num_chankins_corner_cutting": 1,
            },
            "road_network": {
                "interstate": {
                    "use_phantom_lane": False
                }  # Must disable phantom lanes, because otherwise commonroad-dc segfaults...
            },
        },
    },
    # Path root must point to a local revision of commonroad-model-predictive-robustness.
    # This configuration, assumes that the repo is in the same directory as stl-monitor repo.
    # If this is not the case for your setup, adjust the path here accordingly.
    path_root=str(Path(__file__).parent.parent.parent / "commonroad-model-predictive-robustness"),
    folder_config="config_files",
    default_profile="default",
)
scenario_path = scenarios_load_path / f"{scenario_name}.xml"

predicate_evaluator_config = PredicateEvaluatorConfig(
    scale_rob=True,
    mpr=PredicateMprConfig(enabled=True, ml=True, rectification=False, model_path=model_path),
)
predicate_factory = PredicateFactory(predicate_evaluator_config)
predicate_evaluator = predicate_factory.get_predicate(predicate)

scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)
world = World.create_from_scenario(scenario)
mpr_world = WorldMPR.create_from_scenario(scenario)

mpr_robustness = predicate_evaluator.evaluate_mpr_ml(
    world, mpr_world, time_step, [ego_vehicle_id, other_vehicle_id]
)
print("MPR (GP):", mpr_robustness)

mpr_robustness = predicate_evaluator.evaluate_mpr(
    world, mpr_world, time_step, [ego_vehicle_id, other_vehicle_id]
)
print("MPR:", mpr_robustness)

mfr_robustness = predicate_evaluator.evaluate_robustness(
    world, time_step, [ego_vehicle_id, other_vehicle_id]
)
print("MFR:", mfr_robustness)
mfr_bool_robustness = predicate_evaluator.evaluate_boolean(
    world, time_step, [ego_vehicle_id, other_vehicle_id]
)
print("Bool:", mfr_bool_robustness)
