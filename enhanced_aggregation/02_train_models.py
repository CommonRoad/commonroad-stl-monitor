from pathlib import Path

from commonroad_mpr.learning.data_loader import DataLoader
from commonroad_mpr.learning.gp_regression import ModelTrainer
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as MprCfg
from commonroad_mpr.utils.configuration_builder import ScenarioType
from crmonitor.predicates.predicate_factory import PredicateFactory

learning_data_path = (
    Path(__file__).parent.parent / "output" / "learning_data" / "learning_data.csv"
)
models_output_path = Path(__file__).parent.parent / "output" / "models"

# Although the DataGenerator does not require config files, internal mpr methods (might) do.
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
    path_root=str(
        Path(__file__).parent.parent.parent / "commonroad-model-predictive-robustness"
    ),
    folder_config="config_files",
    default_profile="default",
)
# The ModelTrainer requires the arity of each predicate to determine the number of samples that should be used to train the model for each predicate.
arities = {
    predicate_name.value: {"arity": predicate.arity}
    for predicate_name, predicate in PredicateFactory()._evaluators.items()
    if predicate_name != "interface"
}
MprCfg.update_with_config({"feature_variable": arities})

data_loader = DataLoader.create_from_file(learning_data_path)

all_general_predicates = [
    "in_front_of",
    "in_same_lane",
    "cut_in",
    "keeps_safe_distance_prec",
    "brakes_abruptly",
    "brakes_abruptly_relative",
    "precedes",
    "single_lane",
    "keeps_lane_speed_limit",
    "keeps_type_speed_limit",
    "keeps_brake_speed_limit",
    "keeps_fov_speed_limit",
    "keeps_lane_speed_limit_star",
    "slow_leading_vehicle",
    "preserves_traffic_flow",
]

all_interstate_predicates = [
    "in_congestion",
    "exist_standing_leading_vehicle",
    "in_standstill",
    "left_of",
    "drives_faster",
    "in_slow_moving_traffic",
    "in_queue_of_vehicles",
    "drives_with_slightly_higher_speed",
    "right_of_broad_lane_marking",
    "left_of_broad_lane_marking",
    "on_access_ramp",
    "on_main_carriage_way",
    "interstate_broad_enough",
    "on_shoulder",
    "in_leftmost_lane",
    "drives_leftmost",
    "drives_rightmost",
    "in_rightmost_lane",
    "main_carriageway_right_lane",
]

trainer = ModelTrainer(data_loader, ["in_front_of"], ScenarioType.INTERSTATE)

models = trainer.train()

trainer.dump_models(models, models_output_path)
