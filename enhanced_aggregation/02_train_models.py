import logging
from pathlib import Path

from commonroad_mpr.learning.data_loader import DataLoader
from commonroad_mpr.learning.gp_regression import ModelTrainer
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as MprCfg
from commonroad_mpr.utils.configuration_builder import ScenarioType
from crmonitor.predicate_grouping import all_general_predicates, all_interstate_predicates
from crmonitor.predicates.predicate_factory import PredicateFactory

logging.basicConfig(level=logging.INFO)

learning_data_path = Path(__file__).parent.parent / "output" / "learning_data" / "learning_data.csv"
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
                    "use_phantom_lane": True
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
# The ModelTrainer requires the arity of each predicate to determine the number of samples that should be used to train the model for each predicate.
arities = {
    predicate_name.value: {"arity": predicate.arity}
    for predicate_name, predicate in PredicateFactory()._evaluators.items()
    if predicate_name != "interface"
}
MprCfg.update_with_config({"feature_variable": arities})

data_loader = DataLoader.create_from_file(learning_data_path)

trainer = ModelTrainer(data_loader, all_general_predicates + all_interstate_predicates, ScenarioType.INTERSTATE, training_iter=100)

models = trainer.train()

trainer.dump_models(models, models_output_path)
