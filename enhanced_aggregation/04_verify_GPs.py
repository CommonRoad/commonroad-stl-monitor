import logging
from pathlib import Path

import matplotlib.pyplot as plt
from commonroad_mpr.learning import DataLoader, ModelEvaluator
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as MprCfg
from crmonitor.predicates.predicate_factory import PredicateFactory

logging.basicConfig(level=logging.INFO)

learning_data_path = Path(__file__).parent.parent / "output" / "learning_data" / "learning_data.csv"
models_path = Path(__file__).parent.parent / "output" / "models"
# models_path = Path("/home/finf/pretrainedMPR/interstate/2023-07-04")
# models_path = Path("/home/finf/gp_training/commonroad-stl-monitor/output/model_bkp")

metrics_output_path = Path(__file__).parent.parent / "output" / "metrics" / "gp_metrics.csv"
metrics_output_path.parent.mkdir(exist_ok=True)

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

evaluator = ModelEvaluator(["makes_u_turn"], data_loader, models_path)


results = evaluator.evaluate()
evaluator.visualize(results)
evaluator.save(results, metrics_output_path)

plt.show()
