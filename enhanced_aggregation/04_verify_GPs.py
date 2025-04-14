import logging
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from commonroad_mpr.learning import DataLoader, ModelEvaluator
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as MprCfg
from crmonitor.predicate_grouping import (
    all_general_predicates,
    all_interstate_predicates,
    meta_only,
)
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

all_predicates = all_interstate_predicates + all_general_predicates + meta_only
evaluator = ModelEvaluator(all_predicates, data_loader, models_path)

results = evaluator.evaluate()

META_PREDICATES = {
    "$slow_leading_vehicle": (np.logical_and, "slow_as_leading_vehicle", "in_same_lane"),
    "$exist_standing_leading_vehicle": (
        np.logical_and,
        "in_same_lane",
        "in_front_of",
        "in_standstill",
    ),
    "$in_congestion": (np.logical_and, "in_same_lane", "in_front_of", "has_congestion_velocity"),
    "$in_slow_moving_traffic": (
        np.logical_and,
        "in_front_of",
        "in_same_lane",
        "has_slow_moving_velocity",
    ),
    "$in_queue_of_vehicles": (np.logical_and, "in_front_of", "in_same_lane", "has_queue_velocity"),
    "$precedes": (np.logical_and, "in_same_lane", "in_front_of"),
    "$drives_leftmost": (np.logical_or, "close_to_left_bound", "close_to_vehicle_left"),
    "$drives_rightmost": (
        np.logical_or,
        "close_to_right_bound",
        "close_to_vehicle_right",
    ),
    "$cut_in": (
        np.logical_and,
        (np.logical_not, "single_lane"),
        (np.logical_or, "$approach_from_left", "$approach_from_right"),
        "in_same_lane",
    ),
    "$approach_from_left": (np.logical_and, "lat_left_of", "heading_right"),
    "$approach_from_right": (np.logical_and, "lat_left_of", "heading_right"),
    "$left_of": (np.logical_and, "lat_left_of_vehicle", "lon_intersecting_vehicles"),
    "$lon_intersecting_vehicles": "rear_behind_front",  # TODO: Meta predicate consists only of one atomic predicate?
    "$close_to_vehicle_left": (
        np.logical_and,
        "lat_close_to_vehicle_left",
        "lon_intersecting_vehicles",
    ),
    "$close_to_vehicle_right": (
        np.logical_and,
        "lat_close_to_vehicle_right",
        "lon_intersecting_vehicles",
    ),
}


def _resolve_definition(definition):
    if isinstance(definition, str):
        if definition.startswith("$"):
            # recursive meta-predicate
            return _resolve_definition(META_PREDICATES[definition])
        else:
            return results[definition]["bool_pred"], results[definition]["bool_gt"]
    else:
        operator = definition[0]
        operands = definition[1:]
        preds = []
        gts = []
        for operand in operands:
            pred, gt = _resolve_definition(operand)
            preds.append(pred)
            gts.append(gt)

        # Reduce both arrays to the same length.
        # TODO: Should another strategy be used here?
        min_pred_len = min([len(p) for p in preds]) if preds else 0
        min_gt_len = min([len(g) for g in gts]) if gts else 0
        preds = [pred[0:min_pred_len] for pred in preds]
        gts = [gt[0:min_gt_len] for gt in gts]

        return operator(*preds), operator(*gts)


meta_predicate_results = {}
_eps = 1e-9
for meta_predicate_name, definition in META_PREDICATES.items():
    bool_pred, bool_gt = _resolve_definition(definition)

    TP = np.logical_and(bool_pred, bool_gt).sum()
    FP = np.logical_and(bool_pred, ~bool_gt).sum()
    FN = np.logical_and(~bool_pred, bool_gt).sum()
    TN = np.logical_and(~bool_pred, ~bool_gt).sum()

    precision = (TP + _eps) / (TP + FP + _eps)
    recall = (TP + _eps) / (TP + FN + _eps)
    f1_score = 2 * precision * recall / (precision + recall)
    results[meta_predicate_name] = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }

# TODO: No visualization because meta-predicates do not compute SHAP
# evaluator.visualize(results)
# evaluator.save(results, metrics_output_path)
#
# TODO: move to ModelEvaluator
columns_to_exclude = {"shap_values", "bool_gt", "bool_pred", "y_test", "y_pred"}
flattened_results = []
for key, nested_dict in results.items():
    filtered_nested_dict = {"predicate": key}
    for nested_key, nested_value in nested_dict.items():
        if nested_key in columns_to_exclude:
            continue
        filtered_nested_dict[nested_key] = nested_value

    flattened_results.append(filtered_nested_dict)
results_df = pd.DataFrame(flattened_results)
results_df.to_csv(metrics_output_path)

# plt.show()
