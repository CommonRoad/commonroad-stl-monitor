import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from commonroad_mpr.learning import DataLoader, ModelEvaluator
from commonroad_mpr.learning.gp_regression import read_model
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

# Select the atomic predicates that should be evaluated.
predicates = []
# Select the meta-predicates that should be evaluated (prefixed with '$'!). NOTE: only select top-level meta-predicates here.
meta_predicates = ["$cut_in", "$drives_leftmost", "$drives_rightmost", "$left_of"]


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


META_PREDICATE_DEFINITIONS = {
    "$slow_leading_vehicle": (
        np.logical_and,
        "slow_as_leading_vehicle",
        "in_same_lane",
        "in_front_of",
    ),
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
    "$drives_leftmost": (np.logical_or, "close_to_left_bound", "$close_to_vehicle_left"),
    "$drives_rightmost": (
        np.logical_or,
        "close_to_right_bound",
        "$close_to_vehicle_right",
    ),
    "$cut_in": (
        np.logical_and,
        (np.logical_not, "single_lane"),
        (np.logical_or, "$approach_from_left", "$approach_from_right"),
        "in_same_lane",
    ),
    "$approach_from_left": (np.logical_and, "lat_left_of", "heading_right"),
    "$approach_from_right": (np.logical_and, "lat_left_of", "heading_right"),
    "$left_of": (np.logical_and, "lat_left_of_vehicle", "$lon_intersecting_vehicles"),
    "$lon_intersecting_vehicles": "rear_behind_front",  # TODO: Meta predicate consists only of one atomic predicate?
    "$close_to_vehicle_left": (
        np.logical_and,
        "lat_close_to_vehicle_left",
        "$lon_intersecting_vehicles",
    ),
    "$close_to_vehicle_right": (
        np.logical_and,
        "lat_close_to_vehicle_right",
        "$lon_intersecting_vehicles",
    ),
}

data_loader = DataLoader.create_from_file(learning_data_path)
evaluator = ModelEvaluator(predicates, data_loader, models_path)

results = evaluator.evaluate()


def _get_all_atomic_predicates(definition):
    if isinstance(definition, str):
        if definition.startswith("$"):
            return _get_all_atomic_predicates(META_PREDICATE_DEFINITIONS[definition])
        else:
            return [definition]
    else:
        operands = definition[1:]
        return list(
            itertools.chain.from_iterable(
                _get_all_atomic_predicates(operand) for operand in operands
            )
        )


def _resolve_meta_predicate_definition(definition, balanced_df):
    """resolve the definition of a meta-predicate to the metrics of the atomic predicates."""
    if isinstance(definition, str):
        if definition.startswith("$"):
            # recursive meta-predicate
            return _resolve_meta_predicate_definition(
                META_PREDICATE_DEFINITIONS[definition], balanced_df
            )
        else:
            model = read_model(definition, models_path)
            # Use the existing data loader functionality, because it also handles the feature extraction.
            _data_loader = DataLoader([definition], balanced_df)
            X_test, y_test = _data_loader.Xy(definition)
            y_pred, _ = model.predict(X_test.astype(float))
            y_pred = np.clip(y_pred, -1, 1)
            max_pred = np.max(y_pred)
            min_pred = np.min(y_pred)
            max_test = np.max(y_test)
            min_test = np.min(y_test)
            std_pred = np.std(y_pred)
            std_test = np.std(y_pred)

            # Copied from `ModelEvaluator`.
            bool_pred = (y_pred > 0).flatten()
            bool_gt = (y_test > 0).flatten()
            return (bool_pred, bool_gt, std_pred, std_test, max_pred, min_pred, max_test, min_test)
    else:
        operator = definition[0]
        operands = definition[1:]

        results = []

        for operand in operands:
            results.append(_resolve_meta_predicate_definition(operand, balanced_df))

        # Unzip the results into separate lists
        preds, gts, std_preds, std_tests, max_preds, min_preds, max_tests, min_tests = zip(*results)

        if operator == np.logical_not:
            std_test = std_tests[0]
            std_pred = std_preds[0]

            max_pred = max_preds[0]
            min_pred = min_preds[0]

            max_test = max_tests[0]
            min_test = min_tests[0]
        elif operator == np.logical_and:
            std_test = min(std_tests)
            std_pred = min(std_preds)

            max_pred = min(max_preds)
            min_pred = max(min_preds)

            max_test = min(max_tests)
            min_test = max(min_tests)
        elif operator == np.logical_or:
            std_test = max(std_tests)
            std_pred = max(std_preds)

            max_pred = max(max_preds)
            min_pred = min(min_preds)

            max_test = max(max_tests)
            min_test = min(min_tests)
        else:
            raise RuntimeError(f"Invalid operator {operator}")

        return (
            operator(*preds),
            operator(*gts),
            std_pred,
            std_test,
            max_pred,
            min_pred,
            max_test,
            min_test,
        )


eps = 1e-9
n_samples_each = 500  # Configure the number of positive and negative samples.
count_valid_threshold = 10000  # This value must match the value in the `DataLoader`.
for meta_predicate_name in meta_predicates:
    definition = META_PREDICATE_DEFINITIONS[meta_predicate_name]

    # The `DataLoader` filters based on count_valid. Therefore, we must already make sure
    # that our sample contains enough valid samples for any downstream operation.
    atomic_predicates = _get_all_atomic_predicates(definition)
    valid_masks = []
    for atomic_predicate in atomic_predicates:
        valid_mask = (
            data_loader.data[("predicates", atomic_predicate, "count_valid")]
            > count_valid_threshold
        )
        valid_masks.append(valid_mask)

    filtered_data = data_loader.data.loc[np.logical_and.reduce(valid_masks)]
    # The `DataLoader` also removes duplicates for predicates with arity 1.
    # It's difficult to check this here, so all duplicates are removed instead.
    filtered_data_reset = filtered_data.reset_index("other_id")
    filtered_data = filtered_data[~filtered_data_reset.index.duplicated(keep="first")]

    # Select the data of the non-meta-predicate
    original_atomic_predicate_data = filtered_data.predicates[meta_predicate_name.lstrip("$")]
    rows_true = filtered_data[original_atomic_predicate_data.robustness > 0.0]
    rows_false = filtered_data[original_atomic_predicate_data.robustness <= 0.0]

    # Handles the cases where one of the rows does not have at least n_samples_each.
    min_row_length = min(len(rows_true), len(rows_false), n_samples_each)
    rows_true_balanced = rows_true.sample(min_row_length)
    rows_false_balanced = rows_false.sample(min_row_length)

    balanced_df = pd.concat([rows_true_balanced, rows_false_balanced])

    bool_pred, bool_gt, std_pred, std_test, max_pred, min_pred, max_test, min_test = (
        _resolve_meta_predicate_definition(definition, balanced_df)
    )

    TP = np.logical_and(bool_pred, bool_gt).sum()
    FP = np.logical_and(bool_pred, ~bool_gt).sum()
    FN = np.logical_and(~bool_pred, bool_gt).sum()
    TN = np.logical_and(~bool_pred, ~bool_gt).sum()

    precision = (TP + eps) / (TP + FP + eps)
    recall = (TP + eps) / (TP + FN + eps)
    f1_score = 2 * precision * recall / (precision + recall)
    results[meta_predicate_name] = {
        "size": len(bool_pred),
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "std_test": std_test,
        "std_pred": std_pred,
        "span_test": max_test - min_test,
        "span_pred": max_pred - min_pred,
    }

evaluator.visualize(results)
evaluator.save(results, metrics_output_path)

# plt.show()
