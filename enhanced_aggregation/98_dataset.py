import logging
import math
from pathlib import Path

import numpy as np
from commonroad_mpr.learning import DataLoader
from crmonitor.predicate_grouping import (
    all_general_predicates,
    all_interstate_predicates,
)

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

learning_data_path = Path(__file__).parent.parent / "output" / "learning_data" / "learning_data.csv"

selected_predicates = all_general_predicates + all_interstate_predicates
data_loader = DataLoader.create_from_file(learning_data_path)


analysis = {}
for predicate in selected_predicates:
    orig_count_true = sum(
        [1 for x in data_loader.data[("predicates", predicate, "bool")] if x == True]
    )
    orig_count_false = sum(
        [1 for x in data_loader.data[("predicates", predicate, "bool")] if x == False]
    )
    data = data_loader.data[("predicates", predicate, "robustness")]
    count_true = sum([1 for x in data if x > 0])
    count_false = sum([1 for x in data if x < 0])
    count_zero = sum([1 for x in data if x == 0])
    math.isclose(count_true, orig_count_true, rel_tol=0.02, abs_tol=3) or _LOGGER.warning(
        f"Predicate {predicate} has different count_true: {count_true} vs {orig_count_true}"
    )
    math.isclose(count_false, orig_count_false, rel_tol=0.02, abs_tol=3) or _LOGGER.warning(
        f"Predicate {predicate} has different count_false: {count_false} vs {orig_count_false}"
    )
    analysis[predicate] = {
        "balance": (count_true - count_false) / (count_true + count_false),
        "count_true": count_true,
        "count_false": count_false,
        "count_zero": count_zero,
        "rob_pos_mean": np.mean([x for x in data if x > 0]) if count_true > 0 else np.nan,
        "rob_neg_mean": np.mean([x for x in data if x < 0]) if count_false > 0 else np.nan,
        "rob_all_mean": np.mean(data),
        "rob_pos_std": np.std([x for x in data if x > 0]) if count_true > 0 else np.nan,
        "rob_neg_std": np.std([x for x in data if x < 0]) if count_false > 0 else np.nan,
        "rob_all_std": np.std(data),
        "rob_pos_span": np.ptp([x for x in data if x > 0]) if count_true > 0 else 0,
        "rob_neg_span": np.ptp([x for x in data if x < 0]) if count_false > 0 else 0,
        "rob_all_span": np.max(data) - np.min(data),
    }
    # balance
    abs(analysis[predicate]["balance"]) < 0.95 or _LOGGER.info(
        f"Unbalanced predicate {predicate}: {analysis[predicate]['balance']:.3f}"
    )
    # span
    # analysis[predicate]["rob_all_span"] > 0.3 or _LOGGER.info(
    #     f"Small span for predicate {predicate}: {analysis[predicate]['rob_all_span']:.3f}"
    # )
    # std
    # analysis[predicate]["rob_all_std"] > 0.1 or _LOGGER.info(
    #     f"Small std for predicate {predicate}: {analysis[predicate]['rob_all_std']:.3f}"
    # )
    analysis[predicate]["count_zero"] < (
        analysis[predicate]["count_true"] + analysis[predicate]["count_false"]
    ) * 0.01 or _LOGGER.info(
        f"Many zero values for predicate {predicate}: {analysis[predicate]['count_zero'] / (analysis[predicate]['count_true'] + analysis[predicate]['count_false']):.3f}"
    )

print(analysis)
