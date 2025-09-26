from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from commonroad_mpr.learning import DataLoader

path = Path(__file__).parent.parent / "output" / "learning_data" / "learning_data.csv"

data = DataLoader.create_from_file(path).data

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


def _compute_metrics_for_input(path: Tuple[str, ...], mask=None) -> dict:
    if mask is not None:
        values = np.array(data[mask][path])
    else:
        values = np.array(data[path])
    return {
        "var": np.var(values),
        "span": np.ptp(values),
        "max": np.max(values),
        "min": np.min(values),
        "mean": np.average(values),
    }


def compute_additional_metrics(predicate: str, inputs):
    """
    Compute and print some additional metrics about inputs, like variance or span.
    """
    metrics = {}
    for input_ in inputs:
        metrics[input_] = {}
        vals = data[input_]
        metrics[input_]["overall"] = _compute_metrics_for_input(input_)
        mask_false = ~mask
        metrics[input_][f"{predicate}_true"] = _compute_metrics_for_input(input_, mask)
        metrics[input_][f"{predicate}_false"] = _compute_metrics_for_input(input_, mask_false)
        overall = metrics[input_]["overall"]
        print(
            f"{predicate}: var={overall['var']:.4f}, span={overall['span']:.4f}, max={overall['max']:.4f}, min={overall['min']:.4f}"
        )

        true_metrics = metrics[input_][f"{predicate}_true"]
        false_metrics = metrics[input_][f"{predicate}_false"]

        print(
            f"{predicate} (TRUE): var={true_metrics['var']:.4f}, span={true_metrics['span']:.4f}, max={true_metrics['max']:.4f}, min={true_metrics['min']:.4f}, mean={true_metrics['mean']:.4f}, n={len(vals[mask])}"
        )
        print(
            f"{predicate} (FALSE): var={false_metrics['var']:.4f}, span={false_metrics['span']:.4f}, max={false_metrics['max']:.4f}, min={false_metrics['min']:.4f}, mean={false_metrics['mean']:.4f}, n={len(vals[mask_false])}"
        )
        print(
            "relative deviation:",
            (true_metrics["span"] - false_metrics["span"]) / overall["span"],
        )


def plot_input_robustness_relations(predicate_name: str, inputs):
    """
    Plot the robutsness values alongside the input values.
    """
    rows = int(len(inputs) / 2 + 0.6)
    _, axes = plt.subplots(rows, 2)
    for i, input_ in enumerate(inputs):
        ax = axes.flat[i]
        vals = data[input_]
        robs = data[("predicates", predicate_name, "robustness")]
        ax.scatter(vals, robs, s=1)
        ax.set_ylabel(input_[-1])
    plt.show()


# Select inputs that you want to investigate.
inputs = [
    ("inputs", "ego", "velocity"),
    ("inputs", "ego", "acceleration"),
    ("inputs", "ego", "position"),
    ("inputs", "ego", "curvature"),
    ("inputs", "ego_other", "distance"),
    ("inputs", "ego_other", "lateral_distance"),
    ("inputs", "ego", "width"),
    ("inputs", "ego", "length"),
    ("inputs", "ego", "distance_to_ref_lane_left"),
    ("inputs", "ego", "distance_to_ref_lane_right"),
    # ("inputs", "ego", "distance_to_road_left"),
    # ("inputs", "ego", "distance_to_road_right"),
]


for predicate in ["in_same_lane"]:
    mask = data[("predicates", predicate, "bool")]
    val_true = mask.sum()
    val_false = len(data) - val_true
    balance = (val_true - val_false) / (val_true + val_false)

    print(f"Balance {predicate}: {balance}")
