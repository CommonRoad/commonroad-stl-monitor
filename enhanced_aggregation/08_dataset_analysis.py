import numpy as np

from pathlib import Path

from commonroad_mpr.learning import DataLoader
from crmonitor.predicate_grouping import all_general_predicates, yuanfei_predicates

import logging

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

learning_data_path = Path(__file__).parent.parent / "output" / "learning_data" / "learning_data_yuanfei.csv"

selected_predicates = yuanfei_predicates
data_loader = DataLoader.create_from_file(learning_data_path)

# check mean, std, and span
for predicate in selected_predicates:
    data = data_loader.data[("predicates", predicate, "normalized_robustness")]
    _LOGGER.info(f"Predicate {predicate}: mean={np.mean(data):.3f}, std={np.std(data):.3f}, span={np.max(data)-np.min(data):.3f}")

# check velocity stuff
velocity_results = {}
velocities = {
    "has_slow_moving_velocity": 30,
    "velocity_below_20": 20,
    "velocity_below_15": 15,
    "has_queue_velocity": 10.8,  # 60 according to Maierhofer paper # TODO fix
    "has_congestion_velocity": 10,
    "velocity_below_5": 5,
    "velocity_below_2": 2,
    "in_standstill": 0.01,
    "reverses": -0.01,
}
for predicate in [
    "reverses",
    "in_standstill",
    # "velocity_below_2",
    # "velocity_below_5",
    "has_congestion_velocity",
    "has_queue_velocity",
    # "velocity_below_15",
    # "velocity_below_20",
    "has_slow_moving_velocity"
]:
    data = data_loader.data[("predicates", predicate, "bool")]
    velocity_results[predicate] = sum([1 for x in data if x==True]) / len(data)

#  data_loader.data[[("inputs", "ego", "velocity"), ("predicates", "reverses", "bool")]]

print(velocity_results)