from pathlib import Path

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

balance = {}
for predicate in all_general_predicates + all_interstate_predicates:
    val_true = data[('predicates', predicate, 'bool')].sum()
    val_false = len(data) - val_true
    balance[predicate] = (val_true - val_false) / (val_true + val_false)

print([f"{k}: {v:.3f} \n" for k, v in balance.items()])