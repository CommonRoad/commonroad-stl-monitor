# List of all predicates that are needed for
all_general_predicates = [
    "in_front_of",
    "in_same_lane",
    "keeps_safe_distance_prec",
    "brakes_abruptly",
    "brakes_abruptly_relative",
    "single_lane",
    "keeps_lane_speed_limit",
    "keeps_type_speed_limit",
    "keeps_brake_speed_limit",
    "keeps_fov_speed_limit",
    "keeps_lane_speed_limit_star",
    "preserves_traffic_flow",
    "slow_as_leading_vehicle",
]

all_interstate_predicates = [
    "in_standstill",
    "drives_faster",
    "drives_with_slightly_higher_speed",
    "right_of_broad_lane_marking",
    "left_of_broad_lane_marking",
    "on_access_ramp",  # problematic → missing lane information
    "on_main_carriage_way",
    "makes_u_turn",  # input features sufficient? → should be fine; only limitation: we just consider one lanelet instead of possibly multiple lanelets. For our use case, this will be fine.
    "reverses",
    "interstate_broad_enough",
    "on_shoulder",
    "in_leftmost_lane",
    "in_rightmost_lane",
    "main_carriageway_right_lane",
    "velocity_below_2",
    "velocity_below_5",
    "velocity_below_15",
    "velocity_below_20",
    "has_congestion_velocity",
    "has_slow_moving_velocity",
    "has_queue_velocity",
    "close_to_left_bound",
    "close_to_right_bound",
    "lat_left_of",
    "heading_right",
    "lat_left_of_vehicle",
    "rear_behind_front",
    "lat_close_to_vehicle_left",
    "lat_close_to_vehicle_right",
]

changed_to_meta = [
    "slow_leading_vehicle",
    "exist_standing_leading_vehicle",
    "in_congestion",
    "in_slow_moving_traffic",
    "in_queue_of_vehicles",
    "precedes",
    "drives_leftmost",
    "drives_rightmost",
    "cut_in",
    "left_of",
]

meta_only = [
    "lon_intersecting_vehicles",
    "close_to_vehicle_left",
    "close_to_vehicle_right",
    "approach_from_left",
    "approach_from_right",
]

yuanfei_predicates = [
    "in_front_of",
    "in_same_lane",
    "cut_in",
    "keeps_safe_distance_prec",
    "brakes_abruptly",
    "brakes_abruptly_relative",
    "single_lane",
    "keeps_lane_speed_limit",
    "keeps_type_speed_limit",
    "keeps_brake_speed_limit",
    "keeps_fov_speed_limit",
]

insufficient = [
    "lat_close_to_vehicle_left",
    "lat_close_to_vehicle_right",
    "brakes_abruptly_relative",
]
