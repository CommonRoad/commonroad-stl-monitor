# List of all predicates that are needed for
all_general_predicates = [
    "in_front_of",
    "in_same_lane",
    "cut_in",
    "keeps_safe_distance_prec",
    "brakes_abruptly",
    "brakes_abruptly_relative",
    "precedes",  # missing; problematic → all other vehicles must be considered; not too important
    "single_lane",  # missing
    "keeps_lane_speed_limit",
    "keeps_type_speed_limit",
    "keeps_brake_speed_limit",
    "keeps_fov_speed_limit",
    "keeps_lane_speed_limit_star",  # missing
    "slow_leading_vehicle",  # missing; problematic → all other vehicles must be considered
    "preserves_traffic_flow",  # missing
    "slow_as_leading_vehicle",
]

# all missing
all_interstate_predicates = [
    "in_congestion",  # problematic → all other vehicles must be considered
    "exist_standing_leading_vehicle",  # problematic → all other vehicles must be considered
    "in_standstill",
    "left_of",
    "drives_faster",
    "in_slow_moving_traffic",  # problematic → all other vehicles must be considered
    "in_queue_of_vehicles",  # problematic → all other vehicles must be considered
    "drives_with_slightly_higher_speed",
    "right_of_broad_lane_marking",  # problematic → missing lane information
    "left_of_broad_lane_marking",  # problematic → missing lane information
    "on_access_ramp",  # problematic → missing lane information
    "on_main_carriage_way",  # problematic → missing lane information
    "makes_u_turn",  # input features sufficient? → should be fine; only limitation: we just consider one lanelet instead of possibly multiple lanelets. For our use case, this will be fine.
    "reverses",
    "interstate_broad_enough",
    "on_shoulder",  # problematic → missing lane information
    "in_leftmost_lane",  # problematic → missing lane information
    "drives_leftmost",
    "drives_rightmost",
    "in_rightmost_lane",  # problematic → missing lane information
    "main_carriageway_right_lane",  # problematic → missing lane information
    "velocity_below_2",
    "velocity_below_5",
    "velocity_below_15",
    "velocity_below_20",
    "has_congestion_velocity",
    "has_slow_moving_velocity",
    "has_queue_velocity",
    "close_to_left_bound",
    "close_to_right_bound",
    "close_to_vehicle_left",
    "close_to_vehicle_right",
]

insufficient = [
    "cut_in",
    "brakes_abruptly_relative",
    # "has_slow_moving_velocity",
    # "reverses",
    # "drives_faster",
    # "drives_with_slightly_higher_speed",
    # "preserves_traffic_flow",
    # "drives_leftmost",
    # "drives_rightmost",
    # "slow_as_leading_vehicle",
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
