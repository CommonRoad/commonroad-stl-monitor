from typing import Dict, Set


class LanePredicateCollection:
    def __init__(self, ego_vehicle_param: Dict):
        """
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        """
        self._ego_vehicle_param = ego_vehicle_param

    @staticmethod
    def in_fov(s_other: float, s_ego, fov: float) -> bool:
        if abs(s_other - s_ego) < fov:
            return True
        else:
            return False

    @staticmethod
    def same_lane_behind_other(s_ego: float, s_other: float, lanelet_ids_ego: Set[int],
                               lanelet_ids_other: Set[int]) -> bool:
        if s_ego < s_other:
            for lanelet_id in lanelet_ids_ego:
                if lanelet_id in lanelet_ids_other:
                    return True
            return False
        else:
            return False
