class VehicleClassification:

    def __init__(self):
        self._ego_lane_front = []
        self._left_lane = []
        self._right_lane = []
        self._right = []
        self._left = []

    def in_fov(self, s: float) -> bool:
        if s < self._ego_vehicle_param.get("fov"):
            return True
        else:
            return False

    @staticmethod
    def ego_same_lane_behind_other(s_ego: float, s_other: float, lanelet_ids_ego: Set[int],
                                   lanelet_ids_other: Set[int]) -> bool:
        if s_ego < s_other:
            for lanelet_id in lanelet_ids_ego:
                if lanelet_id in lanelet_ids_other:
                    return True
            return False
        else:
            return False