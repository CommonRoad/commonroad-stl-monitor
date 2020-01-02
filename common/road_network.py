from commonroad.scenario.lanelet import LaneletNetwork, Lanelet, LaneletType
from commonroad.scenario.trajectory import State
from commonroad_ccosy.geometry.util import chaikins_corner_cutting, resample_polyline
import numpy as np
from common.vehicle import StateLongitudinal, StateLateral
from pycrccosy import SegmentCoordinateSystem
from typing import Tuple, List


class Lane:
    def __init__(self, merged_lanelet: Lanelet, contained_lanelets: List[int]):
        self.lanelet = merged_lanelet
        self.clcs = self._create_curvilinear_coordinate_system_from_lanelet(merged_lanelet.center_vertices)
        self.contained_lanelets = set(contained_lanelets)
        self.orientation = self._compute_orientation_from_polyline(merged_lanelet.center_vertices)
        self.curvature = self._compute_curvature_from_polyline(merged_lanelet.center_vertices)
        self.path_length = self._compute_path_length_from_polyline(merged_lanelet.center_vertices)

    @staticmethod
    def _create_curvilinear_coordinate_system_from_lanelet(ref_path: np.array) -> SegmentCoordinateSystem:
        """
        Generates curvilinear coordinate system for a reference path

        :param ref_path: reference path (polyline)
        :returns curvilinear coordinate system for reference path
        """
        new_ref_path = np.array([])
        for i in range(0, 250):
            new_ref_path = chaikins_corner_cutting(ref_path)
        new_ref_path = resample_polyline(new_ref_path, 0.1)

        curvilinear_cosy = SegmentCoordinateSystem(new_ref_path)
        return curvilinear_cosy

    def create_curvilinear_states(self, state: State) -> Tuple[StateLongitudinal, StateLateral]:
        """
        Computes initial state of ego vehicle

        :param state: CommonRoad state
        :return: lateral and longitudinal state of vehicle
        """
        s, d = self.clcs.convert_to_curvilinear_coords(state.position[0], state.position[1])
        theta_cl = np.interp(s, self.path_length, self.orientation)
        if hasattr(state, "acceleration"):
            x_lon = StateLongitudinal(s, state.velocity, state.acceleration)
        else:
            x_lon = StateLongitudinal(s, state.velocity, 0)
        x_lat = StateLateral(d, theta_cl - state.orientation, 0, 0)

        return x_lon, x_lat

    @staticmethod
    def _compute_orientation_from_polyline(polyline: np.ndarray) -> np.ndarray:
        """
        Computes orientation along a polyline

        :param polyline: polyline for which orientation should be calculated
        :return: orientation along polyline
        """
        assert isinstance(polyline, np.ndarray) and len(polyline) > 1 and polyline.ndim == 2 and len(
            polyline[0, :]) == 2, '<Math>: not a valid polyline. polyline = {}'.format(polyline)
        if len(polyline) < 2:
            raise ValueError('Cannot create orientation from polyline of length < 2')

        orientation = [0]
        for i in range(1, len(polyline)):
            pt1 = polyline[i - 1]
            pt2 = polyline[i]
            tmp = pt2 - pt1
            orientation.append(np.arctan2(tmp[1], tmp[0]))

        return np.array(orientation)

    @staticmethod
    def _compute_curvature_from_polyline(polyline: np.ndarray) -> np.ndarray:
        """
        Computes curvature along a polyline

        :param polyline: polyline for which curvature should be calculated
        :return: curvature along  polyline
        """
        assert isinstance(polyline, np.ndarray) and polyline.ndim == 2 and len(
            polyline[:, 0]) > 2, 'Polyline malformed for curvature computation p={}'.format(polyline)

        x_d = np.gradient(polyline[:, 0])
        x_dd = np.gradient(x_d)
        y_d = np.gradient(polyline[:, 1])
        y_dd = np.gradient(y_d)

        return (x_d * y_dd - x_dd * y_d) / ((x_d ** 2 + y_d ** 2) ** (3. / 2.))

    @staticmethod
    def _compute_path_length_from_polyline(polyline: np.ndarray) -> np.ndarray:
        """
        Computes the path length of a polyline

        :param polyline: polyline for which path length should be calculated
        :return: path length along polyline
        """
        assert isinstance(polyline, np.ndarray) and polyline.ndim == 2 and len(
            polyline[:, 0]) > 2, 'Polyline malformed for pathlenth computation p={}'.format(polyline)

        distance = np.zeros((len(polyline),))
        for i in range(1, len(polyline)):
            distance[i] = distance[i - 1] + np.linalg.norm(polyline[i] - polyline[i - 1])

        return np.array(distance)


class RoadNetwork:
    def __init__(self, lanelet_network: LaneletNetwork):
        self.lanelet_network = lanelet_network
        self.lanes = self._create_lanes()

    def _create_lanes(self) -> List[Lane]:
        lanes = []
        lane_lanelets = []
        start_lanelets = []
        for lanelet in self.lanelet_network.lanelets:
            if len(lanelet.predecessor) == 0:
                start_lanelets.append(lanelet)
        for lanelet in start_lanelets:
            if LaneletType.ACCESS_RAMP in lanelet.lanelet_type:
                lanelet_type = LaneletType.ACCESS_RAMP
            elif LaneletType.EXIT_RAMP in lanelet.lanelet_type:
                lanelet_type = LaneletType.EXIT_RAMP
            elif LaneletType.MAIN_CARRIAGE_WAY in lanelet.lanelet_type:
                lanelet_type = LaneletType.MAIN_CARRIAGE_WAY
            else:
                lanelet_type = None
            merged_lanelets, merge_jobs = \
                Lanelet.all_lanelets_by_merging_successors_from_lanelet(lanelet, self.lanelet_network, 10000.0,
                                                                        lanelet_type)
            if len(merged_lanelets) == 0 or len(merge_jobs) == 0:
                merged_lanelets.append(lanelet)
                merge_jobs.append([lanelet.lanelet_id])
            for idx in range(len(merged_lanelets)):
                lane_lanelets.append((merged_lanelets[idx], merge_jobs[idx]))
        for lane_element in lane_lanelets:
            lanes.append(Lane(lane_element[0], lane_element[1]))

        return lanes

    def find_lane(self, obstacle_id: int, time_step: int):
        for lane in self.lanes:
            if obstacle_id in lane.lanelet.dynamic_obstacle_by_time_step(time_step):
                return lane
