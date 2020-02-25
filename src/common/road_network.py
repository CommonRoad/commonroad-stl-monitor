from pycrccosy import CurvilinearCoordinateSystem
from typing import Tuple, List, Set, Dict, Union
import numpy as np

from commonroad.scenario.lanelet import LaneletNetwork, Lanelet, LaneletType
from commonroad.scenario.trajectory import State
from commonroad_ccosy.geometry.util import chaikins_corner_cutting, resample_polyline

from src.common.vehicle import StateLongitudinal, StateLateral


class Lane:
    """
    Lane representation build from several lanelets
    """
    def __init__(self, merged_lanelet: Lanelet, contained_lanelets: List[int], road_network_param: Dict):
        """
        :param merged_lanelet: lanelet element of lane
        :param contained_lanelets: lanelets lane consists of
        :param road_network_param: dictionary with parameters for the road network
        """
        self._lanelet = merged_lanelet
        self._road_network_param = road_network_param
        self._clcs = self._create_curvilinear_coordinate_system_from_lanelet(merged_lanelet.center_vertices)
        self._contained_lanelets = set(contained_lanelets)
        self._orientation = self._compute_orientation_from_polyline(merged_lanelet.center_vertices)
        self._curvature = self._compute_curvature_from_polyline(merged_lanelet.center_vertices)
        self._path_length = self._compute_path_length_from_polyline(merged_lanelet.center_vertices)

    @property
    def lanelet(self) -> Lanelet:
        return self._lanelet

    @property
    def contained_lanelets(self) -> Set[int]:
        return self._contained_lanelets

    def orientation(self, position) -> float:
        """
        Calculates orientation of lane given a longitudinal position along lane

        :param position: longitudinal position
        :returns orientation of lane at a given position
        """
        return np.interp(position, self._path_length, self._orientation)

    def width(self, position: float) -> float:
        """
        Calculates width of lane given a longitudinal position along lane

        :param position: longitudinal position
        :returns width of lane at a given position
        """
        vertice_idx = []
        for idx, length in enumerate(self._path_length):
            if position < length:
                vertice_idx = [idx - 1, idx]
                break
        s_left_1, d_left_1 = self._clcs.convert_to_curvilinear_coords(self._lanelet.left_vertices[vertice_idx[0]][0],
                                                               self._lanelet.left_vertices[vertice_idx[0]][1])
        s_left_2, d_left_2 = self._clcs.convert_to_curvilinear_coords(self._lanelet.left_vertices[vertice_idx[1]][0],
                                                               self._lanelet.left_vertices[vertice_idx[1]][1])
        s_right_1, d_right_1 = self._clcs.convert_to_curvilinear_coords(self._lanelet.right_vertices[vertice_idx[0]][0],
                                                               self._lanelet.right_vertices[vertice_idx[0]][1])
        s_right_2, d_right_2 = self._clcs.convert_to_curvilinear_coords(self._lanelet.right_vertices[vertice_idx[1]][0],
                                                               self._lanelet.right_vertices[vertice_idx[1]][1])

        points = [(s_left_1, d_left_1), (s_left_2, d_left_2)]
        x_coords, y_coords = zip(*points)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m_left, c_left = np.linalg.lstsq(A, y_coords, rcond=None)[0]
        points = [(s_right_1, d_right_1), (s_right_2, d_right_2)]
        x_coords, y_coords = zip(*points)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m_right, c_right= np.linalg.lstsq(A, y_coords, rcond=None)[0]

        d_left = m_left * position + c_left
        d_right = m_right * position + c_right

        return abs(d_left - d_right)

    def _create_curvilinear_coordinate_system_from_lanelet(self, ref_path: np.array) -> CurvilinearCoordinateSystem:
        """
        Generates curvilinear coordinate system for a reference path

        :param ref_path: reference path (polyline)
        :returns curvilinear coordinate system for reference path
        """
        new_ref_path = np.array([])
        for i in range(0, self._road_network_param.get("num_chankins_corner_cutting")):
            new_ref_path = chaikins_corner_cutting(ref_path)
        new_ref_path = resample_polyline(new_ref_path, self._road_network_param.get("polyline_resampling_step"))

        curvilinear_cosy = CurvilinearCoordinateSystem(new_ref_path)
        return curvilinear_cosy

    def create_curvilinear_states(self, state: State) -> Union[Tuple[StateLongitudinal, StateLateral],
                                                               Tuple[None, None]]:
        """
        Computes initial state of ego vehicle

        :param state: CommonRoad state
        :return: lateral and longitudinal state of vehicle
        """
        try:
            s, d = self._clcs.convert_to_curvilinear_coords(state.position[0], state.position[1])
        except ValueError:
            #print("Vehicle out of projection domain: State will not be considered")
            return None, None
        theta_cl = np.interp(s, self._path_length, self._orientation)
        if hasattr(state, "acceleration") and hasattr(state, "jerk"):
            x_lon = StateLongitudinal(s, state.velocity, state.acceleration, state.jerk)
        elif hasattr(state, "acceleration"):
            x_lon = StateLongitudinal(s, state.velocity, state.acceleration, None)
        else:
            x_lon = StateLongitudinal(s, state.velocity, None, None)
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
    """
    Representation of the complete road network of a CommonRoad scenario abstracted to lanes
    """
    def __init__(self, lanelet_network: LaneletNetwork, road_network_param: Dict):
        """
        :param lanelet_network: CommonRoad lanelet network
        :param road_network_param: dictionary with parameters for the road network
        """
        self.lanelet_network = lanelet_network
        self.lanes = self._create_lanes(road_network_param)

    def _create_lanes(self, road_network_param: Dict) -> List[Lane]:
        """
        Creates lanes for road network

        :param road_network_param: dictionary with parameters for the road network
        """
        lanes = []
        lane_lanelets = []
        start_lanelets = []
        for lanelet in self.lanelet_network.lanelets:
            if len(lanelet.predecessor) == 0:
                start_lanelets.append(lanelet)
            for pred in lanelet.predecessor:
                if len(self.lanelet_network.find_lanelet_by_id(pred).successor) > 1:
                    if lanelet.adj_left_same_direction is None and lanelet.adj_right_same_direction is None:
                        start_lanelets.append(lanelet)
                    if self.lanelet_network.find_lanelet_by_id(pred).adj_left_same_direction and lanelet.adj_left not \
                            in self.lanelet_network.find_lanelet_by_id(
                               self.lanelet_network.find_lanelet_by_id(pred).adj_left).successor and \
                            self.lanelet_network.find_lanelet_by_id(pred).adj_right_same_direction and \
                            lanelet.adj_right not in self.lanelet_network.find_lanelet_by_id(
                               self.lanelet_network.find_lanelet_by_id(pred).adj_right).successor:
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
                Lanelet.all_lanelets_by_merging_successors_from_lanelet(lanelet, self.lanelet_network,
                                                                        road_network_param.get("merging_length"),
                                                                        lanelet_type)
            if len(merged_lanelets) == 0 or len(merge_jobs) == 0:
                merged_lanelets.append(lanelet)
                merge_jobs.append([lanelet.lanelet_id])
            for idx in range(len(merged_lanelets)):
                lane_lanelets.append((merged_lanelets[idx], merge_jobs[idx]))
        for lane_element in lane_lanelets:
            lanes.append(Lane(lane_element[0], lane_element[1], road_network_param))

        return lanes

    def find_lane_ids_by_obstacle(self, obstacle_id: int, time_step: int) -> Set[int]:
        """
        Finds the lanes an obstacle belongs to and returns their IDs

        :param obstacle_id: ID of the obstacle
        :param time_step: time step of interest
        """
        lane_ids = set()
        for lane in self.lanes:
            if obstacle_id in lane.lanelet.dynamic_obstacle_by_time_step(time_step):
                lane_ids.add(lane.lanelet.lanelet_id)

        return lane_ids

    def find_lane_ids_by_lanelets(self, lanelets: Set[int]) -> Set[int]:
        """
        Finds the lanes given set of lanelets belong to and returns their IDs

        :param lanelets: list of lanelet IDs
        :returns set of lanelet IDs
        """
        lane_ids = set()
        for lane in self.lanes:
            for lanelet_id in lanelets:
                if lanelet_id in lane.contained_lanelets:
                    lane_ids.add(lane.lanelet.lanelet_id)

        return lane_ids

    def find_lanes_by_lanelets(self, lanelets: Set[int]) -> Set[Lane]:
        """
        Finds the lanes to which a given set of lanelets belongs to

        :param lanelets: list of lanelet IDs
        :returns set of lane objects
        """
        lanes = set()
        for lane in self.lanes:
            for lanelet_id in lanelets:
                if lanelet_id in lane.contained_lanelets:
                    lanes.add(lane)

        return lanes

    def find_lane_by_lanelet(self, lanelet_id: int) -> Lane:
        """
        Finds the lane a lanelet belongs to

        :param lanelet: CommonRoad lanelet ID
        :returns lane object
        """
        for lane in self.lanes:
            if lanelet_id in lane.contained_lanelets:
                return lane

    def find_lane_by_obstacle(self, obs_lanelet_center: List[int], obs_lanelet_shape: List[int]) -> Lane:
        """
        Finds the lanes an obstacle belongs to

        :param obs_lanelet_center: IDs of lanelet the obstacle center is on (use only first one)
        :param obs_lanelet_shape: IDs of lanelet the obstacle shape is on
        :returns lane the obstacle center is on
        """
        if len(obs_lanelet_center) > 0:
            for lane in self.lanes:
                if obs_lanelet_center[0] in lane.contained_lanelets:
                    return lane
        else:
            # if no lane is found, e.g. center on exterior of polygon usage of shape
            for lane in self.lanes:
                for lanelet in obs_lanelet_shape:
                    if lanelet in lane.contained_lanelets:
                        return lane
