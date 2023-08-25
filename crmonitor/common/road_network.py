from typing import List, Set, Dict, Union

import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet, LaneletType
from commonroad_dc.geometry.util import chaikins_corner_cutting, resample_polyline
from commonroad_dc.geometry.geometry import CurvilinearCoordinateSystem
import commonroad_dc.pycrccosy as pycrccosy

from scipy.interpolate import splprep, splev

from commonroad.scenario.intersection import IntersectionIncomingElement

class Lane:
    """
    Lane representation build from several lanelets
    """

    def __init__(
        self,
        merged_lanelet: Lanelet,
        contained_lanelets: List[int],
        road_network_param: Dict,
    ):
        """
        :param merged_lanelet: lanelet element of lane
        :param contained_lanelets: lanelets lane consists of
        :param road_network_param: dictionary with parameters for the road network
        """
        self._lanelet = merged_lanelet
        self._contained_lanelets = set(contained_lanelets)
        self.lane_id = int("".join((str(i) for i in self._contained_lanelets)))
        if "large_resampling_step" in road_network_param.keys():
            # intersection
            # TODO: currently only consider AAH1 map
            if 7 in contained_lanelets:
                weight_left = 5
                smooth_factor_left = 1.5
            else:
                weight_left = 12
                smooth_factor_left = 1.5
            self.clcs_left, new_left_vertices, self.clcs_left_large_step, left_vertices_resample_large_step = self._create_clcs_from_reference(
                merged_lanelet.left_vertices, weight=weight_left, smooth_factor=smooth_factor_left, road_network_param=road_network_param
            )
            if 0 in contained_lanelets or 1 in contained_lanelets or 2 in contained_lanelets or 3 in contained_lanelets:
                weight_right = 25
                smooth_factor_right = 1.5
            else:
                weight_right = 12
                smooth_factor_right = 1.5
            self.clcs_right, new_right_vertices, self.clcs_right_large_step, right_vertices_resample_large_step = self._create_clcs_from_reference(
                merged_lanelet.right_vertices, weight=weight_right, smooth_factor=smooth_factor_right, road_network_param=road_network_param
            )
            self._clcs, new_center_vertices, self.clcs_large_step, _ = self._create_clcs_from_reference(
                merged_lanelet.center_vertices, weight=12, smooth_factor=1.5, road_network_param=road_network_param
            )

            self._orientation = self._compute_orientation_from_polyline(
                new_center_vertices
            )
            self._curvature = self._compute_curvature_from_polyline(
                new_center_vertices
            )
            self._path_length = self._compute_path_length_from_polyline(
                new_center_vertices
            )
            if road_network_param.get("map_type") == "hand_draft":
                self._width = self._compute_width_from_lanalet_boundary(
                    merged_lanelet.left_vertices, merged_lanelet.right_vertices
                )
            else:
                self._width = self._compute_width_from_lanalet_boundary(
                    new_left_vertices, new_right_vertices
                )

            self._adj_left = None
            self._adj_right = None
        else:
            self.clcs_left = Lane.create_curvilinear_coordinate_system_from_reference(
                merged_lanelet.left_vertices, road_network_param
            )
            self.clcs_right = Lane.create_curvilinear_coordinate_system_from_reference(
                merged_lanelet.right_vertices, road_network_param
            )
            self._clcs = Lane.create_curvilinear_coordinate_system_from_reference(
                merged_lanelet.center_vertices, road_network_param
            )
            self._orientation = self._compute_orientation_from_polyline(
                merged_lanelet.center_vertices
            )
            self._curvature = self._compute_curvature_from_polyline(
                merged_lanelet.center_vertices
            )
            self._path_length = self._compute_path_length_from_polyline(
                merged_lanelet.center_vertices
            )
            self._width = self._compute_width_from_lanalet_boundary(
                merged_lanelet.left_vertices, merged_lanelet.right_vertices
            )

            self._adj_left = None
            self._adj_right = None
        

    def __lt__(self, other):
        assert isinstance(other, Lane)
        return tuple(sorted(self.contained_lanelets)) < tuple(
            sorted(other.contained_lanelets)
        )

    @property
    def lanelet(self) -> Lanelet:
        return self._lanelet

    @property
    def contained_lanelets(self) -> Set[int]:
        return self._contained_lanelets

    @property
    def clcs(self) -> CurvilinearCoordinateSystem:
        return self._clcs

    def orientation(self, position) -> float:
        """
        Calculates orientation of lane given a longitudinal position along lane

        :param position: longitudinal position
        :returns orientation of lane at a given position
        """
        return np.interp(position, self._path_length, self._orientation)

    def width(self, s_position: float) -> float:
        """
        Calculates width of lane given a longitudinal position along lane

        :param s_position: longitudinal position
        :returns width of lane at a given position
        """
        return np.interp(s_position, self._path_length, self._width)

    @property
    def adj_left(self):
        return self._adj_left

    @property
    def adj_right(self):
        return self._adj_right

    def set_adj_lanes(self, adj_left=None, adj_right=None):
        self._adj_left = adj_left
        self._adj_right = adj_right

    @staticmethod
    def _compute_orientation_from_polyline(polyline: np.ndarray) -> np.ndarray:
        """
        Computes orientation along a polyline

        :param polyline: polyline for which orientation should be calculated
        :return: orientation along polyline
        """
        assert (
            isinstance(polyline, np.ndarray)
            and len(polyline) > 1
            and polyline.ndim == 2
            and len(polyline[0, :]) == 2
        ), "<Math>: not a valid polyline. polyline = {}".format(polyline)
        if len(polyline) < 2:
            raise ValueError("Cannot create orientation from polyline of length < 2")

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
        assert (
            isinstance(polyline, np.ndarray)
            and polyline.ndim == 2
            and len(polyline[:, 0]) > 2
        ), "Polyline malformed for curvature computation p={}".format(polyline)

        x_d = np.gradient(polyline[:, 0])
        x_dd = np.gradient(x_d)
        y_d = np.gradient(polyline[:, 1])
        y_dd = np.gradient(y_d)

        return (x_d * y_dd - x_dd * y_d) / ((x_d**2 + y_d**2) ** (3.0 / 2.0))

    @staticmethod
    def _compute_path_length_from_polyline(polyline: np.ndarray) -> np.ndarray:
        """
        Computes the path length of a polyline

        :param polyline: polyline for which path length should be calculated
        :return: path length along polyline
        """
        assert (
            isinstance(polyline, np.ndarray)
            and polyline.ndim == 2
            and len(polyline[:, 0]) > 2
        ), "Polyline malformed for pathlenth computation p={}".format(polyline)

        distance = np.zeros((len(polyline),))
        for i in range(1, len(polyline)):
            distance[i] = distance[i - 1] + np.linalg.norm(
                polyline[i] - polyline[i - 1]
            )

        return np.array(distance)

    @staticmethod
    def _compute_width_from_lanalet_boundary(
        left_polyline: np.ndarray, right_polyline: np.ndarray
    ) -> np.ndarray:
        """
        Computes the width of a lanelet

        :param left_polyline: left boundary of lanelet
        :param right_polyline: right boundary of lanelet
        :return: width along lanelet
        """
        width_along_lanelet = np.zeros((len(left_polyline),))
        for i in range(len(left_polyline)):
            width_along_lanelet[i] = np.linalg.norm(
                left_polyline[i] - right_polyline[i]
            )
        return width_along_lanelet

    @staticmethod
    def create_curvilinear_coordinate_system_from_reference(
        ref_path: np.array, road_network_param: Dict
    ) -> CurvilinearCoordinateSystem:
        """
        Generates curvilinear coordinate system for a reference path

        :param ref_path: reference path (polyline)
        :param road_network_param: dictionary containing parameters of the road network
        :returns curvilinear coordinate system for reference path
        """
        new_ref_path = ref_path
        for i in range(0, road_network_param.get("num_chankins_corner_cutting")):
            new_ref_path = chaikins_corner_cutting(new_ref_path)
        new_ref_path = resample_polyline(
            new_ref_path, road_network_param.get("polyline_resampling_step")
        )

        curvilinear_cosy = CurvilinearCoordinateSystem(new_ref_path, 20, 0.1, 5.0)

        return curvilinear_cosy

    def _create_clcs_from_reference(self, ref_path: np.ndarray, weight, smooth_factor, road_network_param):
        if road_network_param.get("map_type") == "hand_draft":
            reference_path_smooth = resample_polyline(ref_path, road_network_param.get("polyline_resampling_step"))
        else:
            reference_path = self._extrapolate_resample_polyline(ref_path)
            reference_path_smooth = self._smoothing_reference_path(reference_path, smooth_factor=smooth_factor, weight_coefficient=weight)

        curvilinear_cosy = CurvilinearCoordinateSystem(
            reference_path_smooth,
            road_network_param.get("lateral_projection_domain_limit"),
            road_network_param.get("lateral_eps"),
        )

        ref_path_resample_large_step = resample_polyline(
            reference_path_smooth, road_network_param.get("large_resampling_step")
        )
        curvilinear_cosy_large_step = CurvilinearCoordinateSystem(
            ref_path_resample_large_step,
            road_network_param.get("lateral_projection_domain_limit"),
            road_network_param.get("lateral_eps"),
        )
        return curvilinear_cosy, reference_path_smooth, curvilinear_cosy_large_step, ref_path_resample_large_step

    @staticmethod
    def _smoothing_reference_path(reference_path: np.ndarray, smooth_factor=None, weight_coefficient=None):
        # generate a smooth reference path
        transposed_reference_path = reference_path.T
        # how to generate index okay
        okay = np.where(
            np.abs(np.diff(transposed_reference_path[0]))
            + np.abs(np.diff(transposed_reference_path[1]))
            > 0
        )
        xp = np.r_[transposed_reference_path[0][okay], transposed_reference_path[0][-1]]
        yp = np.r_[transposed_reference_path[1][okay], transposed_reference_path[1][-1]]

        curvature = pycrccosy.Util.compute_curvature(np.array([xp, yp]).T)
        # set weights for interpolation:
        # see details: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html
        weights = np.exp(-weight_coefficient * (abs(curvature) - np.min(abs(curvature))))
        # B spline interpolation
        tck, u = splprep([xp, yp], s=smooth_factor, w=weights)
        u_new = np.linspace(u.min(), u.max(), 2000)
        x_new, y_new = splev(u_new, tck, der=0)
        ref_path_smooth = np.array([x_new, y_new]).transpose()
        return ref_path_smooth

    @staticmethod
    def _extrapolate_resample_polyline(
        polyline: np.ndarray, step: float = 2.0
        ) -> np.ndarray:
        # extend start point
        p = np.poly1d(np.polyfit(polyline[:2, 0], polyline[:2, 1], 1))

        x = 2 * polyline[0, 0] - polyline[1, 0]
        a = np.array([[x, p(x)]])
        polyline = np.concatenate((a, polyline), axis=0)

        # extend end point
        # extrapolate final point
        p = np.poly1d(np.polyfit(polyline[-2:, 0], polyline[-2:, 1], 1))

        # x = 2 * polyline[-1, 0] - polyline[-2, 0]
        # this extension helps the ego vehicle can drive to the end of the lane.
        x = polyline[-1, 0] + 99 * (polyline[-1, 0] - polyline[-2, 0])
        a = np.array([[x, p(x)]])
        polyline_extend = resample_polyline(np.concatenate((polyline[-1, np.newaxis], a), axis=0), step=20.0)
        polyline_origin = resample_polyline(polyline, step=step)

        return np.concatenate((polyline_origin, polyline_extend[1:, :]), axis=0)


class RoadNetwork:
    """
    Representation of the complete road network of a CommonRoad scenario abstracted to lanes
    """

    def __init__(self, lanelet_network: LaneletNetwork, road_network_param: Dict, scenario_type="interstate"):
        """
        :param lanelet_network: CommonRoad lanelet network
        :param road_network_param: dictionary with parameters for the road network
        """
        self.lanelet_network = lanelet_network
        self.scenario_type = scenario_type
        self.lanes = self._create_lanes(road_network_param)
        if len(lanelet_network.intersections) != 0:
            self.incoming = self._create_incoming_dict(lanelet_network)
            self.lanes_incoming = self._create_lanes_of_incoming(lanelet_network)
        else:
            self.incoming = {}
            self.lanes_incoming = {}

    def _create_lanes(self, road_network_param: Dict) -> List[Lane]:
        """
        Creates lanes for road network

        :param road_network_param: dictionary with parameters for the road network
        """
        lanes = []
        lane_lanelets = []
        start_lanelets = self._get_start_lanelets(self.lanelet_network)
        for lanelet in start_lanelets:
            if LaneletType.ACCESS_RAMP in lanelet.lanelet_type:
                lanelet_type = LaneletType.ACCESS_RAMP
            elif LaneletType.EXIT_RAMP in lanelet.lanelet_type:
                lanelet_type = LaneletType.EXIT_RAMP
            elif LaneletType.MAIN_CARRIAGE_WAY in lanelet.lanelet_type:
                lanelet_type = LaneletType.MAIN_CARRIAGE_WAY
            else:
                lanelet_type = None
            (
                merged_lanelets,
                merge_jobs,
            ) = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                lanelet,
                self.lanelet_network,
                road_network_param.get("merging_length"),
            )
            if len(merged_lanelets) == 0 or len(merge_jobs) == 0:
                merged_lanelets.append(lanelet)
                merge_jobs.append([lanelet.lanelet_id])
            for idx in range(len(merged_lanelets)):
                lane_lanelets.append((merged_lanelets[idx], merge_jobs[idx]))
        for lane_element in lane_lanelets:
            lanes.append(Lane(lane_element[0], lane_element[1], road_network_param))

        lanes.sort(key=lambda x: x.lane_id)

        # todo: the adjacency assignments only work for highway so far. For intersections, more dedicated approach
        #  is needed
        if len(lanes) == 0:
            pass
        elif len(lanes) == 1:
            lanes[0].set_adj_lanes(None, None)
        elif len(lanes) == 2:
            lanes[0].set_adj_lanes(lanes[1], None)
            lanes[-1].set_adj_lanes(None, lanes[-2])
        else:
            lanes[0].set_adj_lanes(lanes[1], None)
            lanes[-1].set_adj_lanes(None, lanes[-2])
            for k in range(1, len(lanes) - 1):
                lanes[k].set_adj_lanes(lanes[k + 1], lanes[k - 1])

        return lanes

    def _get_start_lanelets(self, lanelet_network: LaneletNetwork) -> List[Lanelet]:
        start_lanelets = []
        for lanelet in lanelet_network.lanelets:
            # only consider the longest lanes in intersection
            if len(lanelet.predecessor) == 0:
                start_lanelets.append(lanelet)
            elif self.scenario_type == "interstate":
                predecessors = [
                    self.lanelet_network.find_lanelet_by_id(pred_id)
                    for pred_id in lanelet.predecessor
                ]
                for pred in predecessors:
                    if not lanelet.lanelet_type == pred.lanelet_type:
                        start_lanelets.append(lanelet)
        return start_lanelets

    @staticmethod
    def _create_incoming_dict(lanelet_network: LaneletNetwork) -> Dict[int, IntersectionIncomingElement]:
        incoming_dict = {}
        for incoming_element in lanelet_network.intersections[0].incomings:
            incoming_dict[incoming_element.incoming_id] = incoming_element
        return incoming_dict

    def _create_lanes_of_incoming(self, lanelet_network: LaneletNetwork) -> Dict[int, List[Lane]]:
        lanes_incoming = {}
        for intersection in lanelet_network.intersections:
            for incoming in intersection.incomings:
                lanes_incoming[incoming.incoming_id] = [
                    self.get_turning_lane_from_incoming(self.lanes, incoming, "right"),
                    self.get_turning_lane_from_incoming(self.lanes, incoming, "straight"),
                    self.get_turning_lane_from_incoming(self.lanes, incoming, "left")]
        return lanes_incoming

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

        :param lanelet_id: CommonRoad lanelet ID
        :returns lane object
        """
        for lane in self.lanes:
            if lanelet_id in lane.contained_lanelets:
                return lane

    def find_lane_by_obstacle(
        self, obs_lanelet_center: List[int], obs_lanelet_shape: List[int]
    ) -> Lane:
        """
        Finds the lanes an obstacle occupies

        :param obs_lanelet_center: IDs of lanelet the obstacle center is on
            (use only first one)
        :param obs_lanelet_shape: IDs of lanelet the obstacle shape is on
        :returns lane the obstacle center is on
        """

        occupied_lanes = set()
        lanelets_center_updated = obs_lanelet_center
        obs_lanelet_shape_updated = obs_lanelet_shape
        if len(obs_lanelet_center) > 0:
            for lane in self.lanes:
                for lanelet in lanelets_center_updated:
                    if lanelet in lane.contained_lanelets:
                        occupied_lanes.add(lane)
        else:
            for lane in self.lanes:
                for lanelet in obs_lanelet_shape_updated:
                    if lanelet in lane.contained_lanelets:
                        occupied_lanes.add(lane)
        if len(occupied_lanes) == 1:
            return list(occupied_lanes)[0]
        for lane in occupied_lanes:
            for lanelet_id in lane.contained_lanelets:
                if (
                    LaneletType.MAIN_CARRIAGE_WAY
                    in self.lanelet_network.find_lanelet_by_id(lanelet_id).lanelet_type
                ):
                    return lane
        return list(occupied_lanes)[0]

    def find_lanes_incoming_by_id(self, incoming_id: int) -> "List[Lane]":
        return self.lanes_incoming[incoming_id]

    def lanelet_reach_suc(self, lanelet_id: int) -> "np.array":
        paths = self.lanes_suc(lanelet_id)
        paths = [l_id for path in paths for l_id in path]
        return np.unique(paths)

    def lanes_suc(self, lanelet_id: int) -> "List[List[int]]":
        lanelet_network = self.lanelet_network
        lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
        lanelet_ids = set()
        lanelet_ids.add(lanelet_id)
        successors = lanelet.successor
        if len(successors) == 0:
            return [[lanelet_id]]
        paths = []
        for suc in successors:
            suc_lanelet = lanelet_network.find_lanelet_by_id(suc)
            suc_paths = self.lanes_suc(suc_lanelet.lanelet_id)
            for suc_path in suc_paths:
                suc_path.insert(0, suc)
                paths.append(list(lanelet_ids.union(set(suc_path))))
        return paths

    def lanelet_reach_pre(self, lanelet_id: int) -> "np.array":
        paths = self.lanes_pre(lanelet_id)
        paths = [l_id for path in paths for l_id in path]
        return np.unique(paths)

    def lanes_pre(self, lanelet_id: int) -> "List[List[int]]":
        lanelet_network = self.lanelet_network
        lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
        lanelet_ids = set()
        lanelet_ids.add(lanelet_id)
        predecessors = lanelet.predecessor
        if len(predecessors) == 0:
            return [[lanelet_id]]
        paths = []
        for pre in predecessors:
            pre_lanelet = lanelet_network.find_lanelet_by_id(pre)
            pre_paths = self.lanes_pre(pre_lanelet.lanelet_id)
            for pre_path in pre_paths:
                pre_path.insert(0, pre)
                paths.append(list(lanelet_ids.union(set(pre_path))))
        return paths

    def find_incoming_intersection(self, lanelets_dir: "List[int]") -> "IntersectionIncomingElement":
        # TODO: further check needed
        possible_incomings = list()
        lanelet_pre = self.lanelet_reach_pre(lanelets_dir[0])
        lanelet_suc = self.lanelet_reach_suc(lanelets_dir[-1])
        possible_occupied_lanelets = lanelets_dir + list(lanelet_pre) + list(lanelet_suc)
        for incoming_element in self.lanelet_network.intersections[0].incomings:
            if len(incoming_element.incoming_lanelets.intersection(set(possible_occupied_lanelets))) > 0:
                possible_incomings.append(incoming_element)
        if len(possible_incomings) == 0:
            return None
        elif len(possible_incomings) == 1:
            return possible_incomings[0]
        else:
            # more than one incoming is searched, assume vehicle drives straight.
            # if no straight goning lane is matched, select the first incoming
            for incoming in possible_incomings:
                straight_going_lane = self.get_turning_lane_from_incoming(self.lanes, incoming, "straight")
                if len(straight_going_lane.contained_lanelets.intersection(possible_occupied_lanelets)) != 0:
                    return incoming
            return possible_incomings[0]

    @staticmethod
    def get_turning_lane_from_incoming(
            lanes: "List[Lane]",
            incoming: IntersectionIncomingElement,
            turning_direction: str,
    ) -> "Lane":
        incoming_lanelets_ids = incoming.incoming_lanelets
        if turning_direction == "right":
            turning_lanelets_ids = incoming.successors_right
        elif turning_direction == "left":
            turning_lanelets_ids = incoming.successors_left
        elif turning_direction == "straight":
            turning_lanelets_ids = incoming.successors_straight
        else:
            assert False, "turning_direction should be named right, left or straight"
        possible_lanes = list()
        for lane in lanes:
            # choose the lane which contains both incoming and right-turning successors
            if (len(lane.contained_lanelets.intersection(turning_lanelets_ids)) > 0
                    and len(lane.contained_lanelets.intersection(incoming_lanelets_ids)) > 0):
                possible_lanes.append(lane)
        selected_lanes = list()
        # find the longest lane
        # TODO: can separate in a new function
        for lane in possible_lanes:
            subset_find = False
            for index, selected_lane in enumerate(selected_lanes):
                if set(selected_lane.contained_lanelets).issubset(lane.contained_lanelets):
                    subset_find = True
                    selected_lanes[index] = lane
                    break
                elif set(lane.contained_lanelets).issubset(selected_lane.contained_lanelets):
                    subset_find = True
                    break
                else:
                    subset_find = False
            if not subset_find:
                selected_lanes.append(lane)
        return selected_lanes[0]

    def get_lanelets_start_end_s(self, lanelets_id: "Union[List, Set]", reference_lane: "Lane") -> (float, float):
        lanelets_start_s = np.inf
        lanelets_end_s = -np.inf
        for lanelet_id in lanelets_id:
            lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_id)
            # TODO: check: now assume start and end lines vertical to reference lane
            start_s = reference_lane.clcs.convert_to_curvilinear_coords(*lanelet.right_vertices[0, :])[0]
            end_s = reference_lane.clcs.convert_to_curvilinear_coords(*lanelet.right_vertices[-1, :])[0]
            lanelets_start_s = min(lanelets_start_s, start_s)
            lanelets_end_s = max(lanelets_end_s, end_s)
        return lanelets_start_s, lanelets_end_s
