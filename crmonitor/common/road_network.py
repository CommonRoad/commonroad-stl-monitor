from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union

import commonroad_clcs.pycrccosy as pycrccosy
import numpy as np
from commonroad.scenario.intersection import IntersectionIncomingElement
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork, LaneletType
from commonroad_clcs.clcs import CurvilinearCoordinateSystem
from commonroad_clcs.config import CLCSParams, ProcessingOption, ResamplingParams
from commonroad_clcs.util import (
    chaikins_corner_cutting,
    compute_orientation_from_polyline,
    compute_pathlength_from_polyline,
    resample_polyline,
)

from .scenario_type import ScenarioType


@dataclass
class RoadNetworkParam:
    num_chankins_corner_cutting: int = 1
    """Number of refinements used in chankins corner cutting when constructing curvilinear coordinate systems for lanes."""

    polyline_resampling_step: float = 0.5
    """Resampling step when constructing cuvrilinear coordinate systems for lanes."""

    large_resampling_step: float = 3.5
    """Larger resampling step used when constructing large step curvilinear coordinate systems for lanes."""

    merging_length: int = 10000
    lateral_projection_domain_limit: int = 80
    lateral_eps: float = 0.1


class VariableStepCurvilinearCoordinateSystem:
    """
    Wrapper around `CurvilinearCoordinateSystem`s to circumvent frequent projection domain issues.

    Use `VariableStepCurvilinearCoordinateSystem.create_from_reference_path` to construct this from a polyline, e.g., a lenelet vertex.
    """

    def __init__(
        self, clcs: CurvilinearCoordinateSystem, clcs_large_step: CurvilinearCoordinateSystem
    ) -> None:
        self._clcs = clcs
        self._clcs_large_step = clcs_large_step

    @property
    def clcs(self) -> CurvilinearCoordinateSystem:
        """The internal curvilinear coordinate system."""
        return self._clcs

    @property
    def clcs_large_step(self) -> CurvilinearCoordinateSystem:
        """The internal large step curvilinear coordinate system."""
        return self._clcs_large_step

    @classmethod
    def create_from_reference_path(
        cls, ref_path: np.ndarray, road_network_param: RoadNetworkParam
    ) -> "VariableStepCurvilinearCoordinateSystem":
        """
        Create a new variable step curvilinear coordinate system from a reference polyline, e.g. the vertices of a lanelet.

        :param ref_path: The reference polyline. The polyline will be resampled based on the steps set in `road_network_param`.
        :param road_network_param: Additional configuration for the path pre-processing and the curvilinear coordinate system.

        :returns: A new `VariableStepCurvilinearCoordinateSystem`.
        """
        new_ref_path = chaikins_corner_cutting(
            ref_path, road_network_param.num_chankins_corner_cutting
        )

        clcs = cls._create_variable_step_clcs_from_reference(
            new_ref_path, road_network_param.polyline_resampling_step, road_network_param
        )

        clcs_large_step = cls._create_variable_step_clcs_from_reference(
            new_ref_path, road_network_param.large_resampling_step, road_network_param
        )

        return cls(clcs, clcs_large_step)

    def convert_to_curvilinear_coords(self, x: float, y: float) -> tuple[float, float]:
        """
        Convert cartesian coordinates to curvilinear coordinates.

        Automatically tries to recover from projection issues, by falling back to curvilinear
        coordinate systems with large step size.

        :param x: Cartesian x coordinate.
        :param y: Cartesian y coordinate.

        :returns: Tuple with corresponding curvilinear coords: (s, d).

        :raises ValueError: If cartesian coordinates are outside of projection domain.
        """
        if self._clcs.cartesian_point_inside_projection_domain(x, y):
            return self._clcs.convert_to_curvilinear_coords(x, y)

        if self._clcs_large_step.cartesian_point_inside_projection_domain(x, y):
            return self._clcs_large_step.convert_to_curvilinear_coords(x, y)

        raise ValueError(
            f"Cartesian coordinates ({x},{y}) are outside the curvilinear projection domain"
        )

    def convert_to_cartesian_coords(self, s: float, d: float) -> tuple[float, float]:
        """
        Convert curvilinear coordinates to cartesian coordinates.

        Automatically tries to recover from projection issues, by falling back to curvilinear
        coordinate systems with large step size.

        :param s: Curvilinear s coordinate.
        :param d: Curvilinear d coordinate.

        :returns: Tuple with corresponding cartesian coords: (x, y).

        :raises ValueError: If curvilinear coordinates are outside of projection domain.
        """
        if self._clcs.curvilinear_point_inside_projection_domain(s, d):
            return self._clcs.convert_to_cartesian_coords(s, d)

        if self._clcs_large_step.curvilinear_point_inside_projection_domain(s, d):
            return self._clcs_large_step.convert_to_cartesian_coords(s, d)

        raise ValueError(
            f"Curvilinear coordinates ({s},{d}) are outside the cartesian projection domain"
        )

    @staticmethod
    def _create_variable_step_clcs_from_reference(
        ref_path: np.ndarray, resampling_step: float, road_network_param: RoadNetworkParam
    ) -> CurvilinearCoordinateSystem:
        """
        Create the internal curvilinear coordinate systems based on the reference path.

        :param ref_path: Reference polyline for the curvilinear coordinate system.
        :param resampling_step: Step with which the reference path is resampled.
        :param road_network_param: Additional parameters used to construct the curvilinear coordinate system.

        :returns: A new `CurvilinearCoordinateSystem` for the reference path.
        """
        # The path has to be pre-processed before the curvilinear coordinate system can be constructed.
        # `CurvilinearCoordinateSystem` enforces a maximum allowed orientation difference, which
        # is frequently exceeded due to the merging of lanelets.
        # By resampling the polyline we make sure that the polyline is valid.
        new_ref_path = resample_polyline(ref_path, resampling_step)

        curvilinear_cosy = CurvilinearCoordinateSystem(
            new_ref_path,
            CLCSParams(
                default_proj_domain_limit=road_network_param.lateral_projection_domain_limit,
                eps=road_network_param.lateral_eps,
                processing_option=ProcessingOption.ELASTIC_BAND,
                resampling=ResamplingParams(fixed_step=resampling_step),
            ),
            preprocess_path=True,
        )

        return curvilinear_cosy


class Lane:
    """
    Lane representation build from several lanelets
    """

    def __init__(
        self,
        merged_lanelet: Lanelet,
        contained_lanelets: List[int],
        road_network_param: Optional[RoadNetworkParam] = None,
        scenario_type: ScenarioType = ScenarioType.INTERSTATE,
    ):
        """
        :param merged_lanelet: lanelet element of lane
        :param contained_lanelets: lanelets lane consists of
        :param road_network_param: dictionary with parameters for the road network
        """
        self._lanelet = merged_lanelet
        self._contained_lanelets = set(contained_lanelets)
        self.lane_id = int("".join((str(i) for i in self._contained_lanelets)))

        if road_network_param is None:
            road_network_param = RoadNetworkParam()

        self._clcs_left = VariableStepCurvilinearCoordinateSystem.create_from_reference_path(
            merged_lanelet.left_vertices, road_network_param
        )
        self._clcs_right = VariableStepCurvilinearCoordinateSystem.create_from_reference_path(
            merged_lanelet.right_vertices, road_network_param
        )
        self._clcs = VariableStepCurvilinearCoordinateSystem.create_from_reference_path(
            merged_lanelet.center_vertices, road_network_param
        )

        self._orientation = compute_orientation_from_polyline(merged_lanelet.center_vertices)
        self._curvature = pycrccosy.Util.compute_curvature(merged_lanelet.center_vertices)
        self._path_length = compute_pathlength_from_polyline(merged_lanelet.center_vertices)
        self._width = self._compute_width_from_lanalet_boundary(
            merged_lanelet.left_vertices, merged_lanelet.right_vertices
        )

        self._adj_left = None
        self._adj_right = None
        self.center_vertices = None
        self.smoothed_vertices = None

    def __lt__(self, other):
        assert isinstance(other, Lane)
        return tuple(sorted(self.contained_lanelets)) < tuple(sorted(other.contained_lanelets))

    def __eq__(self, other):
        return self.lane_id == other.lane_id

    def __hash__(self) -> int:
        return hash(self.lane_id)

    @property
    def lanelet(self) -> Lanelet:
        return self._lanelet

    @property
    def contained_lanelets(self) -> Set[int]:
        return self._contained_lanelets

    @property
    def clcs(self) -> VariableStepCurvilinearCoordinateSystem:
        """The curvilinear coordinate system based on the center reference line."""
        return self._clcs

    @property
    def clcs_left(self) -> VariableStepCurvilinearCoordinateSystem:
        """The curvilinear coordinate system corresponding to the left lane boundary."""
        return self._clcs_left

    @property
    def clcs_right(self) -> VariableStepCurvilinearCoordinateSystem:
        """The curvilinear coordinate system corresponding to the right lane boundary."""
        return self._clcs_right

    def orientation(self, position) -> float:
        """
        Calculates orientation of lane given a longitudinal position along lane

        :param position: longitudinal position
        :returns: orientation of lane at a given position
        """
        return np.interp(position, self._path_length, self._orientation)

    def width(self, s_position: float) -> float:
        """
        Calculates width of lane given a longitudinal position along lane

        :param s_position: longitudinal position
        :returns: width of lane at a given position
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
            width_along_lanelet[i] = np.linalg.norm(left_polyline[i] - right_polyline[i])
        return width_along_lanelet

    def distance_to_left(self, x: float, y: float) -> float:
        """
        Distance of point to left boundary of lane.

        :param x: Cartesian x coordinate.
        :param y: Cartesian y coordinate.

        :returns: Distance to left boundary of lane, or -inf if the point is outside the projection domain.
        """
        try:
            _, d = self._clcs_left.convert_to_curvilinear_coords(x, y)
        except ValueError:
            return -np.inf

        return -d

    def distance_to_right(self, x: float, y: float) -> float:
        """
        Distance of point to right boundary of lane.

        :param x: Cartesian x coordinate.
        :param y: Cartesian y coordinate.

        :returns: Distance to right boundary of lane, or inf if the point is outside the projection domain.
        """
        try:
            _, d = self._clcs_right.convert_to_curvilinear_coords(x, y)
        except ValueError:
            return np.inf

        return d

    def min_max_distance_to_left(self, points: np.ndarray) -> tuple[float, float]:
        minimum = np.inf
        maximum = -np.inf
        for x, y in points:
            distance = self.distance_to_left(x, y)
            minimum = min(minimum, distance)
            maximum = max(maximum, distance)
        return minimum, maximum

    def min_max_distance_to_right(self, points: np.ndarray) -> tuple[float, float]:
        minimum = np.inf
        maximum = -np.inf
        for x, y in points:
            distance = self.distance_to_right(x, y)
            minimum = min(minimum, distance)
            maximum = max(maximum, distance)
        return minimum, maximum

    def convert_to_curvilinear_coords(self, x: float, y: float) -> tuple[float, float]:
        """
        Convert cartesian coordinates to curvilinear coordinates.

        :param x: Cartesian x coordinate.
        :param y: Cartesian y coordinate.

        :returns: Tuple with corresponding curvilinear coords: (s, d).

        :raises ValueError: If cartesian coordinates are outside of projection domain.
        """
        return self._clcs.convert_to_curvilinear_coords(x, y)

    def convert_to_cartesian_coords(self, s: float, d: float) -> tuple[float, float]:
        """
        Convert curvilinear coordinates to cartesian coordinates.

        :param s: Curvilinear s coordinate.
        :param d: Curvilinear d coordinate.

        :returns: Tuple with corresponding cartesian coords: (x, y).

        :raises ValueError: If curvilinear coordinates are outside of projection domain.
        """
        return self._clcs.convert_to_cartesian_coords(s, d)


class RoadNetwork:
    """
    Representation of the complete road network of a CommonRoad scenario abstracted to lanes
    """

    def __init__(
        self,
        lanelet_network: LaneletNetwork,
        road_network_param: Optional[RoadNetworkParam] = None,
        scenario_type: ScenarioType = ScenarioType.INTERSTATE,
    ):
        """
        :param lanelet_network: CommonRoad lanelet network
        :param road_network_param: dictionary with parameters for the road network
        :param scenario_type: scenario type (interstate or intersection)
        """
        self.lanelet_network = lanelet_network
        self.scenario_type = scenario_type

        if road_network_param is None:
            road_network_param = RoadNetworkParam()

        self.lanes = self._create_lanes(road_network_param, self.scenario_type)
        # add intersection elements for intersection scenarios
        if len(lanelet_network.intersections) != 0:
            self.incoming = self._create_incoming_dict(lanelet_network)
            self.lanes_incoming = self._create_lanes_of_incoming(lanelet_network)
            self.reach_suc_cache = {}
            self.reach_pre_cache = {}
        else:
            self.intersection_lanelets = set()
            self.incoming = {}
            self.lanes_incoming = {}
            self.reach_suc_cache = {}
            self.reach_pre_cache = {}

    def _create_lanes(
        self, road_network_param: RoadNetworkParam, scenario_type: ScenarioType
    ) -> List[Lane]:
        """
        Creates lanes for road network

        :param road_network_param: dictionary with parameters for the road network
        """
        lanes = []
        lane_lanelets = []
        start_lanelets = [
            lanelet for lanelet in self.lanelet_network.lanelets if len(lanelet.predecessor) == 0
        ]
        for lanelet in start_lanelets:
            (
                merged_lanelets,
                merge_jobs,
            ) = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                lanelet,
                self.lanelet_network,
                road_network_param.merging_length,
            )
            if len(merged_lanelets) == 0 or len(merge_jobs) == 0:
                merged_lanelets.append(lanelet)
                merge_jobs.append([lanelet.lanelet_id])
            for idx in range(len(merged_lanelets)):
                lane_lanelets.append((merged_lanelets[idx], merge_jobs[idx]))
        for lane_element in lane_lanelets:
            lanes.append(Lane(lane_element[0], lane_element[1], road_network_param, scenario_type))

        lanes.sort(key=lambda x: x.lane_id)

        # TODO: the adjacency assignments only work for highway so far. For intersections, more dedicated approach
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

    def find_lane_by_id(self, lane_id: int) -> Lane | None:
        for lane in self.lanes:
            if lane.lane_id == lane_id:
                return lane

        return None

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

    def find_lane_ids_by_lanelets(self, lanelets: Iterable[int]) -> set[int]:
        """
        Finds the lanes given set of lanelets belong to and returns their IDs

        :param lanelets: list of lanelet IDs
        :returns: set of lanelet IDs
        """
        lane_ids = set()
        for lane in self.lanes:
            for lanelet_id in lanelets:
                if lanelet_id in lane.contained_lanelets:
                    lane_ids.add(lane.lane_id)

        return lane_ids

    def find_lanes_by_lanelets(self, lanelets: Iterable[int]) -> set[Lane]:
        """
        Finds the lanes to which a given set of lanelets belongs to

        :param lanelets: list of lanelet IDs
        :returnsi: set of lane objects
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
        :returns: lane object
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
        :returns: lane the obstacle center is on
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

    # functions in intersection scenarios
    @staticmethod
    def _create_incoming_dict(
        lanelet_network: LaneletNetwork,
    ) -> Dict[int, IntersectionIncomingElement]:
        """
        creates incoming direction for an intersection

        :param lanelet_network: lanelets network
        """
        # TODO: currently only consider the first intersection
        incoming_dict = {}
        for incoming_element in lanelet_network.intersections[0].incomings:
            incoming_dict[incoming_element.incoming_id] = incoming_element
        return incoming_dict

    def _create_lanes_of_incoming(self, lanelet_network: LaneletNetwork) -> Dict[int, List[Lane]]:
        """
        find right turning, left turning, and going straight lanes with respect to incomings

        :param lanelet_network: lanelets network
        """
        lanes_incoming = {}
        for intersection in lanelet_network.intersections:
            for incoming in intersection.incomings:
                lanes_incoming[incoming.incoming_id] = [
                    self.get_turning_lane_from_incoming(self.lanes, incoming, "right"),
                    self.get_turning_lane_from_incoming(self.lanes, incoming, "straight"),
                    self.get_turning_lane_from_incoming(self.lanes, incoming, "left"),
                ]
        return lanes_incoming

    def find_lanes_incoming_by_id(self, incoming_id: int) -> "List[Lane]":
        """
        Finds lanes by given an incoming id

        :param incoming_id: ID of the incoming
        """
        return self.lanes_incoming[incoming_id]

    def get_reach_suc_cache(self, lanelet_id: int) -> "np.array":
        if lanelet_id in self.reach_suc_cache:
            return self.reach_suc_cache[lanelet_id]
        else:
            self.reach_suc_cache[lanelet_id] = self.lanelet_reach_suc(lanelet_id)
            return self.reach_suc_cache[lanelet_id]

    def get_reach_pre_cache(self, lanelet_id: int) -> "np.array":
        if lanelet_id in self.reach_pre_cache:
            return self.reach_pre_cache[lanelet_id]
        else:
            self.reach_pre_cache[lanelet_id] = self.lanelet_reach_pre(lanelet_id)
            return self.reach_pre_cache[lanelet_id]

    def lanelet_reach_suc(self, lanelet_id: int) -> "np.array":
        """
        Finds reach_suc of a lanelet

        :param lanelet_id: ID of the lanelet
        """
        paths = self.lanes_suc(lanelet_id)
        paths = [l_id for path in paths for l_id in path]
        return np.unique(paths)

    def lanes_suc(self, lanelet_id: int) -> "List[List[int]]":
        """
        Finds successors of a lanelets along lanes

        :param lanelet_id: ID of the lanelet
        """
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
        """
        Finds reach_pre of a lanelet

        :param lanelet_id: ID of the lanelet
        """
        paths = self.lanes_pre(lanelet_id)
        paths = [l_id for path in paths for l_id in path]
        return np.unique(paths)

    def lanes_pre(self, lanelet_id: int) -> "List[List[int]]":
        """
        Finds predecessors of a lanelets along lanes

        :param lanelet_id: ID of the lanelet
        """
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

    def find_incoming_intersection(
        self, lanelets_dir: "List[int]"
    ) -> "IntersectionIncomingElement":
        """
        Finds the incoming by given the lanelets_dir

        :param lanelets_dir: lanelets_dir of the vehicle (driving direction)
        """
        # TODO: further check needed
        possible_incomings = list()
        # get all possible occupied lanelets with respect to lanelets_dir
        lanelet_pre = self.get_reach_pre_cache(lanelets_dir[0])
        lanelet_suc = self.get_reach_suc_cache(lanelets_dir[-1])
        possible_occupied_lanelets = lanelets_dir + list(lanelet_pre) + list(lanelet_suc)
        # find possible incoming elements
        if len(self.lanelet_network.intersections) == 0:
            return None
        for incoming_element in self.lanelet_network.intersections[0].incomings:
            if (
                len(
                    incoming_element.incoming_lanelets.intersection(set(possible_occupied_lanelets))
                )
                > 0
            ):
                possible_incomings.append(incoming_element)
        if len(possible_incomings) == 0:
            return None
        elif len(possible_incomings) == 1:
            return possible_incomings[0]
        else:
            # more than one incoming is searched, assume vehicle drives straight.
            # if no straight goning lane is matched, select the first incoming
            for incoming in possible_incomings:
                straight_going_lane = self.get_turning_lane_from_incoming(
                    self.lanes, incoming, "straight"
                )
                if (
                    len(
                        straight_going_lane.contained_lanelets.intersection(
                            possible_occupied_lanelets
                        )
                    )
                    != 0
                ):
                    return incoming
            return possible_incomings[0]

    @staticmethod
    def get_turning_lane_from_incoming(
        lanes: "List[Lane]",
        incoming: IntersectionIncomingElement,
        turning_direction: str,
    ) -> "Lane":
        """
        Finds turning lane by given incoming and turning direction

        :param lanes: list of possible lanes
        :param incoming: incoming element
        :param turning_direction: turning direction (right, left, or straight)
        """
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
            if (
                len(lane.contained_lanelets.intersection(turning_lanelets_ids)) > 0
                and len(lane.contained_lanelets.intersection(incoming_lanelets_ids)) > 0
            ):
                possible_lanes.append(lane)
        selected_lanes = list()
        # find the longest lane
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

    def get_lanelets_start_end_s(
        self, lanelets_id: "Union[List, Set]", reference_lane: "Lane"
    ) -> (float, float):
        """
        Finds the longitudinal position of the start and end points of given lanelets along reference lane

        :param lanelets_id: list of IDs of given lanelets
        :param reference_lane: reference lane
        """
        lanelets_start_s = np.inf
        lanelets_end_s = -np.inf
        for lanelet_id in lanelets_id:
            lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_id)
            # TODO: check: now assume start and end lines vertical to reference lane
            start_s = reference_lane.convert_to_curvilinear_coords(*lanelet.right_vertices[0, :])[0]
            end_s = reference_lane.convert_to_curvilinear_coords(*lanelet.right_vertices[-1, :])[0]
            lanelets_start_s = min(lanelets_start_s, start_s)
            lanelets_end_s = max(lanelets_end_s, end_s)
        return lanelets_start_s, lanelets_end_s

    def adjacent_lanelets(self, lanelets_id: "Set[int]") -> "Set[int]":
        """
        Finds adjacent lanelets by given lanelets

        :param lanelets_id: list of IDs of given lanelets
        """
        for lanelet_id in lanelets_id:
            la = self.lanelet_network.find_lanelet_by_id(lanelet_id)
            while la is not None and la.adj_left is not None:
                if la.adj_left_same_direction:
                    la = self.lanelet_network.find_lanelet_by_id(la.adj_left)
                    if la is not None:
                        lanelets_id.add(la.lanelet_id)
                else:
                    la = None

            la = self.lanelet_network.find_lanelet_by_id(lanelet_id)
            while la is not None and la.adj_right is not None:
                if la.adj_right_same_direction:
                    la = self.lanelet_network.find_lanelet_by_id(la.adj_right)
                    if la is not None:
                        lanelets_id.add(la.lanelet_id)
                else:
                    la = None
        return lanelets_id
