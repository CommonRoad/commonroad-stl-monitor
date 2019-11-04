from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle
from common.obstacle_selection import ObstacleSelection
from commonroad_cc.collision_detection.pycrcc_collision_dispatch import create_collision_checker
from commonroad_cc.collision_detection.pycrcc_collision_dispatch import create_collision_object
from commonroad.prediction.prediction import TrajectoryPrediction
from common.vehicle import Vehicle
from commonroad.scenario.trajectory import Trajectory, State
from common.configuration import *
import common.util as util
from common.simulation import Simulation
from acc.safe_acc import SafeACC
import numpy as np
import cProfile
from typing import List


class ScenarioSimulation(Simulation):
    def __init__(self, simulation_param: Dict, acc_param: Dict, ego_vehicle_param: Dict, general_acc_param: Dict,
                 other_vehicles_param: Dict):
        """
        :param simulation_param: dictionary with parameters of the simulation environment
        :param acc_param: dictionary with parameters of the acc related algorithms
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param general_acc_param: dictionary with general parameters of the ACC controller
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        """
        super().__init__(simulation_param, acc_param, ego_vehicle_param, general_acc_param, other_vehicles_param)

        # Initialization of variables
        self._scenario, planning_problem_set = \
            CommonRoadFileReader("./scenarios/" + simulation_param.get("commonroad_benchmark_id") + ".xml").open()
        self._cc = create_collision_checker(self._scenario)
        self._planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
        initial_ego_state = self._planning_problem.initial_state

        self._ego_lanelet_id, self._ego_lanelet, self._ego_lane, self._curvilinear_cosy_ego_lane, self._left_lane, \
        self._right_lane = util.update_ego_lane_info(self._scenario, initial_ego_state, ego_vehicle_param)
        self._ego_vehicle_shape = Rectangle(parameters_vehicle2().l, parameters_vehicle2().w)
        self._theta_ref_ego_lane = util.compute_orientation_from_polyline(self._ego_lane.center_vertices)
        self._ref_pos_ego_lane = util.compute_pathlength_from_polyline(self._ego_lane.center_vertices)
        if self._right_lane is not None:
            self._theta_ref_right_lane = util.compute_orientation_from_polyline(self._right_lane.center_vertices)
            self._ref_pos_right_lane = util.compute_pathlength_from_polyline(self._right_lane.center_vertices)
        if self._left_lane is not None:
            self._theta_ref_left_lane = util.compute_orientation_from_polyline(self._left_lane.center_vertices)
            self._ref_pos_left_lane = util.compute_pathlength_from_polyline(self._left_lane.center_vertice)
        ego_state_lon, ego_state_lat = \
            util.create_curvilinear_states(initial_ego_state, self._curvilinear_cosy_ego_lane,
                                           self._theta_ref_ego_lane, self._ref_pos_ego_lane)
        self._ego_vehicle = Vehicle(ego_state_lon, ego_state_lat, self._ego_vehicle_shape, 0,
                                    initial_ego_state, 0)
        self._max_scenario_time = util.get_max_scenario_duration(self._planning_problem)
        self._ref_orientation = util.calculate_orientation_from_polyline(self._ego_lane.center_vertices)
        self._ref_pathlength = util.compute_pathlength_from_polyline(self._ego_lane.center_vertices)

        # Initialization of class objects and vehicle list
        self._safe_acc = SafeACC(self._cruise_control, self._safety_layer, self._cutin_reaction,
                                 self._scenario, simulation_param, acc_param.get("common"))
        self._object_selection = ObstacleSelection(self._scenario, self._curvilinear_cosy_ego_lane, self._ego_lane,
                                                   self._ego_lanelet, self._left_lane, self._right_lane,
                                                   self._safety_layer, ego_vehicle_param, other_vehicles_param)

    def obstacles_at_time_step(self, time_step: int) -> List[Dict]:
        """
        Extract obstacles existing at current time step from CommonRoad scenario
        :param time_step: current time step
        :returns: list with dictionaries containing obstacle properties and states
        """
        vehicles = []
        for obs in self._scenario.dynamic_obstacles:
            vehicle_properties = {}
            if time_step == obs.initial_state.time_step:
                vehicle_properties["state_cr"] = obs.initial_state
            else:
                for cr_state in obs.prediction.trajectory.state_list:
                    if cr_state.time_step == time_step:
                        vehicle_properties["state_cr"] = cr_state
                        break
            if vehicle_properties["state_cr"] is not None:
                if self._right_lane is not None \
                        and self._right_lane.contains_points(np.array(vehicle_properties["state_cr"].position)):
                    vehicle_properties["lane_number"] = 1
                    state_lon, state_lat = util.create_curvilinear_states(vehicle_properties["state_cr"],
                                                                          self._curvilinear_cosy_ego_lane,
                                                                          self._theta_ref_right_lane,
                                                                          self._ref_pos_right_lane)
                elif self._left_lane is not None and self._left_lane.contains_points(
                        np.array(obs.initial_state.position)):
                    vehicle_properties["lane_number"] = -1
                    state_lon, state_lat = util.create_curvilinear_states(vehicle_properties["state_cr"],
                                                                          self._curvilinear_cosy_ego_lane,
                                                                          self._theta_ref_left_lane,
                                                                          self._ref_pos_left_lane)
                else:
                    vehicle_properties["lane_number"] = 0
                    state_lon, state_lat = util.create_curvilinear_states(vehicle_properties["state_cr"],
                                                                          self._curvilinear_cosy_ego_lane,
                                                                          self._theta_ref_ego_lane,
                                                                          self._ref_pos_ego_lane)
                vehicle_properties["state_lon"] = state_lon
                vehicle_properties["state_lat"] = state_lat
                vehicle_properties["shape"] = obs.obstacle_shape
                vehicle_properties["id"] = obs.obstacle_id
                vehicles.append(vehicle_properties)
        return vehicles

    def simulate(self):
        """
        Executes CommonRoad scenario with ego vehicle using the safe ACC
        """
        for time_step in range(0, self._max_scenario_time):
            if self.verbose_mode:
                pr = cProfile.Profile()
                pr.enable()
            print("time step: " + str(time_step))

            # Load obstacles in the field of view of the ego
            obstacles_at_time_step = self.obstacles_at_time_step(time_step)
            vehicles_fov_same_lane, vehicles_fov_cutin = \
                self._object_selection.extract_vehicles(time_step, self._ego_vehicle, obstacles_at_time_step)
            # Calculate ego vehicle input
            a_lat = self._nominal_acc.lateral_motion(self._ego_vehicle.state_list_lat[time_step].d,
                                                     self._ego_vehicle.state_list_lat[time_step].v_d)
            a_lon = self._safe_acc.execute(vehicles_fov_same_lane, vehicles_fov_cutin, self._ego_vehicle, time_step)

            # Propagate ego vehicle
            self._ego_vehicle = util.simulate_ego_vehicle(self._ego_vehicle, self._dt, a_lat, a_lon,
                                                          self._curvilinear_cosy_ego_lane,
                                                          self._ego_lane.center_vertices,
                                                          self._ego_vehicle_param.get("a_min"),
                                                          self._ego_vehicle_param.get("a_max"), time_step)

            # Collision check
            trajectory = Trajectory(time_step, [self._ego_vehicle.state_list_cr[time_step]])
            traj_pred = TrajectoryPrediction(trajectory=trajectory, shape=self._ego_vehicle_shape)
            co = create_collision_object(traj_pred)
            if self._cc.collide(co):
                print("collision")
                break
            if self._planning_problem.goal_reached(trajectory)[0] is True:
                print("goal reached")
                break
            if self.verbose_mode:
                pr.disable()
                pr.print_stats(sort='time')
