import copy
from dataclasses import dataclass, field
from typing import Set

import numpy as np
from commonroad.common.solution import PlanningProblemSolution, vehicle_parameters
from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import ObstacleType, Scenario, ScenarioID
from commonroad.scenario.trajectory import Trajectory
from commonroad_dc.feasibility.solution_checker import (
    _simulate_trajectory_if_input_vector,
)

from crmonitor.common.road_network import RoadNetwork, RoadNetworkParam
from crmonitor.common.scenario_type import ScenarioType
from crmonitor.common.vehicle import (
    ControlledVehicle,
    Vehicle,
)

l_wb = 2.578  # for BMW_320i

# Obstacle types which are currently supported as vehicles.
# Used to filter unsupported obstacles during World creation.
_SUPPORTED_VEHICLE_OBSTACLE_TYPES = {
    ObstacleType.CAR,
    ObstacleType.BUS,
    ObstacleType.TRUCK,
    ObstacleType.MOTORCYCLE,
    ObstacleType.TAXI,
}


@dataclass
class WorldConfig:
    scenario_type: ScenarioType = ScenarioType.INTERSTATE
    road_network_param: RoadNetworkParam = field(default_factory=RoadNetworkParam)


@dataclass
class World:
    vehicles: set[Vehicle]
    road_network: RoadNetwork
    scenario: Scenario | None = None

    @classmethod
    def create_from_solution(
        cls,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        planning_problem_solution: PlanningProblemSolution,
    ):
        """Create a rule evaluator to check a planning problem solution."""
        pp_id = planning_problem.planning_problem_id
        _, trajectory = _simulate_trajectory_if_input_vector(
            PlanningProblemSet([planning_problem]),
            planning_problem_solution,
            scenario.dt,
        )
        # We have to remove the initial time step
        trajectory = Trajectory(trajectory.state_list[1].time_step, trajectory.state_list[1:])
        shape = Rectangle(
            length=vehicle_parameters[planning_problem_solution.vehicle_type].l,
            width=vehicle_parameters[planning_problem_solution.vehicle_type].w,
        )
        prediction = TrajectoryPrediction(trajectory, shape=shape)
        obstacle = DynamicObstacle(
            # FIXME
            obstacle_id=pp_id + 1000,
            obstacle_type=ObstacleType.CAR,
            obstacle_shape=shape,
            initial_state=planning_problem.initial_state,
            prediction=prediction,
        )
        scenario = copy.deepcopy(scenario)
        scenario.add_objects(obstacle)
        scenario.assign_obstacles_to_lanelets()
        world = World.create_from_scenario(scenario)
        ego_vehicle = world.vehicle_by_id(obstacle.obstacle_id)
        return world, ego_vehicle

    def add_vehicle(self, vehicle: Vehicle):
        self.vehicles.add(vehicle)

    def remove_vehicle(self, vehicle: Vehicle):
        self.vehicles.remove(vehicle)

    @classmethod
    def create_from_scenario(
        cls,
        scenario: Scenario,
        config: WorldConfig | None = None,
        road_network: RoadNetwork | None = None,
    ) -> "World":
        """
        Create a new `World` object from a CommonRoad scenario.
        """
        if config is None:
            config = WorldConfig()

        if road_network is None:
            road_network = RoadNetwork(
                scenario.lanelet_network, config.road_network_param, config.scenario_type
            )
        else:
            road_network = road_network

        # Only convert dynamic obstacles, which have a currently supported obstacle type.
        supported_vehicle_obstacles = filter(
            lambda o: o.obstacle_type in _SUPPORTED_VEHICLE_OBSTACLE_TYPES,
            scenario.dynamic_obstacles,
        )

        # Only convert dynamic obstacle, which have a valid prediction.
        vehicle_obstacles_with_prediction = filter(
            lambda obstacle: obstacle.prediction is not None
            and obstacle.prediction.final_time_step - obstacle.prediction.initial_time_step > 1,
            supported_vehicle_obstacles,
        )

        # For intersection scenarios, some more restrictions are placed on the valid vehicles.
        if config.scenario_type == ScenarioType.INTERSECTION:
            valid_vehicle_obstacles = filter(
                lambda obs: obs.obstacle_type == ObstacleType.CAR and not cls.static_vehicle(obs),
                vehicle_obstacles_with_prediction,
            )
        else:
            valid_vehicle_obstacles = vehicle_obstacles_with_prediction

        # Convert the selected dynamic obstacles to vehicles.
        vehicles = set(
            map(
                lambda obs: Vehicle.from_dynamic_obstacle(
                    obs,
                    road_network=road_network,
                    dt=scenario.dt,
                    scenario_type=config.scenario_type,
                ),
                valid_vehicle_obstacles,
            )
        )

        return cls(vehicles, road_network, scenario)

    @property
    def controlled_vehicle_ids(self) -> Set[int]:
        return {vehicle.id for vehicle in self.vehicles if isinstance(vehicle, ControlledVehicle)}

    def vehicle_ids_for_time_step(self, time_step: int):
        return [v.id for v in self.vehicles if v.is_valid(time_step)]

    @staticmethod
    def static_vehicle(dynamic_obstacle: DynamicObstacle):
        """
        Checks whether the obstacle is static.
        """
        velocity = np.array(
            [state.velocity for state in dynamic_obstacle.prediction.trajectory.state_list]
        )
        return all(velocity <= 0.001)

    @staticmethod
    def augment_state_longitudinal(dt, obs):
        accelerations = (
            np.diff(
                [s.velocity for s in [obs.initial_state] + obs.prediction.trajectory.state_list]
            )
            / dt
        ).tolist()
        obs.initial_state.acceleration = accelerations[0]
        if len(accelerations) >= 2:
            jerk = (np.diff(accelerations) / dt).tolist()
            accelerations += accelerations[-1:]
            jerk += [jerk[-1], 0]
            obs.initial_state.jerk = jerk[0]

            # Compute the gradient of jerk
            jerk_dot = (np.diff(jerk) / dt).tolist()
            jerk_dot += [0]
            obs.initial_state.jerk_dot = jerk_dot[0]
        else:
            jerk = [None] * 2
            jerk_dot = [None] * 2
        for a, j, j_dot, state in zip(
            accelerations[1:], jerk[1:], jerk_dot, obs.prediction.trajectory.state_list
        ):
            state.acceleration = a
            if j is not None:
                state.jerk = j
            if j_dot is not None:
                state.jerk_dot = j_dot

    @staticmethod
    def augment_state_lateral(dt: float, obs):
        kappa_list = []
        for state in [obs.initial_state] + obs.prediction.trajectory.state_list:
            if hasattr(state, "steering_angle"):
                state.kappa = state.velocity / l_wb * np.tan(state.steering_angle)
            elif hasattr(state, "yaw_rate"):
                state.kappa = state.yaw_rate
            else:
                state.kappa = 0.0  # todo: fix for intersection
            kappa_list.append(state.kappa)
        if len(kappa_list) >= 2:
            kappa_dot = (np.diff(kappa_list) / dt).tolist()
            kappa_dot += [0]

            kappa_dot_dot = (np.diff(kappa_dot) / dt).tolist()
            kappa_dot_dot += [0]
            obs.initial_state.kappa_dot = kappa_dot[0]
            obs.initial_state.kappa_dot_dot = kappa_dot_dot[0]
        else:
            kappa_dot = [None] * 2
            kappa_dot_dot = [None] * 2
        for k_dot, k_ddot, state in zip(
            kappa_dot[1:], kappa_dot_dot[1:], obs.prediction.trajectory.state_list
        ):
            if k_dot is not None:
                state.kappa_dot = k_dot
            if k_ddot is not None:
                state.kappa_dot_dot = k_ddot

    def vehicle_by_id(self, vehicle_id: int) -> Vehicle:
        for veh in self.vehicles:
            if veh.id == vehicle_id:
                return veh

        raise RuntimeError(f"Vehicle {vehicle_id} is not part of world {self.scenario_id}")

    def vehicle_ids(self):
        """
        Finds ids of all vehicles.
        """
        vehicle_ids = list()
        for veh in self.vehicles:
            vehicle_ids.append(veh.id)
        return vehicle_ids

    @property
    def dt(self):
        if self.scenario is not None:
            return self.scenario.dt
        else:
            return 0.1

    @property
    def scenario_id(self) -> ScenarioID:
        if self.scenario is not None:
            return self.scenario.scenario_id
        else:
            return ScenarioID()
