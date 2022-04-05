import dataclasses
import logging
import shelve
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Set, Dict

import crmonitor
import numpy as np
from commonroad.scenario.scenario import Scenario
from ruamel.yaml import YAML

from crmonitor.common.helper import (create_scenario_vehicles, create_ego_vehicle_param, create_simulation_param,
                                     create_other_vehicles_param, load_yaml, )
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import Vehicle, DynamicObstacleVehicle, CurvilinearStateManager, ControlledVehicle, \
    PredicateCache
import importlib.resources as pkg_resources

@dataclass
class WorldState:
    vehicles: Set[Vehicle]
    road_network: RoadNetwork
    time_step: int = 0
    scenario: Optional[Scenario] = None
    cache: Optional[shelve.Shelf] = None


    @classmethod
    def create_from_scenario(
        cls, scenario: Scenario, config=None, time_step=0, road_network=None, cache_dir="/tmp"
    ):
        if config is None:
            with pkg_resources.path(crmonitor, "config.yaml") as config_path:
                config = load_yaml(config_path)
        if road_network is None:
            params = config.get("road_network_param")
            road_network = RoadNetwork(scenario.lanelet_network, params)
        else:
            road_network = road_network
        others_params = create_other_vehicles_param(config.get("other_vehicles_param"))
        vehicles = set()
        cache_file = Path(cache_dir) / f"{scenario.scenario_id}"
        cache = shelve.open(str(cache_file), writeback=True)
        for obs in scenario.dynamic_obstacles:
            cls.augment_state_acceleration_jerk(scenario.dt, obs)
            curvi_cache, predicate_dict = cache.setdefault(str(obs.obstacle_id), (dict(), defaultdict(partial(defaultdict, dict))))
            vehicles.add(DynamicObstacleVehicle(obs, CurvilinearStateManager(road_network, curvi_cache), others_params, PredicateCache(predicate_dict)))
        return cls(vehicles, road_network, time_step, scenario, cache)

    @staticmethod
    def augment_state_acceleration_jerk(dt, obs):
        accelerations = (np.diff([s.velocity for s in [obs.initial_state] + obs.prediction.trajectory.state_list]) / dt).tolist()
        obs.initial_state.acceleration = accelerations[0]
        if len(accelerations) >= 2:
            jerk = (np.diff(accelerations) / dt).tolist()
            accelerations += accelerations[-1:]
            jerk += [jerk[-1], 0]
            obs.initial_state.jerk = jerk[0]
        else:
            jerk = [None] * 2
        for a, j, state in zip(accelerations[1:], jerk[1:], obs.prediction.trajectory.state_list):
            state.acceleration = a
            if j is not None:
                state.jerk = j

    def step(self):
        self.time_step += 1

    def vehicle_by_id(self, id) -> Optional[Vehicle]:
        for veh in self.vehicles:
            if veh.id == id:
                break
        else:
            logging.warning(f"Vehicle with ID {id} not found!")
            veh = None
        return veh

    # @property
    # def num_time_steps(self):
    #     return self._ego_vehicle.state_list_cr[-1].time_step + 1

    # @property
    # def other_ids(self):
    #     return [v.id for v in self.other_vehicles]

    @property
    def dt(self):
        if self.scenario is not None:
            return self.scenario.dt
        else:
            return 0.1

    def __del__(self):
        if self.cache is not None:
            logging.info("Cache close!")
            self.cache.close()

    # def __iter__(self):
    #     return self
    #
    # def __next__(self):
    #     if self.time_step < self.num_time_steps - 1:
    #         self.step()
    #         return self
    #     else:
    #         raise StopIteration

    # def __eq__(self, o) -> bool:
    #     if o is None:
    #         return False
    #     return (
    #         self.scenario.scenario_id == o.scenario.scenario_id
    #         and self.time_step == o.time_step
    #         and self.ego_vehicle.id == o.ego_vehicle.id
    #     )

    # def clear_predicate_values_timesteps(self, start_time_step, end_time_step):
    #     """
    #     Clear internal cached predicates between for the given time interval
    #     :param start_time_step: start of the interval (inclusive)
    #     :param end_time_step: end of the interval (inclusive)
    #     :return:
    #     """
    #     for i in range(start_time_step, end_time_step + 1):
    #         if i in self.predicate_values:
    #             self.predicate_values.pop(i)
    #
    # def clear_predicates(self):
    #     self.predicate_values.clear()

    # def copy(self):
    #     return WorldState(scenario=self.scenario, ego_obs_id=self.ego_vehicle.id, config=self.config, time_step=self.time_step, road_network=self.road_network)
