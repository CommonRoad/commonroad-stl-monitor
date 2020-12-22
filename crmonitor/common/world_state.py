import collections
from typing import Iterator

from commonroad.scenario.scenario import Scenario

from crmonitor.common.helper import create_scenario_vehicles, \
    create_ego_vehicle_param, \
    create_simulation_param, \
    create_other_vehicles_param
from crmonitor.common.road_network import RoadNetwork
from commonroad.scenario.scenario import Scenario

from crmonitor.common.helper import create_scenario_vehicles, \
    create_ego_vehicle_param, \
    create_simulation_param, \
    create_other_vehicles_param
from crmonitor.common.road_network import RoadNetwork


class WorldState:

    def __init__(self, scenario: Scenario, ego_obs_id, config, time_step=0, road_network=None):
        self.scenario = scenario
        self.time_step = time_step
        self.config = config
        if road_network is None:
            params = config.get("road_network_param")
            self.road_network = RoadNetwork(scenario.lanelet_network, params)
        else:
            self.road_network = road_network
        ego_obs = scenario.obstacle_by_id(ego_obs_id)
        simulation_param = create_simulation_param(
                config.get("simulation_param"), scenario.dt, scenario.scenario_id.country_id)
        ego_param = create_ego_vehicle_param(config.get("ego_vehicle_param"),
                                             simulation_param)
        others_params = create_other_vehicles_param(
                config.get("other_vehicles_param"))
        self.ego_vehicle, self.other_vehicles = create_scenario_vehicles(0.2,
                                                                         ego_obs,
                                                                         ego_param,
                                                                         others_params,
                                                                         self.road_network,
                                                                         scenario.dynamic_obstacles)

    def step(self):
        self.time_step += 1

    def vehicle_by_id(self, id):
        if id == self.ego_vehicle.id:
            return self.ego_vehicle

        for veh in self.other_vehicles:
            if veh.id == id:
                return veh

    @property
    def num_time_steps(self):
        return self.ego_vehicle.state_list_cr[-1].time_step + 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.time_step < self.num_time_steps - 1:
            self.step()
            return self
        else:
            raise StopIteration
    # def copy(self):
    #     return WorldState(scenario=self.scenario, ego_obs_id=self.ego_vehicle.id, config=self.config, time_step=self.time_step, road_network=self.road_network)
