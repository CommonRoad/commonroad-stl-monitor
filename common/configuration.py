from parameters_vehicle1 import parameters_vehicle1
from parameters_vehicle2 import parameters_vehicle2
from parameters_vehicle3 import parameters_vehicle3
from typing import Dict, Union
import ruamel.yaml
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry


def create_ego_vehicle_param(ego_vehicle_param: Dict, simulation_param: Dict) -> Dict:
    """
    Update ACC vehicle parameters

    :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
    :param simulation_param: dictionary with parameters of the simulation environment
    :returns updated dictionary with parameters of ACC vehicle
    """
    if ego_vehicle_param.get("vehicle_number") == 1:
        ego_vehicle_param["dynamics_param"] = parameters_vehicle1()
    elif ego_vehicle_param.get("vehicle_number") == 2:
        ego_vehicle_param["dynamics_param"] = parameters_vehicle2()
    elif ego_vehicle_param.get("vehicle_number") == 3:
        ego_vehicle_param["dynamics_param"] = parameters_vehicle3()
    else:
        raise ValueError('Wrong vehicle number for ACC vehicle in config file defined.')

    v_max = min(ego_vehicle_param.get("dynamics_param").longitudinal.v_max, ego_vehicle_param.get("v_des"))

    ego_vehicle_param["v_max"] = v_max

    if not -1e-12 <= (ego_vehicle_param.get("t_react") % simulation_param.get("dt")) <= 1e-12:
        raise ValueError('Reaction time must be multiple of time step size.')

    return ego_vehicle_param


def create_other_vehicles_param(other_vehicles_param: Dict) -> Dict:
    """
    Update other vehicle's parameters

    :param other_vehicles_param: dictionary with physical parameters of other vehicles
    :returns updated dictionary with parameters of other vehicles
    """
    if other_vehicles_param.get("vehicle_number") == 1:
        other_vehicles_param["dynamics_param"] = parameters_vehicle1()
    elif other_vehicles_param.get("vehicle_number") == 2:
        other_vehicles_param["dynamics_param"] = parameters_vehicle2()
    elif other_vehicles_param.get("vehicle_number") == 3:
        other_vehicles_param["dynamics_param"] = parameters_vehicle3()
    else:
        raise ValueError('Wrong vehicle number for leading vehicle in config file defined.')

    return other_vehicles_param


def create_simulation_param(simulation_param: Dict, dt: float, country: str) -> Dict:
    """
    Update simulation parameters

    :param simulation_param: dictionary with parameters of the simulation environment
    :param country: country of CommonRoad scenario
    :param dt: time step size of CommonRoad scenario
    :returns updated dictionary with parameters of CommonRoad scenario
    """
    simulation_param["dt"] = dt
    simulation_param["country"] = SupportedTrafficSignCountry(country)

    return simulation_param


def load_yaml(file_name: str) -> Union[Dict, None]:
    """
    Loads configuration setup from a yaml file

    :param file_name: name of the yaml file
    """
    with open(file_name, 'r') as stream:
        try:
            config = ruamel.yaml.round_trip_load(stream, preserve_quotes=True)
            return config
        except ruamel.yaml.YAMLError as exc:
            print(exc)
            return None
