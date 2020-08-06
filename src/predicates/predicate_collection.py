from abc import ABC, abstractmethod
from typing import List, Dict, Set, Tuple, Union
import enum

from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter
from commonroad.geometry.shape import Shape, ShapeGroup, Polygon, Rectangle, Circle

from src.common.road_network import RoadNetwork
from src.common.vehicle import Vehicle


class PredicateCollection(ABC):
    """
    Interface for a predicate class
    """
    def __init__(self, road_network: RoadNetwork, simulation_param: Dict, traffic_rules_param: Dict,
                 necessary_predicates: Set[str], traffic_sign_interpreter: TrafficSigInterpreter):
        """
        Constructor

        :param road_network: CommonRoad lanelet network
        :param simulation_param: dictionary with parameters of the simulation environment
        :param traffic_rules_param: dictionary with parameters of traffic rule parameters
        :param necessary_predicates: set with all predicates which should be evaluated
        :param traffic_sign_interpreter: CommonRoad traffic sign interpreter
        """
        self._road_network = road_network
        self._simulation_param = simulation_param
        self._traffic_rules_param = traffic_rules_param
        self._country = simulation_param.get("country")
        self._necessary_predicates = necessary_predicates
        self._traffic_sign_interpreter = traffic_sign_interpreter

    @abstractmethod
    def evaluate_predicates(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle],
                            time_interval: Tuple[int, int]) -> Dict[str, Dict[int, Dict[int, bool]]]:
        """
        Evaluates trajectory for predicate compliance

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :param time_interval: time interval for which the predicates should be evaluated
        :returns dictionary with traces of bool values for each predicate
        """
        pass

    @abstractmethod
    def evaluate_constraints(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle],
                             time_interval: Tuple[int, int]) -> Dict[str, Dict[int, Dict[int, float]]]:
        """
        Extracts constraints for a vehicle

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :param time_interval: time interval for which the predicates should be evaluated
        :returns dictionary with traces of constraints for each predicate
        """
        pass

    @abstractmethod
    def evaluate_robustness(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle],
                            time_interval: Tuple[int, int]) -> Dict[str, Dict[int, Dict[int, float]]]:
        """
        Extracts robustness values for a vehicle

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :param time_interval: time interval for which the predicates should be evaluated
        :returns dictionary with traces of robustness values for each predicate
        """
        pass


@enum.unique
class ConstraintType(enum.Enum):
    """
    Defines the representation of a constraint
    """
    UPPER = 0   # real valued upper constraint
    LOWER = 1   # real valued lower constraint
    OUTER_BOUNDARY = 2 # CommonRoad shape as an outer boundary
    INNER_BOUNDARY = 3 # CommonRoad shape as an inner boundary


class Constraint:
    """
    Representation of a constraint so that constraints can be uses outside of the CommonRoad monitor.
    """
    def __init__(self, axis: List[str], constraint_type: ConstraintType,
                 value: Union[int, float, Shape, ShapeGroup, Polygon, Rectangle, Circle]):
        """
        Constructor

        :param axis: list of axis values
        :param constraint_type: dictionary with parameters of the simulation environment
        :param value: constraint: can be a CommonRoad shape or a real value.
        """
        self._axis = axis
        self._constraint_type = constraint_type
        self._value = value

    @property
    def axis(self) -> List[str]:
        return self._axis

    @property
    def constraint_type(self) -> ConstraintType:
        return self._constraint_type

    @property
    def value(self) -> Union[int, float, Shape, ShapeGroup, Polygon, Rectangle, Circle]:
        return self._value
