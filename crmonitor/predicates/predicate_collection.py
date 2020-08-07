import enum
from abc import ABC, abstractmethod
from typing import List, Dict, Set, Tuple, Union

from commonroad.geometry.shape import Shape, ShapeGroup, Polygon, Rectangle, Circle
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter

from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import Vehicle


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
class ConstraintRepresentation(enum.Enum):
    """
    Defines the representation of a constraint
    """
    UPPER = 0  # real valued upper constraint
    LOWER = 1  # real valued lower constraint
    OUTER_BOUNDARY = 2  # CommonRoad shape as an outer boundary
    INNER_BOUNDARY = 3  # CommonRoad shape as an inner boundary


@enum.unique
class ConstraintType(enum.Enum):
    """
    Defines the type of constraint axis
    """
    LONGITUDINAL_CURVILINEAR_POSITION = 0
    LATERAL_CURVILINEAR_POSITION = 1
    X_CARTESIAN_POSITION = 2
    Y_CARTESIAN_POSITION = 3
    VELOCITY = 4
    ORIENTATION = 5
    ACCELERATION = 6


class Constraint:
    """
    Representation of a constraint so that constraints can be uses outside of the CommonRoad monitor.
    """

    def __init__(self, axis: List[ConstraintType], constraint_representation: ConstraintRepresentation,
                 value: Union[int, float, Shape, ShapeGroup, Polygon, Rectangle, Circle]):
        """
        Constructor

        :param axis: list of axis values
        :param constraint_representation: dictionary with parameters of the simulation environment
        :param value: constraint: can be a CommonRoad shape or a real value.
        """
        self._axis = axis
        self._constraint_representation = constraint_representation
        self._value = value

    @property
    def axis(self) -> List[ConstraintType]:
        return self._axis

    @property
    def constraint_representation(self) -> ConstraintRepresentation:
        return self._constraint_representation

    @property
    def value(self) -> Union[int, float, Shape, ShapeGroup, Polygon, Rectangle, Circle]:
        return self._value
