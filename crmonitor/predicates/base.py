import abc
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Tuple

from commonroad.visualization.renderer import IRenderer

from crmonitor.common.world import World

from .scaling import IRobustnessScaler, RobustnessScaler

_LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True)
class PredicateConfig:
    scale_rob: bool = True

    eps: float = 1e-17

    min_interstate_width: float = 7.0

    max_congestion_velocity: float = 2.78
    """Determines the velocity of vehicles, when they are considered in congestion. Used for the predicates `PredInCongestion` and `PredHasCongestionVelocity`."""

    num_veh_congestion: float = 3.0
    """Determines the number of vehicles, when it is considered as congestion. Used for the predicate `PredInCongestion`."""

    max_slow_moving_traffic_velocity: float = 8.33
    """Determines the velocity of vehicles when they are considered in slow moving traffic. Used for the predicates `PredInSlowMovingTraffic` and `PredHasSlowMovingVelocity`."""

    num_veh_slow_moving_traffic: float = 3.0
    """Determines the number of vehicles, when it is considered as in slow moving traffic. Used for the predicate `PredInSlowMovingTraffic`."""

    max_queue_of_vehicles_velocity: float = 16.67
    """Determines the velocity of vehicles when they are considered in a queue of vehicles. Used for the predicates `PredInQueueOfVehicles` and `PredHasQueueVelocity`."""

    num_veh_queue_of_vehicles: float = 3.0
    """Determines the number of vehicles, when it is considered as in a queue of vehicles. Used for the predicate `PredInQueueOfVehicles`."""

    max_interstate_speed_truck: float = 22.22
    desired_interstate_velocity: float = 36.11

    u_turn: float = 1.57

    standstill_error: float = 0.01

    min_velocity_diff: float = 15

    slightly_higher_speed_difference: float = 5.55

    close_to_other_vehicle: float = 0.75
    close_to_lane_border: float = 0.2

    d_sl: float = 1.0
    d_br: float = 15.0
    a_br: float = -1.0

    a_abrupt: float = -2.0

    country: str = "DEU"


class PredicateName(str, Enum):
    def __str__(self) -> str:
        return self.value


class AbstractPredicate(abc.ABC):
    """
    Base class for the predicate evaluator
    """

    predicate_name: PredicateName
    arity: int

    def __init__(
        self,
        config: PredicateConfig | None = None,
        scaler: IRobustnessScaler | None = None,
    ) -> None:
        if config is None:
            config = PredicateConfig()
        self.config = config
        self._scaler = scaler or RobustnessScaler(self.config.scale_rob)

    def _scale_speed(self, x):
        return self._scaler.scale_speed(x)

    def _scale_acc(self, x):
        return self._scaler.scale_acc(x)

    def _scale_lon_dist(self, x):
        return self._scaler.scale_lon_dist(x)

    def _scale_lat_dist(self, x):
        return self._scaler.scale_lat_dist(x)

    def _scale_angle(self, x):
        return self._scaler.scale_angle(x)

    def evaluate_boolean(self, world: World, time_step: int, vehicle_ids: tuple[int, ...]) -> bool:
        return self.evaluate_robustness(world, time_step, vehicle_ids) >= 0.0

    @abc.abstractmethod
    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float: ...

    def visualize(
        self,
        vehicle_ids: List[int],
        add_vehicle_draw_params: Callable[[int, any], None],
        world: World,
        time_step: int,
        predicate_names2vehicle_ids2values: Dict[str, Dict[Tuple[int, ...], float]],
    ) -> Tuple[Callable[[IRenderer], None], ...]:
        """
        Overwrite this function for visualizing a predicate in a certain way within the scenario plot.
        """
        self._gather_predicate_values_to_plot(
            vehicle_ids, world, time_step, predicate_names2vehicle_ids2values
        )
        return ()

    def _gather_predicate_values_to_plot(
        self,
        vehicle_ids: List[int],
        world: World,
        time_step: int,
        predicate_names2vehicle_ids2values: Dict[str, Dict[Tuple[int, ...], float]],
    ):
        predicate_names2vehicle_ids2values[self.predicate_name][tuple(vehicle_ids)] = (
            self.evaluate_robustness(world, time_step, vehicle_ids)
        )

    @staticmethod
    def plot_predicate_visualization_legend(ax):
        ax.axis("off")
        ax.text(0.1, 0.5, "[not visualized]", fontsize=12)
