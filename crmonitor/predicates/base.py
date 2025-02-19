import abc
import logging
import warnings
from typing import Callable, Dict, List, Tuple

from commonroad.visualization.renderer import IRenderer
from commonroad_mpr.common.observation import World as WorldMPR
from commonroad_mpr.learning import FeatureExtrator
from commonroad_mpr.learning import PredicateEvaluatorML as PEML
from commonroad_mpr.mpr import ModelPredictiveRobustness
from ruamel.yaml.comments import CommentedMap

from crmonitor.common.world import World
from crmonitor.predicates.scaling import RobustnessScaler

logger = logging.getLogger(__name__)


def _map_crmonitor_predicate_name_to_mpr_predicate_name(
    predicate_name: str,
) -> str:
    """
    Returns the predicate name to lookup in mpr. Used to fix up predicate names which differ between crmonitor and mpr.
    """
    if predicate_name == "rel_brakes_abruptly":
        return "brakes_abruptly_relative"

    return predicate_name


# Those predicates are non-atomic and a composition of other atomic predicates. They do not have a pendant in mpr, and therefore they get special treatment when evaluating with mpr.
_COMPOSED_PREDICATES = [
    "preserves_traffic_flow",
    "slow_leading_vehicle",
    "in_congestion",
    "exist_standing_leading_vehicle",
    "in_slow_moving_traffic",
]


def _should_use_mpr_predicate_for(predicate_name: str) -> bool:
    """
    Determines whether a crmonitor predicate should be replaced with its pendant from mpr.
    This is usefull, to keep non-atomic predicates from crmonitor, and only replace their sub-predicates with predicates from mpr.
    """
    return predicate_name not in _COMPOSED_PREDICATES


class BasePredicateEvaluator(abc.ABC):
    """
    Base class for the predicate evaluator
    """

    predicate_name = "interface"

    def __init__(self, config: CommentedMap, scaler=None):
        self.config = config
        self.eps = 1e-5
        scale_rob = config.get("scale_rob", True)
        self._scaler = scaler or RobustnessScaler(scale_rob)

        # peml and mpr are the evalutors for mpr with and without pre-trained models respectivly.
        self.peml = None
        self.mpr = None
        # There might be
        mpr_predicate_name = _map_crmonitor_predicate_name_to_mpr_predicate_name(
            self.predicate_name
        )
        self._use_mpr_for_evaluation = self.config[
            "use_mpr"
        ] and _should_use_mpr_predicate_for(mpr_predicate_name)
        if self._use_mpr_for_evaluation:
            try:
                self.peml = PEML([mpr_predicate_name])
            except Exception:
                warnings.warn(
                    f"Model for gaussian processes for mpr evaluation of predicate '{self.predicate_name}' is not available. Falling back to MPR without gaussian processes for this predicate. This might result in slower evaluation."
                )
                try:
                    self.mpr = ModelPredictiveRobustness(
                        [mpr_predicate_name], normalize=scale_rob
                    )
                except KeyError as e:
                    raise RuntimeError(
                        f"The predicate '{mpr_predicate_name}' is not supported by MPR."
                    ) from e

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

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        return self.evaluate_robustness(world, time_step, vehicle_ids) >= 0.0

    @abc.abstractmethod
    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        pass

    def evaluate_mpr(
        self, world: World, world_mpr: WorldMPR, time_step, vehicle_ids: List[int]
    ) -> float:
        """
        Evaluation of model predictive robustness
        """
        # computation for single predicate
        vehicles = []
        for veh_id in vehicle_ids:
            vehicles.append(world_mpr.vehicle_by_id(veh_id))

        if self.peml is not None:
            robustness, _ = self.peml.evaluate_robustness(
                world=world_mpr, vehicles=vehicles, time_step=time_step
            )
            return robustness[0]
        elif self.mpr is not None:
            robustness = self.mpr.evaluate(
                world=world_mpr, vehicles=vehicles, time_step=time_step
            )
            # Each mpr evaluator is always configured for exactly one predicate.
            # As the output predicate names might be different then the input predicate names (TOOD: fix in commonroad-mpr), we simply use the single key as to retrive the robustenss value.
            mpr_predicate_name = list(robustness.keys())[0]
            return robustness[mpr_predicate_name]["robustness"]
        else:
            # Should be unreachable, because __init__ should ensure that the evaluators are available if we are evaluating with mpr.
            raise RuntimeError(
                f"Cannot evaluate model predictive robutness of predicate '{self.predicate_name}', as no evaluator is available. This is a bug."
            )

    def gradient_mpr(self):
        """
        Computes the gradient of the MPR w.r.t. the input values
        """
        default = [0.0] * 35
        if not self._use_mpr_for_evaluation:
            # TODO: If the user tries to extract the gradient for a comosed/exempted predicate, should it just be skipped?
            warnings.warn(
                f"Tried to extract the gradient of the model predictive evaluation, but model predictive evaluation is not enabled for '{self.predicate_name}'. This is either because mpr is disabled or this predicate is exempted from MPR."
            )
            return default

        if self.peml is None:
            warnings.warn(
                f"Tried to extract gradient of the model predictive robustness, but no gaussian processes were used for the evaluation of '{self.predicate_name}' and therefore no gradient is available."
            )
            return default

        return self.peml.derivative()[0]

    def evaluate_robustness_with_cache(
        self, world: World, mpr_world: WorldMPR, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        vehicle_ids_tuple = tuple(vehicle_ids)
        value = vehicle.predicate_cache.get_robustness(
            time_step, self.predicate_name, vehicle_ids_tuple[1:]
        )
        if value is None:
            logger.debug(
                "Evaluating predicate %s , t=%d, ids=%s",
                self.predicate_name,
                time_step,
                vehicle_ids_tuple,
            )
            if self._use_mpr_for_evaluation:
                value = self.evaluate_mpr(world, mpr_world, time_step, vehicle_ids)
            else:
                value = self.evaluate_robustness(world, time_step, vehicle_ids)
            vehicle.predicate_cache.set_robustness(
                time_step, self.predicate_name, vehicle_ids_tuple[1:], value
            )
        return value

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
        predicate_names2vehicle_ids2values[self.predicate_name][
            tuple(vehicle_ids)
        ] = self.evaluate_robustness_with_cache(world, time_step, vehicle_ids)

    @staticmethod
    def plot_predicate_visualization_legend(ax):
        ax.axis("off")
        ax.text(0.1, 0.5, "[not visualized]", fontsize=12)
