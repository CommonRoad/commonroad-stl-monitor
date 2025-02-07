import abc
import logging
import warnings
from typing import Callable, Dict, List, Tuple

from commonroad.visualization.renderer import IRenderer

from commonroad_mpr.learning import FeatureExtrator, PredicateEvaluatorML as PEML
from commonroad_mpr.common.predicates import PredicateEvaluator as MprPredicateEvalutor
from commonroad_mpr.common.observation import World as WorldMPR

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


class BasePredicateEvaluator(abc.ABC):
    """
    Base class for the predicate evaluator
    """

    predicate_name = "interface"

    def __init__(self, config: CommentedMap, scaler=None):
        self.config = config
        self.eps = 1e-5
        self._scaler = scaler or RobustnessScaler(
            scale=config.setdefault("scale_rob", True)
        )
        try:
            self.feature_extractor = FeatureExtrator([self.predicate_name])
        except:
            self.feature_extractor = None

        if self.config["use_mpr"]:
            mpr_predicate_name = _map_crmonitor_predicate_name_to_mpr_predicate_name(
                self.predicate_name
            )
            try:
                self.peml = PEML([mpr_predicate_name])
            except Exception:
                logger.warning(
                    "Could not load model for predicate %s; falling back to MPR without model for this predicate.",
                    str(self.predicate_name),
                )
                try:
                    self.peml = MprPredicateEvalutor([mpr_predicate_name])
                except KeyError as e:
                    raise RuntimeError(
                        f"The predicate {mpr_predicate_name} is not supported by MPR."
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
        robustness, _ = self.peml.evaluate_robustness(
            world=world_mpr, vehicles=vehicles, time_step=time_step
        )
        return robustness[0]

    def gradient_mpr(self):
        """
        Computes the gradient of the MPR w.r.t. the input values
        """
        if self.config["use_mpr"] and self.peml is not None:
            return self.peml.derivative()[0]
        else:
            warnings.warn("The MPR is deactivated")
            return [0.0] * 35

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
            if self.config["use_mpr"] and self.peml is not None:
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
        predicate_names2vehicle_ids2values[self.predicate_name][tuple(vehicle_ids)] = (
            self.evaluate_robustness_with_cache(world, time_step, vehicle_ids)
        )

    @staticmethod
    def plot_predicate_visualization_legend(ax):
        ax.axis("off")
        ax.text(0.1, 0.5, "[not visualized]", fontsize=12)
