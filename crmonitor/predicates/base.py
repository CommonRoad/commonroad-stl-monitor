import abc
import copy
import logging
import warnings
from typing import Callable, Dict, List, Tuple

from commonroad.visualization.renderer import IRenderer
from commonroad_mpr.common.observation import World as WorldMPR
from commonroad_mpr.prediction.ego_sampling import StateBasedSampling
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as MprCfg
from ruamel.yaml.comments import CommentedMap

from crmonitor.common.world import World
from crmonitor.predicates.scaling import RobustnessScaler

logger = logging.getLogger(__name__)


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

        self._use_mpr_for_evaluation = self.config["use_mpr"]

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
        ego_vehicle_id = vehicle_ids[0]
        ego_vehicle_mpr = world_mpr.vehicle_by_id(ego_vehicle_id)
        ego_sampler = StateBasedSampling(
            ego_vehicle_mpr,
            time_step,
            MprCfg["common"]["scenario"],
            **MprCfg["sampling_approach"]["state_based_sampling"][
                MprCfg["common"]["scenario"]
            ],
        )

        orig_ego_vehicle = world.vehicle_by_id(ego_vehicle_id)
        ego_vehicle = copy.deepcopy(orig_ego_vehicle)
        ego_vehicle.signal_series[time_step] = None

        world.remove_vehicle(orig_ego_vehicle)
        world.add_vehicle(ego_vehicle)

        count_valid = 0
        count_true = 0
        count_error = 0
        for ego_future_state_mpr in ego_sampler.sample():
            try:
                ego_future_state = ego_future_state_mpr.convert_to_commonroad_state()
                ego_vehicle.states_cr[time_step] = ego_future_state
                ego_loc_shape = ego_vehicle.shape.rotate_translate_local(
                    ego_future_state.position, ego_future_state.orientation
                )
                lanelet_assignment = (
                    world.road_network.lanelet_network.find_lanelet_by_shape(
                        ego_loc_shape
                    )
                )
                if len(lanelet_assignment) == 0:
                    count_error += 1
                    continue
                ego_vehicle.lanelet_assignment[time_step] = lanelet_assignment

                satisfied = self.evaluate_boolean(world, time_step, vehicle_ids)
                if satisfied:
                    count_true += 1
                count_valid += 1
            except Exception as e:
                raise e
                count_error += 1

        world.remove_vehicle(ego_vehicle)
        world.add_vehicle(orig_ego_vehicle)

        satisfied = self.evaluate_boolean(world, time_step, vehicle_ids)
        probability = count_true / (count_valid + MprCfg["robustness"]["eps"])
        robustness = probability if satisfied else -(1 - probability)

        return robustness

    def gradient_mpr(self):
        """
        Computes the gradient of the MPR w.r.t. the input values
        """
        default = [0.0] * 35
        # TODO: reenable
        return default
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
