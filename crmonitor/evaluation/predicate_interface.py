import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, auto

from crmonitor.common import World
from crmonitor.common.cache import LinearTimeStepCache, TimeStepCache
from crmonitor.mpr import (
    ModelLoadError,
    MprGpPredicateEvaluator,
    MprGpPredicateEvaluatorConfig,
    MprPredicateEvaluator,
    MprPredicateEvaluatorConfig,
)
from crmonitor.mpr.mpr_predicate_evaluator import MprSampledStatesCache
from crmonitor.mpr.prediction.state_sampling import StateBasedSamplingResult
from crmonitor.predicates import AbstractPredicate, PredicateName
from crmonitor.predicates.base import PredicateConfig
from crmonitor.predicates.predicate_registry import PredicateRegistry

_LOGGER = logging.getLogger(__name__)


class PredicateEvaluationMode(Enum):
    BOOLEAN = auto()
    MFR = auto()
    MPR = auto()
    MPR_GP = auto()


@dataclass
class PredicateEvaluationInterfaceConfig:
    mode: PredicateEvaluationMode = PredicateEvaluationMode.MFR

    base: PredicateConfig = field(default_factory=PredicateConfig)
    mpr: MprPredicateEvaluatorConfig | None = None
    mpr_gp: MprGpPredicateEvaluatorConfig | None = None


class SinglePredicateEvaluationInterface:
    _config: PredicateEvaluationInterfaceConfig
    _predicate_evaluator: AbstractPredicate
    _mpr_gp_evaluator: MprGpPredicateEvaluator | None = None
    _mpr_evaluator: MprPredicateEvaluator | None = None

    def __init__(
        self,
        predicate: type[AbstractPredicate] | str,
        config: PredicateEvaluationInterfaceConfig | None = None,
        mpr_cache: MprSampledStatesCache | None = None,
    ) -> None:
        if config is None:
            config = PredicateEvaluationInterfaceConfig()
        self._config = config

        self._setup_predicate_evaluator(predicate)

        if self._config.mode == PredicateEvaluationMode.MPR_GP:
            self._setup_mpr_gp_evaluator(mpr_cache)
        elif self._config.mode == PredicateEvaluationMode.MPR:
            self._setup_mpr_evaluator(mpr_cache)

    @property
    def predicate_name(self) -> PredicateName:
        return self._predicate_evaluator.predicate_name

    def _setup_predicate_evaluator(
        self, predicate: type[AbstractPredicate] | str | PredicateName
    ) -> None:
        if isinstance(predicate, str):
            self._predicate_evaluator = PredicateRegistry.get_registry().get_predicate_evaluator(
                predicate
            )(self._config.base)
        else:
            self._predicate_evaluator = predicate(self._config.base)

    def _setup_mpr_evaluator(self, mpr_cache: MprSampledStatesCache | None) -> None:
        self._mpr_evaluator = MprPredicateEvaluator(
            predicates=[self._predicate_evaluator],
            config=self._config.mpr,
            state_sampling_cache=mpr_cache,
        )

    def _setup_mpr_gp_evaluator(self, mpr_cache: MprSampledStatesCache | None) -> None:
        try:
            self._mpr_gp_evaluator = MprGpPredicateEvaluator(
                [self._predicate_evaluator], config=self._config.mpr_gp
            )
        except ModelLoadError as e:
            _LOGGER.warning(
                "Failed to load pre-trained model for predicate '%s' from path '%s'. Falling back to model-predictive evaluation without a pre-trained model.",
                e.predicate_name,
                e.model_path,
            )
            self._setup_mpr_evaluator(mpr_cache)

    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        if self._mpr_gp_evaluator is not None:
            _LOGGER.debug(
                "Evaluating predicate %s on %s at time step %s for vehicles %s with gaussian processes.",
                self.predicate_name,
                world.scenario.scenario_id,
                time_step,
                vehicle_ids,
            )
            mpr_gp_result_dict = self._mpr_gp_evaluator.evaluate(world, time_step, vehicle_ids)

            return mpr_gp_result_dict[self._predicate_evaluator.predicate_name].robustness
        elif self._mpr_evaluator is not None:
            _LOGGER.debug(
                "Evaluating predicate %s on %s at time step %s for vehicles %s with model-predictive robustness",
                self.predicate_name,
                world.scenario.scenario_id,
                time_step,
                vehicle_ids,
            )
            mpr_result_dict = self._mpr_evaluator.evaluate(world, time_step, vehicle_ids)

            return mpr_result_dict[self._predicate_evaluator.predicate_name].robustness
        else:
            _LOGGER.debug(
                "Evaluating predicate %s on %s at time step %s for vehicles %s with model-free robustness",
                self.predicate_name,
                world.scenario.scenario_id,
                time_step,
                vehicle_ids,
            )
            robustness = self._predicate_evaluator.evaluate_robustness(
                world, time_step, vehicle_ids
            )
            return robustness


class _MprSampledStateCacheWrapper:
    _internal_cache: TimeStepCache[int, StateBasedSamplingResult]

    def __init__(self) -> None:
        self._internal_cache = LinearTimeStepCache()

    def set_sampling_result(
        self,
        time_step: int,
        vehicle_id: int,
        result: StateBasedSamplingResult,
    ) -> None:
        self._internal_cache.set_at_time_step(time_step, vehicle_id, result)

    def get_sampling_result(
        self, time_step: int, vehicle_id: int
    ) -> StateBasedSamplingResult | None:
        return self._internal_cache.get_at_time_step(time_step, vehicle_id)


class PredicateEvaluationInterface:
    def __init__(
        self,
        predicates: Iterable[type[AbstractPredicate] | str],
        config: PredicateEvaluationInterfaceConfig | None = None,
    ) -> None:
        mpr_cache = _MprSampledStateCacheWrapper()

        self._predicate_interfaces = {}
        for predicate in predicates:
            predicate_interface = SinglePredicateEvaluationInterface(predicate, config, mpr_cache)
            self._predicate_interfaces[predicate_interface.predicate_name] = predicate_interface

    def evaluate_robustness(
        self, predicate: str, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        predicate_interface = self._predicate_interfaces[predicate]

        return predicate_interface.evaluate_robustness(world, time_step, vehicle_ids)
