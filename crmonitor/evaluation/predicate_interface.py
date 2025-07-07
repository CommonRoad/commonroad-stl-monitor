import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, auto

from commonroad.scenario.scenario import ScenarioID

from crmonitor.common import World
from crmonitor.common.cache import BasicTimeStepCache, TimeStepCache
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
    """Supported predicate evaluation modes."""

    MFR = auto()
    """Model-free robustness."""

    MPR = auto()
    """Model-predictive robustness."""

    MPR_GP = auto()
    """Model-predictive robustness with gaussian processes."""


@dataclass
class PredicateEvaluationInterfaceConfig:
    """Configuration for predicate evaluation interface."""

    mode: PredicateEvaluationMode = PredicateEvaluationMode.MFR
    """Select the mode in which predicates will be evaluated. All predicates will be evaluated with the same mode."""

    base: PredicateConfig = field(default_factory=PredicateConfig)
    """Provide configuration for the basic predicate evaluator."""

    mpr: MprPredicateEvaluatorConfig | None = None
    """Optionally configure the model-predictive evaluation. If None is set and MPR is selected as predicate evaluation mode, the default config is used."""

    mpr_gp: MprGpPredicateEvaluatorConfig | None = None
    """Optionally configure the model-predictive evaluation with gaussian processes. If None is set and MPR_GP is selected as predicate evaluation mode, the default config is used."""


PredicateCache = TimeStepCache[tuple[ScenarioID, str, tuple[int, ...]], float]


class SinglePredicateEvaluationInterface:
    """Interface for evaluating a single predicate across different modes.

    Handles mode-specific setup and provides a unified evaluation API. Automatically
    falls back from MPR-GP to MPR when model loading fails, ensuring robust operation
    even when pre-trained models are unavailable.

    The interface abstracts away the complexity of different evaluation modes while
    maintaining consistent behavior across all modes.
    """

    _config: PredicateEvaluationInterfaceConfig
    _predicate_evaluator: AbstractPredicate
    _mpr_gp_evaluator: MprGpPredicateEvaluator | None = None
    _mpr_evaluator: MprPredicateEvaluator | None = None

    def __init__(
        self,
        predicate: type[AbstractPredicate] | str,
        config: PredicateEvaluationInterfaceConfig | None = None,
        predicate_cache: PredicateCache | None = None,
        mpr_cache: MprSampledStatesCache | None = None,
    ) -> None:
        """Initialize the predicate evaluation interface.

        :param predicate: Predicate class or name to evaluate.
        :param config: Evaluation configuration.
        :param mpr_cache: Shared cache for MPR state sampling results.
        """
        if config is None:
            config = PredicateEvaluationInterfaceConfig()
        self._config = config
        self._predicate_cache = predicate_cache

        self._setup_predicate_evaluator(predicate)

        if self._config.mode == PredicateEvaluationMode.MPR_GP:
            self._setup_mpr_gp_evaluator(mpr_cache)
        elif self._config.mode == PredicateEvaluationMode.MPR:
            self._setup_mpr_evaluator(mpr_cache)

    @property
    def predicate_name(self) -> PredicateName:
        """Get the name of the predicate being evaluated.

        :returns: The predicate name
        """
        return self._predicate_evaluator.predicate_name

    def _setup_predicate_evaluator(
        self, predicate: type[AbstractPredicate] | str | PredicateName
    ) -> None:
        """Setup the base predicate evaluator from class or registry lookup."""
        if isinstance(predicate, str):
            self._predicate_evaluator = PredicateRegistry.get_registry().get_predicate_evaluator(
                predicate
            )(self._config.base)
        else:
            self._predicate_evaluator = predicate(self._config.base)

    def _setup_mpr_evaluator(self, mpr_cache: MprSampledStatesCache | None) -> None:
        """Setup standard MPR evaluator with optional state caching."""
        self._mpr_evaluator = MprPredicateEvaluator(
            predicates=[self._predicate_evaluator],
            config=self._config.mpr,
            state_sampling_cache=mpr_cache,
        )

    def _setup_mpr_gp_evaluator(self, mpr_cache: MprSampledStatesCache | None) -> None:
        """Setup MPR-GP evaluator with automatic fallback to standard MPR.

        Attempts to load pre-trained models for GP-based evaluation. If model loading
        fails, automatically falls back to standard MPR evaluation.
        """
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

    def evaluate_boolean(self, world: World, time_step: int, vehicle_ids: tuple[int, ...]) -> bool:
        """Evaluate predicate as a boolean value.

        :param world: World for evaluation.
        :param time_step: Time step to evaluate at.
        :param vehicle_ids: Vehicle IDs to evaluate for.

        :returns: Boolean evaluation result
        """
        return self._predicate_evaluator.evaluate_boolean(world, time_step, vehicle_ids)

    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        cached_robustness = self._get_predicate_cache_entry(world, time_step, vehicle_ids)
        if cached_robustness is not None:
            return cached_robustness

        if self._mpr_gp_evaluator is not None:
            _LOGGER.debug(
                "Evaluating predicate %s on %s at time step %s for vehicles %s with gaussian processes.",
                self.predicate_name,
                world.scenario.scenario_id,
                time_step,
                vehicle_ids,
            )
            mpr_gp_result_dict = self._mpr_gp_evaluator.evaluate(world, time_step, vehicle_ids)

            robustness = mpr_gp_result_dict[self._predicate_evaluator.predicate_name].robustness
        elif self._mpr_evaluator is not None:
            _LOGGER.debug(
                "Evaluating predicate %s on %s at time step %s for vehicles %s with model-predictive robustness",
                self.predicate_name,
                world.scenario.scenario_id,
                time_step,
                vehicle_ids,
            )
            mpr_result_dict = self._mpr_evaluator.evaluate(world, time_step, vehicle_ids)

            robustness = mpr_result_dict[self._predicate_evaluator.predicate_name].robustness
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

        self._set_predicate_cache_entry(world, time_step, vehicle_ids, robustness)

        return robustness

    def _get_predicate_cache_entry(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float | None:
        if self._predicate_cache is None:
            return None

        return self._predicate_cache.get_at_time_step(
            time_step, (world.scenario.scenario_id, self.predicate_name, vehicle_ids)
        )

    def _set_predicate_cache_entry(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...], robustness: float
    ) -> None:
        if self._predicate_cache is None:
            return None

        self._predicate_cache.set_at_time_step(
            time_step, (world.scenario.scenario_id, self.predicate_name, vehicle_ids), robustness
        )


class _MprSampledStateCacheWrapper:
    """Internal wrapper for caching MPR state sampling results.

    Provides a simplified interface for caching expensive state sampling computations
    that can be shared across multiple predicates. Uses time-step based caching to
    efficiently store and retrieve results.
    """

    _internal_cache: TimeStepCache[int, StateBasedSamplingResult]

    def __init__(self) -> None:
        self._internal_cache = BasicTimeStepCache()

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

    def invalidate(self) -> None:
        self._internal_cache.invalidate()


class PredicateEvaluationInterface:
    """Main interface for evaluating multiple predicates with shared caching.

    Groups predicates under a common interface to enable consistent caching across
    all predicates. State sampling results are shared between predicates to avoid
    redundant computation, significantly improving performance when evaluating
    multiple predicates on the same scenario.

    The interface provides a unified API regardless of the underlying evaluation
    modes, making it easy to switch between different evaluation strategies.
    """

    def __init__(
        self,
        predicates: Iterable[type[AbstractPredicate] | str],
        config: PredicateEvaluationInterfaceConfig | None = None,
    ) -> None:
        self._predicate_cache = BasicTimeStepCache()
        self._mpr_cache = _MprSampledStateCacheWrapper()

        self._predicate_interfaces = {}
        for predicate in predicates:
            predicate_interface = SinglePredicateEvaluationInterface(
                predicate, config, self._predicate_cache, self._mpr_cache
            )
            self._predicate_interfaces[predicate_interface.predicate_name] = predicate_interface

    def evaluate_boolean(
        self, predicate: str, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> bool:
        predicate_interface = self._predicate_interfaces[predicate]

        return predicate_interface.evaluate_boolean(world, time_step, vehicle_ids)

    def evaluate_robustness(
        self, predicate: str, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        predicate_interface = self._predicate_interfaces[predicate]

        return predicate_interface.evaluate_robustness(world, time_step, vehicle_ids)

    def reset(self) -> None:
        self._predicate_cache.invalidate()
        self._mpr_cache.invalidate()
