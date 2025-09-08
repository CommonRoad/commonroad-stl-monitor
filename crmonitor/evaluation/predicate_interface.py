import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
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


FloatPredicateCache = TimeStepCache[tuple[ScenarioID, str, tuple[int, ...]], float]
BoolPredicateCache = TimeStepCache[tuple[ScenarioID, str, tuple[int, ...]], bool]


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
    _robustness_predicate_cache: FloatPredicateCache | None
    _satisfied_predicate_cache: BoolPredicateCache | None

    def __init__(
        self,
        predicate: type[AbstractPredicate] | str,
        config: PredicateEvaluationInterfaceConfig | None = None,
        robustness_predicate_cache: FloatPredicateCache | None = None,
        satisfied_predicate_cache: BoolPredicateCache | None = None,
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
        self._robustness_predicate_cache = robustness_predicate_cache
        self._satisfied_predicate_cache = satisfied_predicate_cache

        self._setup_predicate_evaluator(predicate)

        if self._config.mode == PredicateEvaluationMode.MPR_GP:
            self._setup_mpr_gp_evaluator()
            # Also setup the MPR evaluator to enable automatic fallback from GPs.
            self._setup_mpr_evaluator(mpr_cache)
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

    def _setup_mpr_gp_evaluator(self) -> None:
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

    def evaluate_boolean(self, world: World, time_step: int, vehicle_ids: tuple[int, ...]) -> bool:
        """Evaluate predicate as a boolean value.

        :param world: World for evaluation.
        :param time_step: Time step to evaluate at.
        :param vehicle_ids: Vehicle IDs to evaluate for.

        :returns: Boolean evaluation result
        """

        cached_satisfied = self._get_bool_predicate_cache_entry(world, time_step, vehicle_ids)
        if cached_satisfied is not None:
            return cached_satisfied

        predicate_satisfied = self._predicate_evaluator.evaluate_boolean(
            world, time_step, vehicle_ids
        )

        self._set_bool_predicate_cache_entry(world, time_step, vehicle_ids, predicate_satisfied)

        return predicate_satisfied

    def _get_bool_predicate_cache_entry(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> bool | None:
        if self._satisfied_predicate_cache is None:
            return None

        return self._satisfied_predicate_cache.get_at_time_step(
            time_step, (world.scenario_id, self.predicate_name, vehicle_ids)
        )

    def _set_bool_predicate_cache_entry(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...], satisfied: bool
    ) -> None:
        if self._robustness_predicate_cache is None:
            return None

        self._robustness_predicate_cache.set_at_time_step(
            time_step,
            (world.scenario_id, self.predicate_name, vehicle_ids),
            satisfied,
        )

    def _evaluate_robustness_mpr(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        assert self._mpr_evaluator is not None
        _LOGGER.debug(
            "Evaluating predicate %s on %s at time step %s for vehicles %s with model-predictive robustness",
            self.predicate_name,
            world.scenario_id,
            time_step,
            vehicle_ids,
        )
        mpr_result_dict = self._mpr_evaluator.evaluate(world, time_step, vehicle_ids)

        robustness = mpr_result_dict[self._predicate_evaluator.predicate_name].robustness
        return robustness

    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        cached_robustness = self._get_float_predicate_cache_entry(world, time_step, vehicle_ids)
        if cached_robustness is not None:
            return cached_robustness

        if self._mpr_gp_evaluator is not None:
            _LOGGER.debug(
                "Evaluating predicate %s on %s at time step %s for vehicles %s with gaussian processes.",
                self.predicate_name,
                world.scenario_id,
                time_step,
                vehicle_ids,
            )
            mpr_gp_results_dict = self._mpr_gp_evaluator.evaluate(world, time_step, vehicle_ids)

            # The MPR GP results are a dict indexed by predicate names. We only require the result for the predicate tracked by this evaluator.
            predicate_mpr_gp_result = mpr_gp_results_dict[self.predicate_name]

            if not predicate_mpr_gp_result.prediction_matches_reality():
                # MPR evaluation with GPs might produce robustness values which do not match the real satisfaction of the predicate.
                # If this happens the robustness can either be rectified to a pre-defined robustness,
                # or we fallback to evaluating the predicate with standard MPR.
                if self._mpr_gp_evaluator.config.rectification:
                    robustness = float(1e-3) * np.sign(predicate_mpr_gp_result.characteristic_value)
                else:
                    robustness = self._evaluate_robustness_mpr(world, time_step, vehicle_ids)
            else:
                robustness = predicate_mpr_gp_result.robustness

        elif self._mpr_evaluator is not None:
            robustness = self._evaluate_robustness_mpr(world, time_step, vehicle_ids)
        else:
            _LOGGER.debug(
                "Evaluating predicate %s on %s at time step %s for vehicles %s with model-free robustness",
                self.predicate_name,
                world.scenario_id,
                time_step,
                vehicle_ids,
            )
            robustness = self._predicate_evaluator.evaluate_robustness(
                world, time_step, vehicle_ids
            )

        self._set_float_predicate_cache_entry(world, time_step, vehicle_ids, robustness)

        return robustness

    def _get_float_predicate_cache_entry(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float | None:
        if self._robustness_predicate_cache is None:
            return None

        return self._robustness_predicate_cache.get_at_time_step(
            time_step, (world.scenario_id, self.predicate_name, vehicle_ids)
        )

    def _set_float_predicate_cache_entry(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...], robustness: float
    ) -> None:
        if self._robustness_predicate_cache is None:
            return None

        self._robustness_predicate_cache.set_at_time_step(
            time_step,
            (world.scenario_id, self.predicate_name, vehicle_ids),
            robustness,
        )

    def reset(self) -> None:
        if self._robustness_predicate_cache:
            self._robustness_predicate_cache.invalidate()

        if self._satisfied_predicate_cache:
            self._satisfied_predicate_cache.invalidate()


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
        # The MPR state sampling cache must be setup at this level or hight, to make sure that
        # we really benefit from the caching. Each single predicate evaluator, might sample
        # for the same vehicle, so a cache that is shared across all predicates makes sure
        # we do not have to resample that often.
        self._mpr_cache = _MprSampledStateCacheWrapper()

        self._predicate_interfaces = {}
        for predicate in predicates:
            predicate_interface = SinglePredicateEvaluationInterface(
                predicate,
                config,
                # The robustness and satisfied caches are specific to each predicate.
                # Since we moved to meta predicates, there is also no recursive predicate invocation anymore
                # so shared caching is not necessary anymore.
                robustness_predicate_cache=BasicTimeStepCache(),
                satisfied_predicate_cache=BasicTimeStepCache(),
                mpr_cache=self._mpr_cache,
            )
            self._predicate_interfaces[predicate_interface.predicate_name] = predicate_interface

    def evaluate_boolean(
        self, predicate: str, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> bool:
        if predicate not in self._predicate_interfaces:
            raise ValueError(
                f"Cannot evaluate predicate '{predicate}': predicate is not setup for this predicate interface!"
            )
        predicate_interface = self._predicate_interfaces[predicate]

        return predicate_interface.evaluate_boolean(world, time_step, vehicle_ids)

    def evaluate_robustness(
        self, predicate: str, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        if predicate not in self._predicate_interfaces:
            raise ValueError(
                f"Cannot evaluate predicate '{predicate}': predicate is not setup for this predicate interface!"
            )
        predicate_interface = self._predicate_interfaces[predicate]

        return predicate_interface.evaluate_robustness(world, time_step, vehicle_ids)

    def reset(self) -> None:
        self._mpr_cache.invalidate()

        for predicate_interface in self._predicate_interfaces.values():
            predicate_interface.reset()
