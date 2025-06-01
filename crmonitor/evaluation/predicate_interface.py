import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, auto

from crmonitor.common import World
from crmonitor.mpr import (
    ModelLoadError,
    MprGpPredicateEvaluator,
    MprGpPredicateEvaluatorConfig,
    MprPredicateEvaluator,
    MprPredicateEvaluatorConfig,
)
from crmonitor.predicates import BasePredicateEvaluator, PredicateName
from crmonitor.predicates.base import PredicateEvaluatorConfig

_LOGGER = logging.getLogger(__name__)


class PredicateEvaluationMode(Enum):
    BOOLEAN = auto()
    MFR = auto()
    MPR = auto()
    MPR_GP = auto()


@dataclass
class PredicateInterfaceConfig:
    mode: PredicateEvaluationMode = PredicateEvaluationMode.MFR

    base: PredicateEvaluatorConfig = field(default_factory=PredicateEvaluatorConfig)
    mpr: MprPredicateEvaluatorConfig | None = None
    mpr_gp: MprGpPredicateEvaluatorConfig | None = None


class PredicateInterface:
    def __init__(
        self,
        predicates: Iterable[BasePredicateEvaluator],
        config: PredicateInterfaceConfig | None = None,
    ) -> None:
        self._predicate_evaluators = predicates
        if config is None:
            config = PredicateInterfaceConfig()
        self._config = config
        self._mode = self._config.mode

        self._mpr_gp_evaluator = None
        self._mpr_evaluator = None
        if self._mode == PredicateEvaluationMode.MPR_GP:
            self._setup_mpr_gp_evaluator()
        elif self._mode == PredicateEvaluationMode.MPR:
            self._setup_mpr_evaluator()

    def _setup_mpr_gp_evaluator(self) -> None:
        try:
            self._mpr_gp_evaluator = MprGpPredicateEvaluator(
                self._predicate_evaluators, config=self._config.mpr_gp
            )
        except ModelLoadError as e:
            _LOGGER.warning(
                "Failed to load pre-trained model for predicate '%s' from path '%s'. Falling back to model-predictive evaluation without a pre-trained model.",
                e.predicate_name,
                e.model_path,
            )
            self._setup_mpr_evaluator()

    def _setup_mpr_evaluator(self) -> None:
        self._mpr_evaluator = MprPredicateEvaluator(
            predicates=self._predicate_evaluators, config=self._config.mpr
        )

    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> dict[PredicateName, float]:
        if self._mpr_gp_evaluator is not None:
            mpr_gp_result_dict = self._mpr_gp_evaluator.evaluate(world, time_step, vehicle_ids)
            result_dict = {}
            for predicate_name, result in mpr_gp_result_dict.items():
                result_dict[predicate_name] = result.robustness
            return result_dict
        elif self._mpr_evaluator is not None:
            mpr_result_dict = self._mpr_evaluator.evaluate(world, time_step, vehicle_ids)
            result_dict = {}
            for predicate_name, result in mpr_result_dict.items():
                result_dict[predicate_name] = result.robustness
            return result_dict
        else:
            result_dict = {}
            for predicate_evaluator in self._predicate_evaluators:
                robustness = predicate_evaluator.evaluate_robustness(world, time_step, vehicle_ids)
                result_dict[predicate_evaluator.predicate_name] = robustness
            return result_dict
