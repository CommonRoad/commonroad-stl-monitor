import logging
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
from crmonitor.predicates.predicate_registry import PredicateRegistry

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
    _config: PredicateInterfaceConfig
    _predicate_evaluator: BasePredicateEvaluator
    _mpr_gp_evaluator: MprGpPredicateEvaluator | None = None
    _mpr_evaluator: MprPredicateEvaluator | None = None

    def __init__(
        self,
        predicate: type[BasePredicateEvaluator] | str,
        config: PredicateInterfaceConfig | None = None,
    ) -> None:
        if config is None:
            config = PredicateInterfaceConfig()
        self._config = config

        self._setup_predicate_evaluator(predicate)

        if self._config.mode == PredicateEvaluationMode.MPR_GP:
            self._setup_mpr_gp_evaluator()
        elif self._config.mode == PredicateEvaluationMode.MPR:
            self._setup_mpr_evaluator()

    def _setup_predicate_evaluator(
        self, predicate: type[BasePredicateEvaluator] | str | PredicateName
    ) -> None:
        if isinstance(predicate, str):
            self._predicate_evaluator = PredicateRegistry.get_registry().get_predicate_evaluator(
                predicate
            )(self._config.base)
        else:
            self._predicate_evaluator = predicate(self._config.base)

    def _setup_mpr_gp_evaluator(self) -> None:
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
            self._setup_mpr_evaluator()

    def _setup_mpr_evaluator(self) -> None:
        self._mpr_evaluator = MprPredicateEvaluator(
            predicates=[self._predicate_evaluator], config=self._config.mpr
        )

    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        if self._mpr_gp_evaluator is not None:
            mpr_gp_result_dict = self._mpr_gp_evaluator.evaluate(world, time_step, vehicle_ids)

            return mpr_gp_result_dict[self._predicate_evaluator.predicate_name].robustness
        elif self._mpr_evaluator is not None:
            mpr_result_dict = self._mpr_evaluator.evaluate(world, time_step, vehicle_ids)

            return mpr_result_dict[self._predicate_evaluator.predicate_name].robustness
        else:
            robustness = self._predicate_evaluator.evaluate_robustness(
                world, time_step, vehicle_ids
            )
            return robustness
