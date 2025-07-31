import importlib
import inspect
import logging
import pkgutil
from pathlib import Path

from .base import AbstractPredicate, PredicateName

_LOGGER = logging.getLogger(__name__)


class UnkownPredicateError(Exception): ...


PREDICATE_MODULES = {"general", "position", "velocity", "acceleration", "priority"}


def _get_all_predicate_evaluators() -> dict[str, type[AbstractPredicate]]:
    package_path = Path(__file__).parent

    predicates = {}
    for _, module_name, ispkg in pkgutil.iter_modules([str(package_path)], prefix="."):
        if ispkg:
            continue

        if module_name.lstrip(".") not in PREDICATE_MODULES:
            continue

        try:
            module = importlib.import_module(module_name, package="crmonitor.predicates")
        except ImportError as e:
            # Silently skip modules that can't be imported
            _LOGGER.warning(
                "Failed to load module %s during predicate discovery: %s", module_name, e
            )
            continue

        # Find Pred* classes in this module
        module_predicates = [
            cls
            for _, cls in inspect.getmembers(
                module,
                lambda obj: inspect.isclass(obj)
                and obj.__name__.startswith("Pred")
                and obj.__module__ == module.__name__,
            )
        ]

        predicates_all_based = []
        for predicate in module_predicates:
            if not issubclass(predicate, AbstractPredicate):
                _LOGGER.warning(
                    "Discarding '%s' as possible predicate, because it is not based on '%s'",
                    predicate,
                    AbstractPredicate,
                )
                continue

            predicates_all_based.append(predicate)

        predicates_all_name = []
        for predicate in predicates_all_based:
            if not hasattr(predicate, "predicate_name"):
                _LOGGER.warning(
                    "Discarding '%s' as possible predicate, because it has no name.",
                    predicate,
                )
                continue

            if not isinstance(predicate.predicate_name, PredicateName):
                _LOGGER.warning(
                    "Discarding '%s' as possible predicate, because its name is based on %, but should be '%s'",
                    predicate,
                    type(predicate.predicate_name),
                    PredicateName,
                )
                continue

            predicates_all_name.append(predicate)

        predicates_with_names = {
            predicate.predicate_name: predicate for predicate in predicates_all_name
        }

        predicates.update(predicates_with_names)

    return predicates


class PredicateRegistry:
    """
    A singleton which provides an overview of all predicates.
    """

    _instance: "PredicateRegistry | None" = None
    _predicate_evaluators: dict[str, type[AbstractPredicate]]

    def __init__(self) -> None:
        raise RuntimeError(
            "Cannot initialize `PredicateRegistry` directly. Use `get_registry` instead."
        )

    @classmethod
    def get_registry(cls) -> "PredicateRegistry":
        if cls._instance is None:
            instance = cls.__new__(cls)
            instance._predicate_evaluators = _get_all_predicate_evaluators()
            instance._extensions = {}

            cls._instance = instance

        return cls._instance

    def get_predicate_evaluator(
        self, predicate_name: str | PredicateName
    ) -> type[AbstractPredicate]:
        if predicate_name not in self._predicate_evaluators:
            raise UnkownPredicateError(predicate_name)

        return self._predicate_evaluators[predicate_name]

    def register_predicate_evaluator(self, predicate: type[AbstractPredicate]) -> None:
        if predicate.predicate_name in self._predicate_evaluators:
            _LOGGER.warning(
                "Predicate %s is already in registry. Existing predicate will be overriden.",
                predicate.predicate_name,
            )

        self._predicate_evaluators[predicate.predicate_name] = predicate


def _get_all_predicate_evaluators_for_name_list(
    name_list: list[str],
) -> list[type[AbstractPredicate]]:
    registry = PredicateRegistry.get_registry()
    return [registry.get_predicate_evaluator(name) for name in name_list]


ALL_GENERAL_PREDICATE_NAMES = [
    "in_front_of",
    "in_same_lane",
    "keeps_safe_distance_prec",
    "brakes_abruptly",
    "brakes_abruptly_relative",
    "single_lane",
    "keeps_lane_speed_limit",
    "keeps_type_speed_limit",
    "keeps_lane_speed_limit_star",
    "preserves_traffic_flow",
    "slow_as_leading_vehicle",
]

ALL_GENERAL_PREDICATES = _get_all_predicate_evaluators_for_name_list(ALL_GENERAL_PREDICATE_NAMES)

ALL_INTERSTATE_PREDICATE_NAMES = [
    "velocity_below_five",
    "velocity_below_15",
    "velocity_below_20",
    "has_congestion_velocity",
    "has_slow_moving_velocity",
    "drives_faster",
    "drives_with_slightly_higher_speed",
    "on_shoulder",
    "in_leftmost_lane",
    "in_rightmost_lane",
    "main_carriageway_right_lane",
    "has_queue_velocity",
    "close_to_left_bound",
    "close_to_right_bound",
    "lat_left_of",
    "heading_right",
    "lat_left_of_vehicle",
    "rear_behind_front",
    "lat_close_to_vehicle_left",
    "lat_close_to_vehicle_right",
]

ALL_INTERSTATE_PREDICATES = _get_all_predicate_evaluators_for_name_list(
    ALL_INTERSTATE_PREDICATE_NAMES
)

CHANGED_TO_META_PREDICATE_NAMES = [
    "slow_leading_vehicle",  # arity
    "exist_standing_leading_vehicle",  # arity
    "in_congestion",  # arity; threshold
    "in_slow_moving_traffic",  # arity; threshold
    "in_queue_of_vehicles",  # arity; threshold
    "precedes",  # arity
    "drives_leftmost",  # scarcity
    "drives_rightmost",  # scarcity
    "cut_in",  # scarcity
    "left_of",  # scarcity
]

CHANGED_TO_META_PREDICATES = _get_all_predicate_evaluators_for_name_list(
    CHANGED_TO_META_PREDICATE_NAMES
)

META_ONLY_PREDICATE_NAMES = [
    "lon_intersecting_vehicles",
    "close_to_vehicle_left",
    "close_to_vehicle_right",
    "approach_from_left",
    "approach_from_right",
]


INSUFFICIENT_PREDICATE_NAMES = [
    "drives_with_slightly_higher_speed",
    "has_congestion_velocity",
    "heading_right",
]

INSUFFICIENT_PREDICATES = _get_all_predicate_evaluators_for_name_list(INSUFFICIENT_PREDICATE_NAMES)
