import inspect
import re
import sys
from typing import Optional

# by setting __all__ in __init__.py, all relevant modules are imported
# noinspection PyUnresolvedReferences
from crmonitor.predicates import *  # noqa: F401,F403
from crmonitor.predicates.base import BasePredicateEvaluator, PredicateEvaluatorConfig


class PredicateFactory:
    def __init__(self, predicate_evaluator_config: Optional[PredicateEvaluatorConfig] = None):
        self._predicate_evaluator_config = predicate_evaluator_config or PredicateEvaluatorConfig()
        self._evaluators = self._get_all_predicate_evaluators()

    @staticmethod
    def _get_all_predicate_evaluators():
        modules = inspect.getmembers(sys.modules["crmonitor.predicates"], inspect.ismodule)
        classes = []
        for _, module in modules:
            classes += inspect.getmembers(module, inspect.isclass)

        predicate_class_map = {
            cls.predicate_name: cls
            for name, cls in classes
            if re.match(r"^Pred[A-Z0-9].*$", name) is not None
        }
        return predicate_class_map

    def get_predicate(self, predicate_name: str) -> BasePredicateEvaluator:
        try:
            evaluator = self._evaluators[predicate_name]
        except KeyError:
            raise KeyError(f"Unknown predicate '{predicate_name}'")
        return evaluator(self._predicate_evaluator_config)
