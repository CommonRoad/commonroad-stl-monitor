from crmonitor.predicates.base import BasePredicateEvaluator, PredicateEvaluatorConfig
from crmonitor.predicates.predicate_registry import PredicateRegistry


class PredicateFactory:
    def __init__(self, predicate_evaluator_config: PredicateEvaluatorConfig | None = None):
        self._predicate_evaluator_config = predicate_evaluator_config or PredicateEvaluatorConfig()

    def get_predicate(self, predicate_name: str) -> BasePredicateEvaluator:
        registry = PredicateRegistry.get_registry()
        evaluator_type = registry.get_predicate_evaluator(predicate_name)
        return evaluator_type(self._predicate_evaluator_config)
