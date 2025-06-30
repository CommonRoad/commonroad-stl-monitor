from crmonitor.predicates.base import AbstractPredicate, PredicateConfig
from crmonitor.predicates.predicate_registry import PredicateRegistry


class PredicateFactory:
    def __init__(self, predicate_evaluator_config: PredicateConfig | None = None):
        self._predicate_evaluator_config = predicate_evaluator_config or PredicateConfig()

    def get_predicate(self, predicate_name: str) -> AbstractPredicate:
        registry = PredicateRegistry.get_registry()
        evaluator_type = registry.get_predicate_evaluator(predicate_name)
        return evaluator_type(self._predicate_evaluator_config)
