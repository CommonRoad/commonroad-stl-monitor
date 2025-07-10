import numpy as np
import pytest
from crmonitor.mpr.mpr_gp_predicate_evaluator import (
    MprGpPredicateEvaluator,
    MprGpPredicateEvaluatorConfig,
)
from crmonitor.predicates.position import PredInFrontOf
from crmonitor.predicates.velocity import PredVelocityBelow5
from tests.resources import InterstateScenarios


class TestMprGpPredicateEvaluator:
    @pytest.mark.parametrize(
        "predicate,scenario,time_step,vehicle_ids,robustness",
        [
            (PredVelocityBelow5, InterstateScenarios.SAFE_DISTANCE_LANE_CHANGE, 10, (1002,), -0.94),
            (PredInFrontOf, InterstateScenarios.SAFE_DISTANCE_LANE_CHANGE, 50, (1005, 1004), 0.02),
        ],
    )
    def test_evaluation(self, predicate, scenario, time_step, vehicle_ids, robustness):
        config = MprGpPredicateEvaluatorConfig()

        evaluator = MprGpPredicateEvaluator([predicate()], config=config)

        world = scenario.get_world()
        result = evaluator.evaluate(world=world, time_step=time_step, vehicle_ids=vehicle_ids)

        assert predicate.predicate_name in result, (
            f"Predicate {predicate.predicate_name} is not part of evaluation result {result}"
        )

        predicate_result = result[predicate.predicate_name]

        assert np.isclose(round(predicate_result.robustness, 2), robustness)
