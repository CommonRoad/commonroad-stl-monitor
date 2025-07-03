from pathlib import Path
from typing import List, Type

import pytest
from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.world import World
from crmonitor.predicates.base import AbstractPredicate, PredicateConfig
from crmonitor.predicates.position import PredInSameLane

PREDICATE_TEST_CASES = [(PredInSameLane, "", 30, [30, 31])]


@pytest.parameterize("predicate_cls,scenario_path,time_step,vehicle_ids", PREDICATE_TEST_CASES)
def test_predicates_boolean_matches_mfr_robustness(
    predicate_cls: Type[AbstractPredicate],
    scenario_path: str,
    time_step: int,
    vehicle_ids: List[int],
) -> None:
    config = PredicateConfig()
    predicate_evaluator = predicate_cls(config)

    scenario_path = Path(__file__).parents[2] / "scenarios" / scenario_path

    scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)
    world = World.create_from_scenario(scenario)

    boolean_result = predicate_evaluator.evaluate_boolean(world, time_step, vehicle_ids)
    mfr_result = predicate_evaluator.evaluate_robustness(world, time_step, vehicle_ids)

    assert boolean_result == (mfr_result >= 0.0)
