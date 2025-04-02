import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

import pandas as pd
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario
from crmonitor.common.world import World
from crmonitor.evaluation.visitor import OfflineEvaluationMonitorTreeVisitor
from crmonitor.monitor.monitor_node import (
    ConstantTraceMonitorNode,
    HistoricallyDurationMonitorNode,
    HistoricallyDurationSeverityMonitorNode,
    RtamtRuleMonitorNode,
)
from crmonitor.monitor.rtamt_monitor_stl import RtamtStlMonitor
from crmonitor.rule.rule_node import IOType
from rtamt.semantics.interval.interval import Interval as RtamtInterval

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)
_LOGGER.warning("Fix ToDos!")

operator_types = [HistoricallyDurationMonitorNode, HistoricallyDurationSeverityMonitorNode]
durations = [1, 2, 5, 10]

ablation_study_results_input = (
    Path(__file__).parent.parent / "output" / "ablation_study_results.csv"
)
input_scenarios = Path(__file__).parents[3] / "scenarios-for-semantic-aware-stl" / "highD"
output_file = Path(__file__).parent.parent / "output" / "ablation_study_results_processed.csv"

results_df = pd.read_csv(ablation_study_results_input)


@lru_cache()
def load_scenario(scenario_id: str) -> Scenario:
    scenario_path = input_scenarios / f"{scenario_id}.xml"
    scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)
    return scenario


class VehicleProxy:
    id: int = 0


@dataclass
class WorldProxy:
    scenario: Scenario


def evaluate_with_operator(
    operator, trace, world: World, start_time: int, end_time: int
) -> List[float]:
    ego_vehicle = VehicleProxy()
    eval_visitor = OfflineEvaluationMonitorTreeVisitor()
    final_trace = eval_visitor.evaluate(operator, world, ego_vehicle, start_time, end_time)
    return final_trace


output_rows = []
for _, row in results_df.iterrows():
    trace = list(map(float, row["robustness"].split(",")))
    # Need to load scenario, because evaluation requires dt for interpretation of the operator durations.
    scenario = load_scenario(row["scenario_id"])
    # Construct only a proxy instead of a full world object, since only the scenario is needed for the evaluation of the operators.
    world = WorldProxy(scenario)

    result = {}
    trace_node = ConstantTraceMonitorNode("x", trace)
    for duration_sec in durations:
        rtamt_interval = RtamtInterval(begin=0, end=duration_sec, begin_unit="s")

        # Historically Duration  # TODO shorten final trace according to duration for further evaluation? → how to handle input traces that are shorter than the desired duration of the operator? nan?
        operator = HistoricallyDurationMonitorNode("g1", trace_node, interval=rtamt_interval)
        final_trace = evaluate_with_operator(
            operator, trace, world, row["start_time_step"], row["end_time_step"]
        )
        row[f"{operator}_{duration_sec}_max"] = max(final_trace)
        row[f"{operator}_{duration_sec}_min"] = min(final_trace)
        row[f"{operator}_{duration_sec}_last"] = final_trace[-1]

        # Historically Duration Severity
        operator = HistoricallyDurationSeverityMonitorNode(
            "g1", trace_node, interval=rtamt_interval
        )
        final_trace = evaluate_with_operator(
            operator, trace, world, row["start_time_step"], row["end_time_step"]
        )
        row[f"{operator}_{duration_sec}_max"] = max(final_trace)
        row[f"{operator}_{duration_sec}_min"] = min(final_trace)
        row[f"{operator}_{duration_sec}_last"] = final_trace[-1]

        # Historically  # TODO is this correct?
        rtamt_stl_monitor = RtamtStlMonitor(
            f"historically[0,{duration_sec}s](x)", predicates=[("x", IOType.OUTPUT)], dt=scenario.dt
        )
        rule_monitor = RtamtRuleMonitorNode(
            name="g1", children=[trace_node], monitor=rtamt_stl_monitor
        )
        final_trace = evaluate_with_operator(
            rule_monitor, trace, world, row["start_time_step"], row["end_time_step"]
        )
        row[f"historically_{duration_sec}_max"] = max(final_trace)
        row[f"historically_{duration_sec}_min"] = min(final_trace)
        row[f"historically_{duration_sec}_last"] = final_trace[-1]

    output_rows.append(row)


pd.DataFrame(output_rows).to_csv(output_file, index=False)
