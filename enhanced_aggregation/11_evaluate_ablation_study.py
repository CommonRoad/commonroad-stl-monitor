from pathlib import Path
from typing import List
import logging

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

ablation_study_results_input = Path(__file__).parent.parent / "output" / "ablation_study_results.csv"
input_scenarios = Path(__file__).parents[3] / "scenarios-for-semantic-aware-stl" / "highD"
output_file = Path(__file__).parent.parent / "output" / "ablation_study_results_processed.csv"

results_df = pd.read_csv(ablation_study_results_input)
results_df["robustness"] = results_df["robustness"].apply(
    lambda robs: list(map(float, robs.split(",")))
)
results_df.reset_index()


def evaluate_with_operator(operator, scenario: Scenario, trace, ego_vehicle_id: int) -> List[float]:
    world = World.create_from_scenario(scenario)
    ego_vehicle = world.vehicle_by_id(ego_vehicle_id)

    eval_visitor = OfflineEvaluationMonitorTreeVisitor()
    final_trace = eval_visitor.evaluate(
        operator, world, ego_vehicle, ego_vehicle.start_time, ego_vehicle.end_time
    )
    return final_trace

output_rows = []
for _, row in results_df.iterrows():
    scenario_path = input_scenarios / f"{row['scenario_id']}.xml"
    scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)
    trace = row["robustness"]

    trace_node = ConstantTraceMonitorNode("x", trace)
    for duration_sec in durations:
        rtamt_interval = RtamtInterval(begin=0, end=duration_sec, begin_unit="s")

        # Historically Duration  # TODO shorten final trace according to duration for further evaluation? → how to handle input traces that are shorter than the desired duration of the operator? nan?
        operator = HistoricallyDurationMonitorNode("g1", trace_node, interval=rtamt_interval)
        final_trace = evaluate_with_operator(operator, scenario, trace, row["vehicle_id"])
        row[f"{operator}_{duration_sec}_max"] = max(final_trace)
        row[f"{operator}_{duration_sec}_min"] = min(final_trace)
        row[f"{operator}_{duration_sec}_last"] = final_trace[-1]

        # Historically Duration Severity
        operator = HistoricallyDurationSeverityMonitorNode("g1", trace_node, interval=rtamt_interval)
        final_trace = evaluate_with_operator(operator, scenario, trace, row["vehicle_id"])
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
        final_trace = evaluate_with_operator(rule_monitor, scenario, trace, row["vehicle_id"])
        row[f"historically_{duration_sec}_max"] = max(final_trace)
        row[f"historically_{duration_sec}_min"] = min(final_trace)
        row[f"historically_{duration_sec}_last"] = final_trace[-1]

    output_rows.append(row)

pd.DataFrame(output_rows).to_csv(output_file, index=False)
