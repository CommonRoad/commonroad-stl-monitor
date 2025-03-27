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
    RuleMonitorNode,
)
from crmonitor.monitor.rtamt_monitor_stl import RtamtStlMonitor
from crmonitor.rule.rule_node import HistoricallyDurationSeverityNode, IOType
from rtamt.semantics.interval.interval import Interval as RtamtInterval

operator_types = [HistoricallyDurationMonitorNode, HistoricallyDurationSeverityNode]
durations = [1, 2, 5, 10]

ablation_study_results = Path(__file__).parent.parent / "output" / "ablation_study_results.csv"
input_scenarios = Path(__file__).parent.parent.parent / "highD-scenarios"
output_file = ablation_study_results

results_df = pd.read_csv(ablation_study_results)


def evaluate_with_operator(operator, scenario: Scenario, trace, ego_vehicle_id: int) -> List[float]:
    world = World.create_from_scenario(scenario)
    ego_vehicle = world.vehicle_by_id(ego_vehicle_id)

    eval_visitor = OfflineEvaluationMonitorTreeVisitor()
    final_trace = eval_visitor.walk(operator, world, ego_vehicle.end_time, ego_vehicle)
    return final_trace


for _, row in results_df.iterrows():
    scenario_path = input_scenarios / f"{row['scenario_id']}.xml"
    scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)
    trace = list(map(float, row["robustness"].split(",")))

    trace_node = ConstantTraceMonitorNode("x", trace)
    for duration_sec in durations:
        rtamt_interval = RtamtInterval(begin=0, end=duration_sec, begin_unit="s")
        operator = HistoricallyDurationMonitorNode("g1", trace_node, interval=rtamt_interval)
        final_trace = evaluate_with_operator(operator, scenario, trace, row["vehicle_id"])

        duration_time_step = duration_sec * scenario.dt
        row[(str(operator), duration_sec), "max"] = max(final_trace)
        row[(str(operator), duration_sec), "min"] = min(final_trace)
        row[(str(operator), duration_sec), "last"] = final_trace[-1]

        rtamt_stl_monitor = RtamtStlMonitor(
            f"always[0,{duration_sec}s](x)", predicates=[("x", IOType.OUTPUT)], dt=scenario.dt
        )
        rule_monitor = RuleMonitorNode(name="g1", children=[trace_node], monitor=rtamt_stl_monitor)
        final_trace = evaluate_with_operator(rule_monitor, scenario, trace, row["vehicle_id"])

        row[str(rule_monitor)] = final_trace

# results_df.to_csv(ablation_study_results, index=False)
