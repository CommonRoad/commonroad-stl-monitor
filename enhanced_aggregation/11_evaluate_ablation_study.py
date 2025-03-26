from pathlib import Path

import pandas as pd
from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.world import World
from crmonitor.evaluation.visitor import OfflineEvaluationMonitorTreeVisitor
from crmonitor.monitor.monitor_node import ConstantTraceMonitorNode, HistoricallyDurationMonitorNode

operator_type = HistoricallyDurationMonitorNode
# Add parameters for the operator, e.g., for `HistoricallyDurationMonitorNode` you can add an interval.
operator_params = {}

ablation_study_results = Path(__file__).parent.parent / "output" / "ablation_study_results.csv"
input_scenarios = Path(__file__).parent.parent.parent / "highD-scenarios"
output_file = ablation_study_results

results_df = pd.read_csv(ablation_study_results)
results_df["robustness"] = results_df["robustness"].apply(
    lambda robs: list(map(float, robs.split(",")))
)
results_df.reset_index()

for _, row in results_df.iterrows():
    scenario_path = input_scenarios / f"{row['scenario_id']}.xml"
    scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)

    world = World.create_from_scenario(scenario)
    ego_vehicle = world.vehicle_by_id(row["vehicle_id"])

    trace_node = ConstantTraceMonitorNode("x", row["robustness"])
    operator = operator_type("g1", trace_node, **operator_params)
    eval_visitor = OfflineEvaluationMonitorTreeVisitor()
    final_trace = eval_visitor.walk(operator, world, row["end_time_step"], ego_vehicle)
    row[str(operator)] = final_trace

results_df.to_csv(ablation_study_results, index=False)
