from pathlib import Path

from commonroad_mpr.common.observation import Scenario
import pandas as pd
from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.world import World
from crmonitor.evaluation.visitor import OfflineEvaluationMonitorTreeVisitor
from crmonitor.monitor.monitor_node import ConstantTraceMonitorNode, HistoricallyDurationMonitorNode
from crmonitor.rule.rule_node import HistoricallyDurationSeverityNode

operator_types = [HistoricallyDurationMonitorNode, HistoricallyDurationSeverityNode]
durations = [1, 2, 5, 10]
# Add parameters for the operator, e.g., for `HistoricallyDurationMonitorNode` you can add an interval.
operator_params = {}

ablation_study_results = Path(__file__).parent.parent / "output" / "ablation_study_results.csv"
input_scenarios = Path(__file__).parent.parent.parent / "highD-scenarios"
output_file = ablation_study_results

results_df = pd.read_csv(ablation_study_results)


def evaluate_with_operator_type(
    operator, scenario_path: Path, trace, ego_vehicle_id: int
) -> List[float]:
    scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)
    world = World.create_from_scenario(scenario)
    ego_vehicle = world.vehicle_by_id(ego_vehicle_id)

    eval_visitor = OfflineEvaluationMonitorTreeVisitor()
    final_trace = eval_visitor.walk(operator, world, ego_vehicle.end_time, ego_vehicle)
    return final_trace


for _, row in results_df.iterrows():
    scenario_path = input_scenarios / f"{row['scenario_id']}.xml"
    trace = list(map(float, row["robustness"].split(",")))

    trace_node = ConstantTraceMonitorNode("x", trace)
    operator = operator_type("g1", trace_node, **operator_params)
    row[str(operator)] = final_trace

results_df.to_csv(ablation_study_results, index=False)
