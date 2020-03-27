from commonroad.common.file_reader import CommonRoadFileReader
from src.output.visualization import Visualization

scenario, planning_problem_set = CommonRoadFileReader("./../../scenarios/test/test_max_speed_limit.xml").open()
vis = Visualization(figsize=(9.75, 2))
vis.plot_scenario(scenario, time_begin=20)