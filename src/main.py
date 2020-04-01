import os

from src.common.helper import *
from src.common.commonroad_evaluation import CommonRoadObstacleEvaluation

from commonroad.common.file_reader import CommonRoadFileReader


def create_scenarios_from_directory(directories: List[str], max_num_scenarios: int = 100):
    """
    Creation of CommonRoad scenarios from CommonRoad XML-files which are located in provided directories

    :param directories: directories where XML files are located
    :param max_num_scenarios: maximum number of scenarios which should be created
    :returns list of CommonRoad scenarios
    """
    scenarios = []
    for scenario_dir in directories:
        abs_path = os.path.abspath(os.getcwd() + scenario_dir)
        for filename in os.listdir(abs_path):
            if filename.startswith('C-'):
                continue
            elif "DEU" not in filename:
                continue
            elif "_S-" in filename:
                continue
            elif not filename.endswith('.xml'):
                continue
            fullname = os.path.join(abs_path, filename)
            scenario, planning_problem_set = CommonRoadFileReader(fullname).open()
            if "highway" in scenario.tags:
                scenarios.append(scenario)
            if len(scenarios) > max_num_scenarios:
                break
        if len(scenarios) > max_num_scenarios:
            break

    return scenarios


def main():
    cr_eval = CommonRoadObstacleEvaluation("")
    if cr_eval.simulation_param.get("single_scenario"):
        scenario, planning_problem_set = CommonRoadFileReader("./../scenarios/" + cr_eval.simulation_param.get("scenario_folder") +
                                                              cr_eval.simulation_param.get("benchmark_id") +
                                                              ".xml").open()
        result = cr_eval.evaluate_scenario(scenario, cr_eval.simulation_param.get("rule_set"))
        print(result)
    else:
        scenarios = create_scenarios_from_directory(cr_eval.simulation_param.get("evaluation_folders"), 10)
        for sc in scenarios:
            result = cr_eval.evaluate_scenario(sc, cr_eval.simulation_param.get("rule_set"))
            print(result)


if __name__ == "__main__":
    main()
