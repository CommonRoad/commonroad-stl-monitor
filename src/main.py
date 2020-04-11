import os
import argparse
import time
import sys

from src.common.helper import *
from src.common.commonroad_evaluation import CommonRoadObstacleEvaluation

from commonroad.scenario.scenario import Tag
from commonroad.common.file_reader import CommonRoadFileReader


def create_scenarios_from_directory(directories: List[str], max_num_scenarios: int = sys.maxsize):
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
            if Tag.HIGHWAY in scenario.tags:
                scenarios.append(scenario)
            if len(scenarios) == max_num_scenarios:
                break
        if len(scenarios) == max_num_scenarios:
            break

    return scenarios


def get_args():

    parser = argparse.ArgumentParser(description="Traffic Rule Evaluation of CommonRoad scenarios")
    parser.add_argument('--max_num_scenarios', type=int, help='Maximum number of scenarios to evaluate.')
    parser.add_argument('--scenario_directories', nargs='+', help='List of directories where scenarios are located.')

    return parser.parse_args()


def main():
    print("start time:" + str(time.time()))

    cr_eval = CommonRoadObstacleEvaluation("")
    if cr_eval.simulation_param.get("single_scenario"):
        scenario, planning_problem_set = CommonRoadFileReader("./../scenarios/"
                                                              + cr_eval.simulation_param.get("scenario_dir")
                                                              + cr_eval.simulation_param.get("benchmark_id")
                                                              + ".xml").open()
        result = cr_eval.evaluate_scenario(scenario, cr_eval.simulation_param.get("rule_set"))
        print(result)
    else:
        args = get_args()
        if args.scenario_directories is None:
            scenario_directories = cr_eval.simulation_param.get("scenario_dir")
        else:
            scenario_directories = args.scenario_directories
        if args.max_num_scenarios is None:
            max_num_scenarios = cr_eval.simulation_param.get("max_num_scenarios")
        else:
            max_num_scenarios = args.max_num_scenarios
        if max_num_scenarios < 0:
            max_num_scenarios = sys.maxsize

        scenarios = create_scenarios_from_directory(scenario_directories, max_num_scenarios)
        for sc in scenarios:
            result = cr_eval.evaluate_scenario(sc, cr_eval.simulation_param.get("rule_set"))
            print(result)

    print("end time:" + str(time.time()))


if __name__ == "__main__":
    main()
