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
            if Tag.HIGHWAY in scenario.tags or Tag.INTERSTATE in scenario.tags:
                scenarios.append(scenario)
            if len(scenarios) == max_num_scenarios:
                break
        if len(scenarios) == max_num_scenarios:
            break

    return scenarios


def get_args():

    parser = argparse.ArgumentParser(description="Traffic Rule Evaluation of CommonRoad scenarios")
    parser.add_argument('--max_num_scenarios', default=2, type=int, help='Maximum number of scenarios to evaluate.')
    parser.add_argument('--num_vehicles', default=1, type=int, help='Number of vehicles to evaluate.')
    parser.add_argument('--scenario_directories', nargs='+', help='List of directories where scenarios are located.')

    return parser.parse_args()


def main():
    start_time = time.time()

    cr_eval = CommonRoadObstacleEvaluation(os.path.dirname(os.path.abspath(__file__)) + "/")
    if cr_eval.simulation_param.get("operating_mode") == "single_scenario" \
            or cr_eval.simulation_param.get("operating_mode") == "single_vehicle":
        scenario, planning_problem_set = CommonRoadFileReader(os.path.dirname(os.path.abspath(__file__))
                                                              + cr_eval.simulation_param.get("scenario_dir") + "/"
                                                              + cr_eval.simulation_param.get("benchmark_id")
                                                              + ".xml").open()
        result = cr_eval.evaluate_scenario(scenario)
        print(result)
    else:
        args = get_args()
        if args.scenario_directories is None:
            scenario_directories = cr_eval.simulation_param.get("scenario_directories")
        else:
            cr_eval.simulation_param["operating_mode"] = "evaluation"
            scenario_directories = args.scenario_directories
        if args.max_num_scenarios is None:
            max_num_scenarios = cr_eval.simulation_param.get("max_num_scenarios")
        else:
            cr_eval.simulation_param["operating_mode"] = "evaluation"
            max_num_scenarios = args.max_num_scenarios
        if max_num_scenarios < 0:
            max_num_scenarios = sys.maxsize
        if args.num_vehicles is not None:
            if args.num_vehicles < 0:
                cr_eval.simulation_param["num_vehicles"] = sys.maxsize
            else:
                cr_eval.simulation_param["num_vehicles"] = args.num_vehicles

        scenarios = create_scenarios_from_directory(scenario_directories, max_num_scenarios)
        for sc in scenarios:
            result = cr_eval.evaluate_scenario(sc)
            print(result)
            if cr_eval.simulation_param.get("num_vehicles") <= cr_eval.num_vehicles:
                break

    print(cr_eval.eval_dict)
    print("Num. scenarios: " + str(cr_eval.num_scenarios))
    print("Num. vehicles: " + str(cr_eval.num_vehicles))
    print("Num. all correct: " + str(cr_eval.num_veh_all_correct))
    print("comp. time:" + str(time.time() - start_time))


if __name__ == "__main__":
    main()
