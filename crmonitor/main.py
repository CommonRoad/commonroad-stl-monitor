import os
import argparse
import time
import sys
import time
import multiprocessing

from crmonitor.common.helper import *
from crmonitor.common.commonroad_evaluation import CommonRoadObstacleEvaluation

from commonroad.scenario.scenario import Tag
from commonroad.common.file_reader import CommonRoadFileReader


def create_scenarios_from_directory(
    directories: List[str], max_num_scenarios: int = sys.maxsize
):
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
            if filename.startswith("C-"):
                continue
            elif "DEU" not in filename:
                continue
            elif "_S-" in filename:
                continue
            elif not filename.endswith(".xml"):
                continue
            fullname = os.path.join(abs_path, filename)
            scenario, planning_problem_set = CommonRoadFileReader(fullname).open(
                lanelet_assignment=True
            )
            if Tag.INTERSTATE in scenario.tags or Tag.INTERSTATE in scenario.tags:
                scenarios.append(scenario)
            if len(scenarios) == max_num_scenarios:
                break
        if len(scenarios) == max_num_scenarios:
            break

    return scenarios


def get_args():

    parser = argparse.ArgumentParser(
        description="Traffic Rule Evaluation of CommonRoad scenarios"
    )
    parser.add_argument("--evaluation_mode", help="Evaluation mode for execution.")
    parser.add_argument(
        "--max_num_scenarios",
        default=2,
        type=int,
        help="Maximum number of scenarios to evaluate.",
    )
    parser.add_argument(
        "--num_cores",
        default=1,
        type=int,
        help="Number of processor cores which should be used.",
    )
    parser.add_argument(
        "--scenario_directories",
        nargs="+",
        help="List of directories where scenarios are located.",
    )

    return parser.parse_args()


def main():
    start_time = time.time()

    cr_eval = CommonRoadObstacleEvaluation(
        os.path.dirname(os.path.abspath(__file__)) + "/"
    )
    args = get_args()

    if args.evaluation_mode is None:
        scenario, planning_problem_set = CommonRoadFileReader(
            os.path.dirname(os.path.abspath(__file__))
            + cr_eval.simulation_param.get("scenario_dir")
            + "/"
            + cr_eval.simulation_param.get("benchmark_id")
            + ".xml"
        ).open(lanelet_assignment=True)
        result = cr_eval.evaluate_scenario(scenario)
        print(result)
    else:
        if args.scenario_directories is None:
            scenario_directories = cr_eval.simulation_param.get("scenario_directories")
        else:
            scenario_directories = args.scenario_directories
        if args.evaluation_mode is not None:
            cr_eval.simulation_param["evaluation_mode"] = args.evaluation_mode
        if args.max_num_scenarios < 0:
            max_num_scenarios = 2
        else:
            max_num_scenarios = args.max_num_scenarios

        scenarios = create_scenarios_from_directory(
            scenario_directories, max_num_scenarios
        )

        pool = multiprocessing.Pool(processes=args.num_cores)
        pool.map(
            cr_eval.evaluate_scenario, (scenarios[idx] for idx in range(len(scenarios)))
        )

    print(cr_eval.eval_dict)
    print("Num. scenarios: " + str(cr_eval.num_scenarios))
    print("Num. vehicles: " + str(cr_eval.num_vehicles))
    print("Num. all correct: " + str(cr_eval.num_veh_all_correct))
    print("comp. time:" + str(time.time() - start_time))


if __name__ == "__main__":
    main()
