from common.configuration import *
from monitor.traffic_rule_dispatcher import TrafficRuleDispatcher
from commonroad.common.file_reader import CommonRoadFileReader
from src.common.commonroad_evaluation import CommonRoadObstacleEvaluation
import os


def main():
    cr_eval = CommonRoadObstacleEvaluation("../")
    scenarios = []
    root_dir_cr = "./../../../commonroad/scenarios/tum_cps/scenarios"
    root_dir_hd = "./highD_generator/scenarios"
    max_num_scenarios = 100

    for subdir, dirs, files in os.walk(root_dir_cr):
        for directory in dirs:
            if directory == "cooperative":
                continue
            for filename in os.listdir(subdir + "/" + directory):
                if not "DEU" in filename:
                    continue
                if "Stu" in filename:
                    continue
                if not filename.endswith('.xml') or "_S-" in filename:
                    continue
                fullname = os.path.join(subdir + "/" + directory, filename)
                scenario, planning_problem_set = \
                    CommonRoadFileReader(fullname).open()
                if "highway" in scenario.tags:
                    scenarios.append(scenario)
                if len(scenarios) > max_num_scenarios:
                    break
            if len(scenarios) > max_num_scenarios:
                break
        if len(scenarios) > max_num_scenarios:
            break

    for filename in os.listdir(root_dir_hd):
        if not "DEU" in filename:
            continue
        if "Stu" in filename:
            continue
        if not filename.endswith('.xml') or "_S-" in filename:
            continue
        fullname = os.path.join(root_dir_hd + "/", filename)
        scenario, planning_problem_set = \
            CommonRoadFileReader(fullname).open()
        if "highway" in scenario.tags:
            scenarios.append(scenario)
        if len(scenarios) > max_num_scenarios:
            break

    for sc in scenarios:
        cr_eval.evaluate_scenario(sc, ["S0"])

    print("number scenarios:" + str(cr_eval.num_scenarios))
    print("number vehicles: " + str(cr_eval.num_vehicles))
    print("max. speed limit compliance: " + str(cr_eval.max_speed_limit_satisfaction))
    print("min. speed limit compliance: " + str(cr_eval.min_speed_limit_satisfaction))
    print("no. unnecessary braking compliance: " + str(cr_eval.no_unnecessary_braking_satisfaction))
    print("safe distance compliance: " + str(cr_eval.safe_distance_satisfaction))
    print("perfect vehicles: " + str(cr_eval.num_veh_all_correct))


if __name__ == "__main__":
    main()
