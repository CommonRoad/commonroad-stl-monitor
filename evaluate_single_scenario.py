from commonroad.common.file_reader import CommonRoadFileReader
from statistical_evaluation import CommonRoadObstacleEvaluation


def main():
    cr_eval = CommonRoadObstacleEvaluation()
    filename = "./../../../commonroad/scenarios/tum_cps/scenarios/" + "NGSIM/" + "US101/USA_US101-25_2_T-1" + ".xml"
    #filename = "./../../../commonroad/scenarios/tum_cps/scenarios/" + "SUMO/" + "DEU_Stu-1_4_T-1" + ".xml"
    scenario, planning_problem_set = CommonRoadFileReader(filename).open()

    cr_eval.evaluate_scenario(scenario, [0])

    print("number scenarios:" + str(cr_eval.num_scenarios))
    print("number vehicles: " + str(cr_eval.num_vehicles))
    print("max. speed limit compliance: " + str(cr_eval.max_speed_limit_satisfaction))
    print("min. speed limit compliance: " + str(cr_eval.min_speed_limit_satisfaction))
    print("no. unnecessary braking compliance: " + str(cr_eval.no_unnecessary_braking_satisfaction))
    print("safe distance compliance: " + str(cr_eval.safe_distance_satisfaction))
    print("perfect vehicles: " + str(cr_eval.num_veh_all_correct))


if __name__ == "__main__":
    main()
