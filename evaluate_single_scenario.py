from commonroad.common.file_reader import CommonRoadFileReader
from statistical_evaluation import CommonRoadObstacleEvaluation
from common.vehicle import Vehicle
from predicates.vehicle_state_predicates import VehicleStatePredicateCollection
import math
import matplotlib.pyplot as plt
import matplotlib as mp

def main():
    cr_eval = CommonRoadObstacleEvaluation()
    #filename = "./../../../commonroad/scenarios/tum_cps/scenarios/" + "NGSIM/" + "US101/USA_US101-25_2_T-1" + ".xml"
    #filename = "./../../../commonroad/scenarios/tum_cps/scenarios/" + "SUMO/" + "DEU_A99-1_2_T-1" + ".xml"
    filename = "./../../../commonroad/scenarios/tum_cps/scenarios/" + "hand-crafted/" + "DEU_A99-1_1_T-1" + ".xml"
    #filename = "highD_generator/scenarios/DEU_LocationB-2_13_T-1.xml"
    scenario, planning_problem_set = CommonRoadFileReader(filename).open()

    cr_eval.evaluate_scenario(scenario, [0])

    print("number scenarios:" + str(cr_eval.num_scenarios))
    print("number vehicles: " + str(cr_eval.num_vehicles))
    print("max. speed limit compliance: " + str(cr_eval.max_speed_limit_satisfaction))
    print("min. speed limit compliance: " + str(cr_eval.min_speed_limit_satisfaction))
    print("no. unnecessary braking compliance: " + str(cr_eval.no_unnecessary_braking_satisfaction))
    print("safe distance compliance: " + str(cr_eval.safe_distance_satisfaction))
    print("perfect vehicles: " + str(cr_eval.num_veh_all_correct))

    create_safe_distance_plot(cr_eval.vehicles_dict[204], cr_eval.vehicles_dict[202])


def create_safe_distance_plot(vehicle_follow: Vehicle, vehicle_lead: Vehicle):
    time = []
    delta_s = []
    s_safe = []
    for time_step, state in vehicle_follow.states_lon.items():
        time.append(time_step)
        delta_s.append(vehicle_lead.states_lon[time_step].s - vehicle_follow.states_lon[time_step].s)
        s_safe.append(VehicleStatePredicateCollection.safe_distance(vehicle_follow.states_lon[time_step].v,
                                                                    vehicle_lead.states_lon[time_step].v,
                                                                    -10, -10.5, 0.3))

    # Storage related configuration
    width = 3.75
    height = width * (math.sqrt(5) - 1.0) / 2.0
    figsize = [width, height]
    linewidth_plot = 0.75
    mp.rcParams.update({'font.size': 9})
    mp.rcParams.update({'axes.linewidth': 0.25})
    mp.rcParams.update({'figure.autolayout': True})
    mp.rcParams.update({'legend.frameon': False})
    mp.rcParams['svg.fonttype'] = 'none'

    plt.figure(1, figsize=figsize)
    plt.ylabel(r'$s~[m]$')
    plt.xlabel(r'$t~[s]$')
    time = [i * 0.1 for i in time]
    plt.plot(time, delta_s, color=(0.0, 0.0, 0.5, 1), label=r'$\Delta s$', linewidth=linewidth_plot)
    plt.plot(time, s_safe, "-", color=(0.3, 0.3, 0.3, 0.35), label=r'$d_{safe}$', linewidth=linewidth_plot)
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()
