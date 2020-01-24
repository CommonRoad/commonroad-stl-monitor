import os
import glob
import copy
import time
import argparse
import pandas as pd
import copy
import numpy as np
import pickle

import sys
#sys.path.append("/mnt/c/Users/H/DATEN/Uni/Masterthesis/masterthesis-hanna/Implementation/tests/code_other_authors/highd_generator/")

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.prediction.prediction import Trajectory
from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.file_writer import OverwriteExistingFile

from map_utils import get_meta_scenario
from obstacle_utils import generate_dynamic_obstacle
from planning_utils import get_planning_problem

AUTHOR = 'Xiao Wang'
AFFILIATION = 'Technical University of Munich, Germany'
SOURCE = 'The HighWay Drone Dataset (highD)'
TAGS = 'highway multi_lane parallel_lanes no_oncoming_traffic'

# dicts to map location id to location names and obstacle types
location_dict = {
    1: "LocationA",
    2: "LocationB",
    3: "LocationC",
    4: "LocationD",
    5: "LocationE",
    6: "LocationF"
}

def get_file_lists(path):
    listing = glob.glob(path)
    listing.sort()
    return listing

def get_args():

    parser = argparse.ArgumentParser(description="Generates CommonRoad scenarios from highD dataset")
    parser.add_argument('highd_dir', metavar='h', type=str, help='Path to highD data files')
    parser.add_argument('output_dir', metavar='o', type=str, help='Directory to store generated .xml files')
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--num_planning_problems', type=int, default=1)
    parser.add_argument('--min_time_steps', type=int, default=10)

    return parser.parse_args()

def cut_dynamic_obstacles_to_planning_interval(scenario, initial_timestep, final_timestep):

    for obstacle in scenario.dynamic_obstacles:
        obstacle_initial_time_step = obstacle.initial_state.time_step
        obstacle_final_time_step = obstacle.prediction.trajectory.state_list[-1].time_step
        if obstacle_initial_time_step > final_timestep or obstacle_final_time_step < initial_timestep:
            scenario.remove_obstacle(obstacle)

        elif obstacle_final_time_step >= final_timestep and obstacle_initial_time_step <= initial_timestep:
            if obstacle_initial_time_step == initial_timestep:
                obstacle_prediction_states = obstacle.prediction.trajectory.state_list[:final_timestep+1]
            else:
                obstacle.initial_state = obstacle.prediction.trajectory.state_list[initial_timestep - obstacle_initial_time_step - 1]
                obstacle_prediction_states = obstacle.prediction.trajectory.state_list[(initial_timestep - obstacle_initial_time_step):(final_timestep-obstacle_initial_time_step)]
            if obstacle_prediction_states:
                obstacle.prediction.trajectory = Trajectory(initial_timestep+1, obstacle_prediction_states)
            else:
                obstacle.prediction = None

        elif obstacle_final_time_step >= final_timestep and obstacle_initial_time_step > initial_timestep:
            obstacle_prediction_states = obstacle.prediction.trajectory.state_list[0:(
                                                     final_timestep - obstacle_initial_time_step)]
            if obstacle_prediction_states:
                obstacle.prediction.trajectory = Trajectory(obstacle_initial_time_step + 1, obstacle_prediction_states)
            else:
                obstacle.prediction = None
        elif obstacle_final_time_step < final_timestep and obstacle_initial_time_step <= initial_timestep:
            if obstacle_initial_time_step == initial_timestep:
                continue
            else:
                obstacle.initial_state = obstacle.prediction.trajectory.state_list[initial_timestep - obstacle_initial_time_step - 1]
                obstacle_prediction_states = obstacle.prediction.trajectory.state_list[(initial_timestep - obstacle_initial_time_step):(final_timestep-obstacle_initial_time_step)]
                if obstacle_prediction_states:
                    obstacle.prediction.trajectory = Trajectory(initial_timestep+1, obstacle_prediction_states)
                else:
                    obstacle.prediction = None
        else:
            # trajectory of obstacle fully includes in planning problem time span, nothing to do
            continue

    return scenario

def shift_obstacle_trajectories_to_initial_time_step_zero(scenario, initial_timestep):
    for obstacle in scenario.dynamic_obstacles:
        obstacle.initial_state.time_step -= initial_timestep
        if obstacle.prediction:
            for state in obstacle.prediction.trajectory.state_list:
                state.time_step -= initial_timestep
        else:
            continue

    return scenario


def shift_planning_problem_away_from_lanelet_boundaries(scenario, planning_problem):
    current_position = planning_problem.initial_state.position
    if current_position[0] < 2.7:
        current_position[0] = 2.7
    try:
        lanelet_id = scenario.lanelet_network.find_lanelet_by_position([current_position])[0][0]
    except IndexError:
        current_position[1] = current_position[1] - 1.2
        try:
            lanelet_id = scenario.lanelet_network.find_lanelet_by_position([current_position])[0][0]
        except IndexError:
            current_position[1] = current_position[1] + 2.4
            lanelet_id = scenario.lanelet_network.find_lanelet_by_position([current_position])[0][0]
    lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
    lanelet_left = lanelet.left_vertices[0][1]
    lanelet_right = lanelet.right_vertices[0][1]
    if lanelet_left - current_position[1] < 1.2:
        current_position[1] = lanelet_left - 1.2
    if current_position[1] - lanelet_right < 1.2:
        current_position[1] = lanelet_right + 1.2

    planning_problem.initial_state.position = current_position

    return planning_problem


def generate_cr_scenarios(
    recording_meta_fn,
    tracks_meta_fn,
    tracks_fn,
    min_time_steps,
    num_timesteps,
    num_planning_problems,
    output_dir
    ):
    """
    Generate CommonRoad xml files with given paths to highD recording, tracks_meta, tracks files
    :param recording_meta_fn: path to *_recordingMeta.csv
    :param tracks_meta_fn: path to *_tracksMeta.csv
    :param tracks_fn: path to *_tracks.csv
    :param min_time_steps: vehicles have to appear more than min_time_steps per .xml to be converted
    :param num_timesteps: maximal number of timesteps per .xml file
    :param num_planning_problems: number of planning problems per .xml file
    :param output_dir: path to store generated .xml files
    :return: None
    """
    def enough_time_steps(vehicle_id, tracks_meta_df, min_time_steps):
        vehicle_meta = tracks_meta_df[tracks_meta_df.id == vehicle_id]
        if frame_end - int(vehicle_meta.initialFrame) < min_time_steps or \
            int(vehicle_meta.finalFrame) - frame_start < min_time_steps:
                return False
        return True

    # read data frames from three files
    generation = True
    get_ego_vehicle = True
    ego_list = []
    recording_meta_df = pd.read_csv(recording_meta_fn, header=0)
    tracks_meta_df = pd.read_csv(tracks_meta_fn, header=0)
    tracks_df = pd.read_csv(tracks_fn, header=0)

    # generate meta scenario with lanelet network
    meta_scenario, _, _ = get_meta_scenario(recording_meta_df)

    # number of scenarios generated from this group of files
    # num_scenarios = 10
    shifting_to_zero = True  # shifts the sceanrio time steps to start with 0
    num_scenarios = max(tracks_meta_df.finalFrame) // num_timesteps
    if generation:
        for i in range(num_scenarios):

            # copy meta_scenario with lanelet networks
            scenario = copy.deepcopy(meta_scenario)

            # benchmark id format: COUNTRY_SCENE_CONFIG_PRED
            benchmark_id = "DEU_{0}-{1}_{2}_T-1".format(
                location_dict[recording_meta_df.locationId.values[0]], int(recording_meta_df.id), i+1)
            scenario.benchmark_id = benchmark_id

            # convert obstacles appearing between [frame_start, frame_end]
            frame_start = i * num_timesteps + 1
            frame_end = (i + 1) * num_timesteps

            # read tracks appear between [frame_start, frame_end]
            scenario_tracks_df = tracks_df[(tracks_df.frame >= frame_start) & (tracks_df.frame <= frame_end)]

            # generate CR obstacles
            for vehicle_id in [i for i in scenario_tracks_df.id.unique() if i in tracks_meta_df[tracks_meta_df.drivingDirection == 2].id.unique()]:#scenario_tracks_df.id.unique():
                # if appearing time steps < min_time_steps, skip vehicle
                if not enough_time_steps(vehicle_id, tracks_meta_df, min_time_steps):
                    continue
                print("Generating scenario {}/{}, vehicle id {}".format(i+1, num_scenarios, vehicle_id), end="\r")
                do = generate_dynamic_obstacle(scenario, vehicle_id, tracks_meta_df, scenario_tracks_df)
                scenario.add_objects(do)

            # generate planning problems
            planning_problem_set = PlanningProblemSet()
            for i in range(num_planning_problems):
                planning_problem, final_timestep, initial_timestep, ego_list = get_planning_problem(scenario, shifting_to_zero=shifting_to_zero, store_ego=True, ego_list=ego_list)
                planning_problem = shift_planning_problem_away_from_lanelet_boundaries(scenario, planning_problem)
                planning_problem.translate_rotate(np.array([0.0, 0.0]), -np.pi / 4)
                planning_problem_set.add_planning_problem(planning_problem)
                # reduce scenario to only relevant dynamic obstacles for planning problem
                scenario = cut_dynamic_obstacles_to_planning_interval(scenario,  int(initial_timestep), int(final_timestep))
                if shifting_to_zero:
                    scenario = shift_obstacle_trajectories_to_initial_time_step_zero(scenario,  int(initial_timestep))

            scenario.translate_rotate(np.array([0.0, 0.0]), -np.pi / 4)
            print(scenario)
            # write new scenario
            # if len(scenario.dynamic_obstacles) > 5:
            fw = CommonRoadFileWriter(scenario, planning_problem_set, AUTHOR, AFFILIATION, SOURCE, TAGS)
            filename = os.path.join(output_dir, "{}.xml".format(scenario.benchmark_id))
            fw.write_to_file(filename, OverwriteExistingFile.ALWAYS)
            print("Scenario file stored in {}".format(filename))
            #else:
            #    print('Scenario has only {} obstacles, therefore not stored'.format(len(scenario.dynamic_obstacles)))
        pickle.dump(ego_list, open(output_dir + "ego_list" + benchmark_id + ".p", "wb"))
    elif get_ego_vehicle and not generation:
        for i in range(num_scenarios):

            # copy meta_scenario with lanelet networks
            scenario = copy.deepcopy(meta_scenario)

            # benchmark id format: COUNTRY_SCENE_CONFIG_PRED
            benchmark_id = "DEU_{0}-{1}_{2}_T-1".format(
                location_dict[recording_meta_df.locationId.values[0]], int(recording_meta_df.id), i + 1)
            scenario.benchmark_id = benchmark_id

            # convert obstacles appearing between [frame_start, frame_end]
            frame_start = i * num_timesteps + 1
            frame_end = (i + 1) * num_timesteps

            # read tracks appear between [frame_start, frame_end]
            scenario_tracks_df = tracks_df[(tracks_df.frame >= frame_start) & (tracks_df.frame <= frame_end)]

            # generate CR obstacles
            for vehicle_id in [i for i in scenario_tracks_df.id.unique() if i in tracks_meta_df[
                tracks_meta_df.drivingDirection == 2].id.unique()]:  # scenario_tracks_df.id.unique():
                # if appearing time steps < min_time_steps, skip vehicle
                if not enough_time_steps(vehicle_id, tracks_meta_df, min_time_steps):
                    continue
                print("Generating scenario {}/{}, vehicle id {}".format(i + 1, num_scenarios, vehicle_id), end="\r")
                do = generate_dynamic_obstacle(scenario, vehicle_id, tracks_meta_df, scenario_tracks_df)
                scenario.add_objects(do)

            for i in range(num_planning_problems):
                _, _, _, ego_list = get_planning_problem(scenario,shifting_to_zero=shifting_to_zero, store_ego=True, ego_list=ego_list)
        pickle.dump(ego_list, open("ego_list"+benchmark_id+".p", "wb"))
    else:
        density = recording_meta_df.duration / recording_meta_df.numVehicles
        print(density)


def main():
    start_time = time.time()
    # get arguments
    args = get_args()

    # make output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # generate path to highd data files
    path_tracks = os.path.join(args.highd_dir, "data/*_tracks.csv")
    path_metas = os.path.join(args.highd_dir, "data/*_tracksMeta.csv")
    path_recording = os.path.join(args.highd_dir, "data/*_recordingMeta.csv")
    # get all file names
    listing_tracks = get_file_lists(path_tracks)
    listing_metas = get_file_lists(path_metas)
    listing_recording = get_file_lists(path_recording)

    for index, (recording_meta_fn, tracks_meta_fn, tracks_fn) in \
    enumerate(zip(listing_recording, listing_metas, listing_tracks)):
        print("="*80)
        print("Processing file {}...".format(tracks_fn), end='\n')
        generate_cr_scenarios(
            recording_meta_fn,
            tracks_meta_fn,
            tracks_fn,
            args.min_time_steps,
            args.num_timesteps,
            args.num_planning_problems,
            args.output_dir
        )

    print("Elapsed time: {} s".format(time.time() - start_time), end="\r")

if __name__ == "__main__":
    # sys.argv = ['highd_to_cr.py', 'highD/', 'scenarios_highD/']
    main()
