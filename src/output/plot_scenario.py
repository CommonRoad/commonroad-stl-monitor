import matplotlib.pyplot as plt
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_dispatch_cr import draw_object
from src.common.helper import *
#from src.output.visualization import create_scenario_video
from commonroad.visualization.video import create_scenario_video

# CommonRoad Visualization Parameters:
basic_shape_parameters_static = {'opacity': 1.0,
                                 'facecolor': '#0f55a3',
                                 'edgecolor': '#0f55a3',
                                 'zorder': 20}

basic_shape_parameters_dynamic = {'opacity': 1.0,
                                  'facecolor': '#ff4000',
                                  'edgecolor': '#ff4000',
                                  'zorder': 100}

draw_params_scenario = {'scenario': {
    'dynamic_obstacle': {
        'add_legend': True,
        'draw_shape': True,
        'draw_icon': False,
        'draw_bounding_box': True,
        'show_label': False,
        'zorder': 20,
        'occupancy': {
            'draw_occupancies': 0,  # -1= never, 0= if prediction of vehicle is set-based, 1=always
            'shape': {
                'polygon': {
                    'opacity': 0.2,
                    'facecolor': '#1d7eea',
                    'edgecolor': '#0066cc',
                    'linewidth': 0.5,
                    'zorder': 18,
                },
                'rectangle': {
                    'opacity': 0.2,
                    'facecolor': '#1d7eea',
                    'edgecolor': '#0066cc',
                    'linewidth': 0.5,
                    'zorder': 18,
                },
                'circle': {
                    'opacity': 0.2,
                    'facecolor': '#1d7eea',
                    'edgecolor': '#0066cc',
                    'linewidth': 0.5,
                    'zorder': 18,
                }
            },
        },
        'shape': {
            'polygon': basic_shape_parameters_dynamic,
            'rectangle': basic_shape_parameters_dynamic,
            'circle': basic_shape_parameters_dynamic
        },
        'trajectory': {'draw_trajectory': True,
                       'facecolor': '#000000',
                       'draw_continuous': False,
                       'unique_colors': False,
                       'line_width': 0.17,
                       'z_order': 24}
    },
    'static_obstacle': {
        'shape': {
            'polygon': basic_shape_parameters_static,
            'rectangle': basic_shape_parameters_static,
            'circle': basic_shape_parameters_static,
        }
    },
    'lanelet_network': {
        'draw_traffic_lights': True,
        'traffic_light': {'red_color': 'red',
                          'yellow_color': '#feb609',
                          'green_color': '#00aa16',
                          'red_yellow_color': '#fe4009ff'},
        'draw_traffic_signs': True,
        'draw_signs_in_lanelet': True,  # Todo: False not implemented
        'draw_intersections': True,
        'intersection': {'draw_incoming_lanelets': False,
                         'incoming_lanelets_color': '#3ecbcf',
                         'draw_crossings': False,
                         'crossings_color': '#b62a55',
                         'draw_successors': False,
                         'successors_left_color': '#ff00ff',
                         'successors_straight_color': 'blue',
                         'successors_right_color': '#ccff00'},
        'lanelet': {'left_bound_color': '#555555',
                    'right_bound_color': '#555555',
                    'center_bound_color': '#dddddd',
                    'unique_colors': False,  # colorizes center_vertices and labels of each lanelet differently
                    'draw_stop_line': True,
                    'stop_line_color': '#ffffff',
                    'draw_line_markings': True,
                    'draw_left_bound': True,
                    'draw_right_bound': True,
                    'draw_center_bound': True,
                    'draw_border_vertices': False,
                    'draw_start_and_direction': True,
                    'show_label': False,  # show lanelet_id
                    'draw_linewidth': 0.5,
                    'fill_lanelet': True,
                    'facecolor': '#c7c7c7'}},
}
}

config = load_yaml("../config.yaml")
simulation_param = config.get("simulation_param")
visualization_param = config.get("visualization").get("video")
#filename = "./../../../commonroad/scenarios/tum_cps/scenarios/" + "NGSIM/" + "US101/USA_US101-25_2_T-1" + ".xml"
filename = "../../scenarios/USA_Lanker-1_16_T-1.xml"
#filename = "./../../../commonroad/scenarios/tum_cps/scenarios/" + "SUMO/" + "DEU_Stu-1_4_T-1" + ".xml"
#filename = "./../../../commonroad/scenarios/tum_cps/scenarios/" + "hand-crafted/" + "DEU_A9-1_1_T-1" + ".xml"
#filename = "highD_generator/scenarios/DEU_LocationB-1_17_T-1.xml"
    #simulation_param.get("commonroad_scenario_folder") + simulation_param.get("commonroad_benchmark_id") + ".xml"
scenario, planning_problem_set = CommonRoadFileReader(filename).open()

#plt.style.use('classic')
inch_in_cm = 2.54
figsize = [20, 8]
x = [state.position[0] for obstacle in scenario.dynamic_obstacles for state in obstacle.prediction.trajectory.state_list]
y = [state.position[1] for obstacle in scenario.dynamic_obstacles for state in obstacle.prediction.trajectory.state_list]
x_min = min(x) - 5
y_min = min(y) - 5
x_max = max(x) + 5
y_max = max(y) + 5
plot_limits = [x_min, x_max, y_min, y_max]

plt.figure(figsize=(8, 4.5))
plt.gca().axis('equal')
draw_params_scenario['scenario']['dynamic_obstacle']['show_label'] = False
draw_params_scenario['scenario']['lanelet_network']['lanelet']['show_label'] = False
draw_params_scenario['scenario']['dynamic_obstacle']['shape']['rectangle']['facecolor'] = '#0070fe'
draw_params_scenario['scenario']['dynamic_obstacle']['shape']['rectangle']['edgecolor'] = '#0070fe'
draw_params_scenario['scenario']['dynamic_obstacle']['occupancy']['shape']['polygon']['opacity'] = .1
draw_object(scenario, draw_params=draw_params_scenario, plot_limits=plot_limits)
#draw_object(list(planning_problem_set.planning_problem_dict.values())[0])
#draw_object(planning_problem_set, draw_params=draw_params_scenario, plot_limits=plot_limits)
plt.axis('off')
plt.show()

#create_scenario_video("videos/", scenario, visualization_param, 20)
#create_scenario_video(scenario, "test.mp4", 0, 30, 0, plot_limits)