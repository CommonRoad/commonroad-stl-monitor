import matplotlib.pyplot as plt
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_dispatch_cr import draw_object
from common.configuration import *
import numpy as np
from commonroad.common.file_writer import CommonRoadFileWriter
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
        'draw_shape': True,
        'draw_icon': False,
        'draw_bounding_box': True,
        'show_label': False,
        'trajectory_steps': 1,
        'zorder': 100,
        'occupancy': {
            'draw_occupancies': -1,  # -1= never, 0= if prediction of vehicle is set-based, 1=always
            'shape': {
                'polygon': {
                    'opacity': 1.0,
                    'facecolor': '#0f55a3',
                    'edgecolor': '#0f55a3',
                    'zorder': 100,
                },
                'rectangle': {
                    'opacity': 1,
                    'facecolor': '#0f55a3',
                    'edgecolor': '#0f55a3',
                    'zorder': 18,
                },
                'circle': {
                    'opacity': 0.2,
                    'facecolor': '#1d7eea',
                    'edgecolor': '#0066cc',
                    'zorder': 18,
                }
            },
        },
        'shape': {
            'polygon': basic_shape_parameters_dynamic,
            'rectangle': basic_shape_parameters_dynamic,
            'circle': basic_shape_parameters_dynamic
        },
        'trajectory': {'facecolor': '#0f55a3'}
    },
    'static_obstacle': {
        'shape': {
            'polygon': basic_shape_parameters_static,
            'rectangle': basic_shape_parameters_static,
            'circle': basic_shape_parameters_static,
        }
    },
    'lanelet_network': {
        'lanelet': {'left_bound_color': '#555555',
                    'right_bound_color': '#555555',
                    'center_bound_color': '#dddddd',
                    'draw_left_bound': True,
                    'draw_right_bound': True,
                    'draw_center_bound': True,
                    'draw_border_vertices': False,
                    'draw_start_and_direction': True,
                    'show_label': True,
                    'draw_linewidth': 0.5,
                    'fill_lanelet': True,
                    'facecolor': '#e8e8e8'}},
},
}

config = load_yaml("config.yaml")
simulation_param = config.get("simulation_param")

filename = "./../../../commonroad/scenarios/tum_cps/scenarios/" + "NGSIM/" + "US101/USA_US101-14_1_T-1" + ".xml"
    #simulation_param.get("commonroad_scenario_folder") + simulation_param.get("commonroad_benchmark_id") + ".xml"
scenario, planning_problem_set = CommonRoadFileReader(filename).open()
#scenario.translate_rotate(np.array([0, 0]), -0.030)
#fw = CommonRoadFileWriter(scenario, planning_problem_set, scenario.author, scenario.affiliation, scenario.source, scenario.tags)
#fw.write_scenario_to_file("DEU_A9-3_1_T-1.xml")

plt.style.use('classic')
inch_in_cm = 2.54
figsize = [20, 8]
x = [x for lanelet in scenario.lanelet_network.lanelets for x in lanelet.center_vertices[:, 0]]
y = [y for lanelet in scenario.lanelet_network.lanelets for y in lanelet.center_vertices[:, 1]]
x_min = min(x) - 10
y_min = min(y) - 10
x_max = max(x) + 5
y_max = max(y) + 5
plot_limits = [x_min, x_max, y_min , y_max]

plt.figure(figsize=(8, 4.5))
plt.gca().axis('equal')
draw_params_scenario['scenario']['dynamic_obstacle']['show_label'] = True
draw_params_scenario['scenario']['lanelet_network']['lanelet']['show_label'] = True
draw_params_scenario['scenario']['dynamic_obstacle']['shape']['rectangle']['facecolor'] = '#0070fe'
draw_params_scenario['scenario']['dynamic_obstacle']['shape']['rectangle']['edgecolor'] = '#0070fe'
draw_params_scenario['scenario']['dynamic_obstacle']['occupancy']['shape']['polygon']['opacity'] = .1
draw_object(scenario, draw_params=draw_params_scenario, plot_limits=plot_limits)
#draw_object(planning_problem_set, draw_params=draw_params_scenario, plot_limits=plot_limits)
plt.show()
