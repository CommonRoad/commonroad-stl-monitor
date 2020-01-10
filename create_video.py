from commonroad.common.file_reader import CommonRoadFileReader
from output.visualization import create_scenario_video
from common.configuration import *

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
visualization_param = config.get("visualization").get("video")

filename = simulation_param.get("commonroad_scenario_folder") + simulation_param.get("commonroad_benchmark_id") + ".xml"
scenario, planning_problem_set = CommonRoadFileReader(filename).open()
create_scenario_video(simulation_param.get("video_output_folder"), scenario, visualization_param, 50)