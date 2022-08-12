import importlib.resources as pkg_resources
import logging
from collections import defaultdict
from functools import lru_cache
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from commonroad.visualization.mp_renderer import MPRenderer
from matplotlib import pyplot as plt

import crmonitor
from crmonitor.common.helper import load_yaml, merge_dicts_recursively
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world import World
from crmonitor.evaluation.visitor import (
    MonitorCreationRuleTreeVisitor,
    EvaluationMonitorTreeVisitor,
    PredicateCollectorMonitorTreeVisitor,
    ResetMonitorTreeVisitor,
    PredicateVisualizerMonitorTreeVisitor,
)
from crmonitor.monitor.rtamt_monitor_stl import OutputType
from crmonitor.predicates.rule import VisitorNode, parse_rule

logger = logging.getLogger(__name__)


EGO_VEHICLE_DRAW_PARAMS = {
    "dynamic_obstacle": {
        "vehicle_shape": {
            "occupancy": {"shape": {"rectangle": {"facecolor": "yellow"}}}
        }
    }
}


@lru_cache(maxsize=None)
def get_traffic_rule_config():
    with pkg_resources.path(
        crmonitor, "traffic_rules_rtamt.yaml"
    ) as traffic_rules_path:
        traffic_rules_config = load_yaml(traffic_rules_path)
    return traffic_rules_config



class RuleEvaluator:
    @classmethod
    def create_from_config(
        cls,
        world: World = None,
        ego_vehicle: Vehicle = None,
        rule: str = "R_G1",
        traffic_rules_config=None,
        use_boolean: bool = False,
        output_type: OutputType = OutputType.STANDARD,
    ):
        if traffic_rules_config is None:
            traffic_rules_config = get_traffic_rule_config()
        rule_str_dict = traffic_rules_config["traffic_rules"]
        rule_set = parse_rule(rule_str_dict[rule], traffic_rules_config, name=rule)
        return cls(
            rule_set,
            ego_vehicle,
            world,
            use_boolean=use_boolean,
            output_type=output_type,
        )

    def __init__(
        self,
        rule: VisitorNode,
        ego_vehicle: Vehicle,
        world: World,
        start_time_step=None,
        use_boolean: bool = False,
        output_type: OutputType = OutputType.STANDARD,
    ):
        visitor = MonitorCreationRuleTreeVisitor(world.dt, output_type)
        self._rule = rule
        self._monitor = rule.visit(visitor)
        self._collector_visitor = PredicateCollectorMonitorTreeVisitor()
        self._visualizer_visitor = PredicateVisualizerMonitorTreeVisitor()
        self._eval_visitor = EvaluationMonitorTreeVisitor(
            use_boolean=use_boolean, output_type=output_type
        )
        self._last_evaluation_time_step = -1
        self._ego_vehicle = None
        self._world = None
        if ego_vehicle is not None:
            assert world is not None
            self.reset(ego_vehicle, world, start_time_step)

    @property
    def current_time(self) -> int:
        return self._last_evaluation_time_step

    def get_predicates(self) -> Dict[str, float]:
        predicate_values = dict(self._monitor.visit(self._collector_visitor))
        return predicate_values

    def visualize_predicates(
            self, visualization_config=Dict[str,any], plot_predicate_values=True, plot_scale=1.
    ) -> None:
        """
        Renders a scenario visualization using the MPRenderer and adds plots of the predicates.

        :renderer: currently, only MPRenderer is supported. For supporting any IRenderer, the methods inheriting from
        BasePredicateEvaluator:visualize have to be adapted.

        :visualization_config: predicate-name | 'default' -> {
            show_non_effective_predicate_instances_for_vehicles: List[Tuple[int]], # show predicate value for certain
            # vehicle-ids
        }. Allows predicate-type wise configuration of the visualization. Here, an effective predicate instance is one
        that belongs to an effective group within all enclosing all- and exist-quantifiers; "effective" group denotes
        the group giving the minimum resp. maximum value for all- resp.
        """
        default_fig_size = (25, 3)
        figsize = tuple(d*plot_scale for d in default_fig_size)
        if plot_predicate_values:
            # make scenario-plot and bar-chart side-by-side
            fig, axes = plt.subplots(figsize=figsize, nrows=1, ncols=2, gridspec_kw={'width_ratios': [5, 1]})
            ax = axes[0]
        else:
            plt.figure(figsize=figsize)
            ax = plt.gca()


        renderer = MPRenderer(ax=ax)

        commonroad_scenario = self._world.scenario

        general_draw_params = {"time_begin": self.current_time, 'dynamic_obstacle': {'show_label': True}}

        commonroad_scenario.lanelet_network.draw(renderer, draw_params=general_draw_params)

        vehicle2draw_params = {} # FIXME replace by adder function which does merging automatically...
        predicate_names2vehicle_ids2values = defaultdict(dict)

        draw_functions = self._monitor.visit(
            self._visualizer_visitor, vehicle2draw_params, predicate_names2vehicle_ids2values, self._world, self.current_time, visualization_config
        )

        for i in self._world.vehicle_ids_for_time_step(self.current_time):
            draw_params = vehicle2draw_params.get(i, {})
            commonroad_scenario.obstacle_by_id(i).draw(
                    renderer, draw_params=merge_dicts_recursively(general_draw_params, draw_params)
            )

        commonroad_scenario.obstacle_by_id(self._ego_vehicle.id).draw(renderer, draw_params=merge_dicts_recursively(general_draw_params, EGO_VEHICLE_DRAW_PARAMS))

        # Hint: plotting further stuff on the scenario only works after renderer.render() was called; therefore, the
        #   predicates need to return functions instead of directly plotting
        renderer.render()
        for fun in draw_functions:
            fun(renderer)

        if plot_predicate_values:
            df = pd.DataFrame.from_dict({predicate_name: {str(vehicle_ids): values for vehicle_ids, values in vehicle_ids2values.items()} for predicate_name, vehicle_ids2values in predicate_names2vehicle_ids2values.items()})
            cmap = plt.get_cmap('turbo') # different color map, as default one produces non-distinguishable colors for different bars
            numbers_for_bars = np.linspace(0, 1, num=len(df.columns), endpoint=False)
            ax = df.plot.barh(rot=0, ax=axes[1], width=1., edgecolor='black', linewidth=0.5, color=cmap(numbers_for_bars))
            ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5)) # place legend to the right


    def update(self):
        """
        Advance the monitor state by one time step and return the corresponding rule evaluation value.

        :return: robustness or boolean rule value
        """
        self._last_evaluation_time_step += 1
        if (
            self._ego_vehicle.start_time > self._last_evaluation_time_step
            or self._last_evaluation_time_step > self._ego_vehicle.end_time
        ):
            logger.warning("Evaluating vehicle outside its lifetime!")
            return np.inf
        rule_value = self._eval_visitor.walk(
            self._monitor,
            self._world,
            self._last_evaluation_time_step,
            self._ego_vehicle,
        )
        return rule_value

    def evaluate(self) -> np.ndarray:
        """
        Evaluate the rule exhaustively until the final time step of the vehicle object is reached.

        Caution: This will change the time step of the world object!

        :return: Array of all rule values for all time steps of the vehicle's known trajectory
        """
        robustness_values = []
        for i in range(
            self._last_evaluation_time_step + 1, self._ego_vehicle.end_time + 1
        ):
            robustness_values.append(self.update())
        return np.array(robustness_values)

    @property
    def other_ids(self) -> Tuple[int]:
        return self._eval_visitor.other_ids[1:]

    def reset(self, ego_vehicle: Vehicle, world: World, start_time_step=None):
        self._last_evaluation_time_step = (
            start_time_step - 1
            if start_time_step is not None
            else ego_vehicle.start_time - 1
        )
        self._ego_vehicle = ego_vehicle
        self._world = world
        # Reset monitor
        reset_visitor = ResetMonitorTreeVisitor()
        self._monitor.visit(reset_visitor)
