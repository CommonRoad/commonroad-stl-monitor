from typing import Dict, List, Callable, Tuple, Optional, Union

import numpy as np
import pandas as pd
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.renderer import IRenderer
from matplotlib import pyplot as plt

from crmonitor.common.helper import merge_dicts_recursively
from crmonitor.common.world import World
from crmonitor.predicates.predicate import BasePredicateEvaluator

EGO_VEHICLE_DRAW_PARAMS = {
    "dynamic_obstacle": {
        "vehicle_shape": {
            "occupancy": {"shape": {"rectangle": {"facecolor": "yellow"}}}
        }
    }
}


def _plot_rule_robustness_course(
    ax,
    rule_robustness_course: List[Tuple[int, float]],
    plot_limits: Tuple[float, float],
):
    np_rule_robustness_course = np.array(rule_robustness_course)
    rob_values = np_rule_robustness_course[:, 1]
    ax.plot(rob_values, "b-")
    ax.plot(np.where(rob_values < 0.0, rob_values, np.nan), "rx")
    ax.plot(np.where(rob_values >= 0.0, rob_values, np.nan), "g.")
    ax.set_xlim([np_rule_robustness_course[0, 0], np_rule_robustness_course[-1, 0]])
    ax.set_ylim(plot_limits)
    ax.grid(True)
    ax.set_xlabel("rule robustness")


def _plot_predicate_bar_chart(
    predicate_names2vehicle_ids2values: Dict[str, Dict[Tuple[int, ...], float]],
    ax,
    bar_chart_plot_limits: Tuple[float, float],
):
    df = pd.DataFrame.from_dict(
        {
            predicate_name: {
                str(vehicle_ids): values
                for vehicle_ids, values in vehicle_ids2values.items()
            }
            for predicate_name, vehicle_ids2values in predicate_names2vehicle_ids2values.items()
        }
    )

    # we use a different color map, as default one produces non-distinguishable colors for different bars
    cmap = plt.get_cmap("turbo")
    numbers_for_bars = np.linspace(0, 1, num=len(df.columns), endpoint=False)
    ax = df.plot.barh(
        rot=0,
        ax=ax,
        width=1.0,
        edgecolor="black",
        linewidth=0.5,
        color=cmap(numbers_for_bars),
        xlim=bar_chart_plot_limits,
    )
    ax.set_ylabel("vehicle ids")
    ax.set_xlabel("predicate robustness")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))  # place legend to the right


SCENARIO_FIG_SIZE = (20, 2)


def _create_axes(
    plot_scale: float, plot_predicate_bar_chart: bool, plot_rule_robustness_course: bool
):
    figsize = tuple(d * plot_scale for d in SCENARIO_FIG_SIZE)

    additional_plots = [plot_predicate_bar_chart, plot_rule_robustness_course]
    axes_of_additional_plots = [None] * len(additional_plots)
    n_additional_cols = sum(int(v) for v in additional_plots)

    if n_additional_cols > 0:
        width_ratio = 6
        width, height = figsize
        figsize = (width * (1 + n_additional_cols / width_ratio), height)
        # make scenario-plot and bar-chart side-by-side
        fig, axes = plt.subplots(
            figsize=figsize,
            nrows=1,
            ncols=1 + n_additional_cols,
            gridspec_kw={"width_ratios": [width_ratio, *([1] * n_additional_cols)]},
            layout="constrained",  # makes the layout consider overlaps of columns automatically
        )
        scenario_ax = axes[0]
        k = 0
        for i, p in enumerate(additional_plots):
            if p:
                axes_of_additional_plots[i] = axes[1 + k]
                k += 1
    else:
        plt.figure(figsize=figsize)
        scenario_ax = plt.gca()

    return (scenario_ax, *axes_of_additional_plots)


def _plot_scenario_legend(
    predicate_name2predicate_evaluator: Dict[str, BasePredicateEvaluator]
):
    num_predicates = len(predicate_name2predicate_evaluator)
    fig, (axes_row_1, axes_row_2) = plt.subplots(
        figsize=SCENARIO_FIG_SIZE, nrows=2, ncols=num_predicates
    )
    fig.suptitle("Legend: predicate visualization in scenario", fontsize=14)
    for ax1, ax2, (pred_name, pred_evaluator) in zip(
        axes_row_1, axes_row_2, predicate_name2predicate_evaluator.items()
    ):
        ax1.text(0.1, 0.5, pred_name, fontsize=12)
        ax1.axis("off")
        pred_evaluator.plot_predicate_visualization_legend(ax2)


def plot_predicate_visualization(
    world: World,
    ego_vehicle_id: int,
    time_step: int,
    vehicle2draw_params: Dict[int, any],
    draw_functions: List[Callable[[IRenderer], None]],
    predicate_names2vehicle_ids2values: Dict[str, Dict[Tuple[int, ...], float]],
    plot_scale: float,
    plot_scenario_legend: Optional[bool],
    predicate_name2predicate_evaluator: Dict[str, BasePredicateEvaluator],
    plot_predicate_bar_chart: bool,
    bar_chart_plot_limits: Tuple[float, float],
    plot_rule_robustness_course: bool,
    rule_robustness_course: List[Tuple[int, float]],
    rule_robustness_course_plot_limits: Tuple[float, float],
    scenario_plot_limits: Union[List[Union[int, float]], None]
):
    if plot_scenario_legend or plot_scenario_legend is None and time_step == 0:
        _plot_scenario_legend(predicate_name2predicate_evaluator)

    scenario_ax, bar_chart_ax, robustness_course_ax = _create_axes(
        plot_scale, plot_predicate_bar_chart, plot_rule_robustness_course
    )

    renderer = MPRenderer(ax=scenario_ax, plot_limits=scenario_plot_limits)
    commonroad_scenario = world.scenario

    general_draw_params = {
        "time_begin": time_step,
        "dynamic_obstacle": {
            "show_label": True,
            "vehicle_shape": {
                "occupancy": {"shape": {"rectangle": {"facecolor": "#90ee90"}}}
            },
        },
    }

    commonroad_scenario.lanelet_network.draw(renderer, draw_params=general_draw_params)

    for i in world.vehicle_ids_for_time_step(time_step):
        draw_params = vehicle2draw_params.get(i, {})
        commonroad_scenario.obstacle_by_id(i).draw(
            renderer,
            draw_params=merge_dicts_recursively(general_draw_params, draw_params),
        )

    commonroad_scenario.obstacle_by_id(ego_vehicle_id).draw(
        renderer,
        draw_params=merge_dicts_recursively(
            general_draw_params, EGO_VEHICLE_DRAW_PARAMS
        ),
    )

    # Hint: plotting further stuff on the scenario only works after renderer.render() was called; therefore, the
    #   predicates need to return functions instead of directly plotting
    renderer.render()
    for fun in draw_functions:
        fun(renderer)

    if plot_predicate_bar_chart:
        _plot_predicate_bar_chart(
            predicate_names2vehicle_ids2values, bar_chart_ax, bar_chart_plot_limits
        )

    if plot_rule_robustness_course:
        _plot_rule_robustness_course(
            robustness_course_ax,
            rule_robustness_course,
            rule_robustness_course_plot_limits,
        )
