from typing import Dict, List, Callable, Tuple

import numpy as np
import pandas as pd
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.renderer import IRenderer
from matplotlib import pyplot as plt

from crmonitor.common.helper import merge_dicts_recursively
from crmonitor.common.world import World

EGO_VEHICLE_DRAW_PARAMS = {
    "dynamic_obstacle": {
        "vehicle_shape": {
            "occupancy": {"shape": {"rectangle": {"facecolor": "yellow"}}}
        }
    }
}


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
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))  # place legend to the right


def _create_axes(plot_predicate_bar_chart: bool, plot_scale: float):
    default_fig_size = (20, 3)
    figsize = tuple(d * plot_scale for d in default_fig_size)

    if plot_predicate_bar_chart:
        width_ratio = 5
        width, height = figsize
        figsize = (width * (1 + 1.0 / width_ratio), height)
        # make scenario-plot and bar-chart side-by-side
        fig, axes = plt.subplots(
            figsize=figsize,
            nrows=1,
            ncols=2,
            gridspec_kw={"width_ratios": [width_ratio, 1]},
        )
        scenario_ax = axes[0]
        bar_chart_ax = axes[1]
    else:
        plt.figure(figsize=figsize)
        scenario_ax = plt.gca()
        bar_chart_ax = None

    return scenario_ax, bar_chart_ax


def plot_predicate_visualization(
    world: World,
    ego_vehicle_id: int,
    time_step: int,
    vehicle2draw_params: Dict[int, any],
    draw_functions: List[Callable[[IRenderer], None]],
    predicate_names2vehicle_ids2values: Dict[str, Dict[Tuple[int, ...], float]],
    plot_scale: float,
    plot_predicate_bar_chart: bool,
    bar_chart_plot_limits: Tuple[float, float],
):
    scenario_ax, bar_chart_ax = _create_axes(plot_predicate_bar_chart, plot_scale)

    renderer = MPRenderer(ax=scenario_ax)
    commonroad_scenario = world.scenario

    general_draw_params = {
        "time_begin": time_step,
        "dynamic_obstacle": {"show_label": True},
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
