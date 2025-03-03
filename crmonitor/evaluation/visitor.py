import itertools
from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from crmonitor.common.helper import gather
from crmonitor.monitor.monitor_node import (
    AllMonitorNode,
    AndsmoothMonitorNode,
    CompareToThresholdScaledMonitorNode,
    ExistMonitorNode,
    HistoricallyDurationMonitorNode,
    HistoricallyDurationSeverityMonitorNode,
    MonitorNode,
    RuleMonitorNode,
    SumIfPositiveMonitorNode,
)
from crmonitor.monitor.rtamt_monitor_stl import OutputType, RtamtStlMonitor
from crmonitor.rule.rule_node import (
    AllNode,
    AndsmoothNode,
    CompareToThresholdScaledNode,
    ExistNode,
    HistoricallyDurationNode,
    HistoricallyDurationSeverityNode,
    IOType,
    PredicateNode,
    RuleNode,
    SumIfPositiveNode,
)


class RuleTreeVisitor(ABC):
    @abstractmethod
    def visit_rule_node(self, rule_node: Union[RuleNode, RuleMonitorNode], *ctx):
        pass

    @abstractmethod
    def visit_all_node(self, all_node: Union[AllNode, AllMonitorNode], *ctx):
        pass

    @abstractmethod
    def visit_exist_node(self, exist_node: Union[ExistNode, ExistMonitorNode], *ctx):
        pass

    @abstractmethod
    def visit_andsmooth_node(
        self, andsmooth_node: Union[AndsmoothNode, AndsmoothMonitorNode], *ctx
    ):
        pass

    @abstractmethod
    def visit_historicallyduration_node(
        self,
        historicallyduration_node: Union[
            HistoricallyDurationNode, HistoricallyDurationMonitorNode
        ],
        *ctx,
    ):
        pass

    @abstractmethod
    def visit_historicallydurationseverity_node(
        self,
        historicallydurationseverity_node: Union[
            HistoricallyDurationSeverityNode, HistoricallyDurationSeverityMonitorNode
        ],
        *ctx,
    ):
        pass

    @abstractmethod
    def visit_sum_if_positive_node(
        self,
        sum_if_positive_node: Union[SumIfPositiveNode, SumIfPositiveMonitorNode],
        *ctx,
    ):
        ...

    @abstractmethod
    def visit_compare_to_threshold_scaled_node(
        self,
        compare_to_threshold_scaled_node: Union[
            CompareToThresholdScaledNode, CompareToThresholdScaledMonitorNode
        ],
        *ctx,
    ):
        ...

    @abstractmethod
    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        pass


class MonitorCreationRuleTreeVisitor(RuleTreeVisitor):
    def __init__(self, dt, output_type=OutputType.STANDARD):
        self.dt = dt
        self.output_type = output_type

    def visit_rule_node(self, rule_node: RuleNode, *ctx):
        children = [c.visit(self, *ctx) for c in rule_node.children]
        monitor = RtamtStlMonitor.create_from_rule_node(
            rule_node, self.dt, self.output_type
        )
        return RuleMonitorNode(rule_node.name, children, monitor)

    def visit_all_node(self, all_node: AllNode, *ctx):
        children = [c.visit(self, *ctx) for c in all_node.children]
        return AllMonitorNode(all_node.name, children)

    def visit_exist_node(self, exist_node: ExistNode, *ctx):
        children = [c.visit(self, *ctx) for c in exist_node.children]
        return ExistMonitorNode(exist_node.name, children)

    def visit_andsmooth_node(self, andsmooth_node: AndsmoothNode, *ctx):
        children = [c.visit(self, *ctx) for c in andsmooth_node.children]
        return AndsmoothMonitorNode(andsmooth_node.name, children)

    def visit_historicallyduration_node(
        self, historicallyduration_node: HistoricallyDurationNode, *ctx
    ):
        children = [c.visit(self, *ctx) for c in historicallyduration_node.children]
        return HistoricallyDurationMonitorNode(
            historicallyduration_node.name, children, historicallyduration_node.interval
        )

    def visit_historicallydurationseverity_node(
        self,
        historicallydurationseverity_node: HistoricallyDurationSeverityNode,
        *ctx,
    ):
        children = [
            c.visit(self, *ctx) for c in historicallydurationseverity_node.children
        ]
        return HistoricallyDurationSeverityMonitorNode(
            historicallydurationseverity_node.name,
            children,
            historicallydurationseverity_node.interval,
        )

    def visit_sum_if_positive_node(
        self,
        sum_if_positive_node: SumIfPositiveNode,
        *ctx,
    ):
        children = [c.visit(self, *ctx) for c in sum_if_positive_node.children]
        return SumIfPositiveMonitorNode(
            sum_if_positive_node.name,
            children,
        )

    def visit_compare_to_threshold_scaled_node(
        self, compare_to_threshold_scaled_node: CompareToThresholdScaledNode, *ctx
    ):
        children = [
            c.visit(self, *ctx) for c in compare_to_threshold_scaled_node.children
        ]
        return CompareToThresholdScaledMonitorNode(
            compare_to_threshold_scaled_node.name,
            children,
            compare_to_threshold_scaled_node.threshold,
        )

    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        return predicate_node


class EvaluationMonitorTreeVisitor(RuleTreeVisitor):
    def __init__(self, use_boolean=False, output_type=OutputType.STANDARD):
        self.other_ids = tuple()
        self.use_boolean = use_boolean
        self.output_type = output_type
        self.all_values_all_ids = {}
        self.all_props_all_ids = {}

    def walk(self, node: MonitorNode, world, mpr_world, time_step, ego_vehicle, *ctx):
        self.other_ids = tuple()
        return node.visit(self, world, mpr_world, time_step, (ego_vehicle.id,), *ctx)

    def visit_rule_node(self, rule_node: RuleMonitorNode, *ctx):
        world = ctx[0]
        time_step = ctx[2]
        # Collect child_values
        assert rule_node.monitor.dt == world.dt, (
            f"Monitor constructed with dt="
            f"{rule_node.monitor.dt} but got "
            f"world state with dt={world.dt}!"
        )
        child_values = {c.name: c.visit(self, *ctx) for c in rule_node.children}
        val = rule_node.update(time_step, list(child_values.items()))
        return val

    def _visit_quant_node(self, node, *ctx):
        world, mpr_world, time_step, other_ids = ctx[:4]
        all_ids = world.vehicle_ids_for_time_step(time_step)
        remaining_ids = tuple(set(all_ids).difference(other_ids))
        values = []
        selected_ids = []
        for i in remaining_ids:
            ids = other_ids + (i,)
            val = node.monitors[i].visit(
                self, world, mpr_world, time_step, ids, *ctx[2:]
            )
            values.append(val)
            selected_ids.append(ids)
        return values, selected_ids

    def visit_all_node(self, all_node: AllMonitorNode, *ctx):
        values, selected_ids = self._visit_quant_node(all_node, *ctx)
        # TODO: fix since adding mpr world in ctx
        other_ids = ctx[3]
        if not isinstance(other_ids, tuple):
            other_ids = ctx[2]

        self.all_values_all_ids = {}  # reset to empty
        self.all_props_all_ids = {}  # reset to empty

        if len(values) > 0:
            idx = np.argmin(values)
            val = values[idx]
            self.other_ids = selected_ids[idx]
            all_node.last_selected = all_node.monitors[self.other_ids[-1]]

            # Loop through all selected_ids and populate the dictionary
            for i, sid in enumerate(selected_ids):
                self.all_values_all_ids[sid[-1]] = values[i]
                if hasattr(all_node.monitors[sid[-1]].monitor, "_propositions"):
                    self.all_props_all_ids[sid[-1]] = all_node.monitors[
                        sid[-1]
                    ].monitor._propositions
        else:
            val = 1.0
            self.other_ids = other_ids
            all_node.last_selected = None

        return val

    def visit_exist_node(self, exist_node: ExistMonitorNode, *ctx):
        values, selected_ids = self._visit_quant_node(exist_node, *ctx)
        other_ids = ctx[3]
        if not isinstance(other_ids, tuple):
            other_ids = ctx[2]

        self.all_values_all_ids = {}  # reset to empty
        self.all_props_all_ids = {}  # reset to empty

        if len(values) > 0:
            idx = np.argmax(values)
            val = values[idx]
            self.other_ids = selected_ids[idx]

            exist_node.last_selected = exist_node.monitors[self.other_ids[-1]]

            # Loop through all selected_ids and populate the dictionary
            for i, sid in enumerate(selected_ids):
                self.all_values_all_ids[sid[-1]] = values[i]
        else:
            val = -1.0
            self.other_ids = other_ids
            exist_node.last_selected = None

            # If no values, only add other_ids if it's not empty
            if other_ids:
                self.all_values_all_ids[other_ids[-1]] = val
        return val

    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        world, mpr_world, time_step, other_ids = ctx[:4]
        predicate_ids = gather(other_ids, predicate_node.agent_placeholders)
        if (
            self.use_boolean
            or predicate_node.io_type == IOType.INPUT
            and self.output_type == OutputType.OUTPUT_ROBUSTNESS
        ):
            value = predicate_node.evaluate_boolean(world, time_step, predicate_ids)
            value = 1.0 if value else -1.0
        else:
            value = predicate_node.evaluate_robustness(
                world, mpr_world, time_step, predicate_ids
            )
        return value

    def visit_andsmooth_node(
        self, andsmooth_node: Union[AndsmoothNode, AndsmoothMonitorNode], *ctx
    ):
        raise RuntimeError(
            "The 'andsmooth' operator is currently only supported when using offline evaluation."
        )

    def visit_historicallyduration_node(
        self,
        historicallyduration_node: Union[
            HistoricallyDurationNode, HistoricallyDurationMonitorNode
        ],
        *ctx,
    ):
        raise RuntimeError(
            "The 'historicallyDuration' operator is currently only supported when using offline evaluation."
        )

    def visit_historicallydurationseverity_node(
        self,
        historicallydurationseverity_node: Union[
            HistoricallyDurationSeverityNode, HistoricallyDurationMonitorNode
        ],
        *ctx,
    ):
        raise RuntimeError(
            "The 'historicallyDurationSeverity' operator is currently only supported when using offline evaluation."
        )


class BaseValueMonitorTreeVisitor(RuleTreeVisitor, ABC):
    def visit_all_node(self, all_node: AllMonitorNode, *ctx):
        if all_node.last_selected is None:
            # Visit the prototype monitor
            val = all_node.children[0].visit(self, *ctx)
            val = [(n, v if v is not None else 1.0) for n, v in val]
        else:
            val = all_node.last_selected.visit(self, *ctx)
        return val

    def visit_exist_node(self, exist_node: ExistMonitorNode, *ctx):
        if exist_node.last_selected is None:
            # Visit the prototype monitor
            val = exist_node.children[0].visit(self, *ctx)
            val = [(n, v if v is not None else -1.0) for n, v in val]
        else:
            val = exist_node.last_selected.visit(self, *ctx)
        return val

    def visit_andsmooth_node(self, andsmooth_node: AndsmoothMonitorNode, *ctx):
        return [c.visit(self, *ctx) for c in andsmooth_node.children]

    def visit_historicallyduration_node(
        self, historicallyduration_node: HistoricallyDurationMonitorNode, *ctx
    ):
        return [c.visit(self, *ctx) for c in historicallyduration_node.children]

    def visit_historicallydurationseverity_node(
        self, historicallydurationseverity_node: HistoricallyDurationSeverityNode, *ctx
    ):
        return [c.visit(self, *ctx) for c in historicallydurationseverity_node.children]

    def visit_compare_to_threshold_scaled_node(
        self,
        compare_to_threshold_scaled_node: Union[
            CompareToThresholdScaledNode, CompareToThresholdScaledMonitorNode
        ],
        *ctx,
    ):
        return [c.visit(self, *ctx) for c in compare_to_threshold_scaled_node.children]

    def visit_sum_if_positive_node(
        self,
        sum_if_positive_node: Union[SumIfPositiveNode, SumIfPositiveMonitorNode],
        *ctx,
    ):
        if sum_if_positive_node.last_selected is None:
            # Visit the prototype monitor
            val = sum_if_positive_node.children[0].visit(self, *ctx)
            val = [(n, v if v is not None else -1.0) for n, v in val]
        else:
            val = sum_if_positive_node.last_selected.visit(self, *ctx)
        return val


class PredicateCollectorMonitorTreeVisitor(BaseValueMonitorTreeVisitor):
    def visit_rule_node(self, rule_node: "RuleMonitorNode", *ctx):
        r = []
        for c in rule_node.children:
            r.extend(c.visit(self))
        return r

    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        return [(predicate_node.name, predicate_node.latest_value)]


class MPRGradientCollectorMonitorTreeVisitor(PredicateCollectorMonitorTreeVisitor):
    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        return [(predicate_node.name, predicate_node.mpr_gradient)]


class AstNodeValueCollectorMonitorTreeVisitor(BaseValueMonitorTreeVisitor):
    @staticmethod
    def visit_rule_node(rule_node: "RuleMonitorNode", *ctx):
        return list(rule_node.monitor.ast_node_values.items())

    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        raise NotImplementedError()


class PredicateVisualizerMonitorTreeVisitor(RuleTreeVisitor):
    """
    Returns list of dictionaries, each dictionary mapping vehicle ids to a possibly
    nested dict of draw-parameters
    """

    def _split_context(self, ctx):
        idx = 6
        is_effective = ctx[idx] if len(ctx) > idx else True
        return ctx[:idx], is_effective

    def visit_rule_node(self, rule_node: RuleMonitorNode, *ctx):
        draw_functions_nested = [c.visit(self, *ctx) for c in rule_node.children]
        return list(itertools.chain(*draw_functions_nested))

    def _visit_quant_node(self, node, *ctx):
        ctx, is_effective_so_far = self._split_context(ctx)
        draw_functions_for_effective_node = []
        if node.last_selected is not None:
            draw_functions_for_effective_node = node.last_selected.visit(
                self, *ctx, True and is_effective_so_far
            )
        draw_functions_nested = [
            monitor.visit(self, *ctx, False)
            for i, monitor in node.monitors.items()
            if monitor != node.last_selected
        ]
        return (
            list(itertools.chain(*draw_functions_nested))
            + draw_functions_for_effective_node
        )

    def visit_all_node(self, all_node: AllMonitorNode, *ctx):
        return self._visit_quant_node(all_node, *ctx)

    def visit_exist_node(self, exist_node: ExistMonitorNode, *ctx):
        return self._visit_quant_node(exist_node, *ctx)

    def visit_andsmooth_node(self, andsmooth_node: AndsmoothMonitorNode, *ctx):
        print("TODO implement")
        return self._visit_quant_node(andsmooth_node, *ctx)

    def visit_historicallyduration_node(
        self, historicallyduration_node: HistoricallyDurationMonitorNode, *ctx
    ):
        return self._visit_quant_node(historicallyduration_node, *ctx)

    def visit_historicallydurationseverity_node(
        self,
        historicallydurationseverity_node: HistoricallyDurationSeverityMonitorNode,
        *ctx,
    ):
        return self._visit_quant_node(historicallydurationseverity_node, *ctx)

    def visit_sum_if_positive_node(
        self,
        sum_if_positive_node: Union[SumIfPositiveNode, SumIfPositiveMonitorNode],
        *ctx,
    ):
        return self._visit_quant_node(sum_if_positive_node, *ctx)

    def visit_compare_to_threshold_scaled_node(
        self,
        compare_to_threshold_scaled_node: Union[
            CompareToThresholdScaledNode, CompareToThresholdScaledMonitorNode
        ],
        *ctx,
    ):
        return self._visit_quant_node(compare_to_threshold_scaled_node, *ctx)

    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        ctx, is_effective = self._split_context(ctx)

        (
            add_vehicle_draw_params,
            predicate_names2vehicle_ids2values,
            predicate_name2predicate_evaluator,
            world,
            time_step,
            visualization_config,
        ) = ctx

        pred_name = predicate_node.evaluator.predicate_name
        latest_vehicle_ids = predicate_node.latest_vehicle_ids

        config_entry_key = pred_name if pred_name in visualization_config else "default"
        config_obj = visualization_config.get(config_entry_key, {})
        show_non_effective_predicate_instances_for_vehicles = config_obj.get(
            "show_non_effective_predicate_instances_for_vehicles", []
        )

        if (
            not is_effective
            and latest_vehicle_ids
            not in show_non_effective_predicate_instances_for_vehicles
        ):
            return ()

        predicate_name2predicate_evaluator[pred_name] = predicate_node.evaluator

        return predicate_node.evaluator.visualize(
            latest_vehicle_ids,
            add_vehicle_draw_params,
            world,
            time_step,
            predicate_names2vehicle_ids2values,
        )


class ResetMonitorTreeVisitor(RuleTreeVisitor):
    def _visit(self, node, *ctx):
        for c in node.monitors.values():
            c.visit(self, *ctx)

    def visit_rule_node(self, rule_node: Union[RuleNode, RuleMonitorNode], *ctx):
        rule_node.reset()

    def visit_all_node(self, all_node: Union[AllNode, AllMonitorNode], *ctx):
        self._visit(all_node, *ctx)

    def visit_exist_node(self, exist_node: Union[ExistNode, ExistMonitorNode], *ctx):
        self._visit(exist_node, *ctx)

    def visit_andsmooth_node(
        self, andsmooth_node: Union[AndsmoothNode, AndsmoothMonitorNode], *ctx
    ):
        self._visit(andsmooth_node, *ctx)

    def visit_historicallyduration_node(
        self,
        historicallyduration_node: Union[
            HistoricallyDurationNode, HistoricallyDurationMonitorNode
        ],
        *ctx,
    ):
        self._visit(historicallyduration_node, *ctx)

    def visit_historicallydurationseverity_node(
        self,
        historicallydurationseverity_node: Union[
            HistoricallyDurationSeverityNode, HistoricallyDurationSeverityMonitorNode
        ],
        *ctx,
    ):
        self._visit(historicallydurationseverity_node, *ctx)

    def visit_compare_to_threshold_scaled_node(
        self,
        compare_to_threshold_scaled_node: Union[
            CompareToThresholdScaledNode, CompareToThresholdScaledMonitorNode
        ],
        *ctx,
    ):
        self._visit(compare_to_threshold_scaled_node)

    def visit_sum_if_positive_node(
        self,
        sum_if_positive_node: Union[SumIfPositiveNode, SumIfPositiveMonitorNode],
        *ctx,
    ):
        self._visit(sum_if_positive_node)

    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        pass
