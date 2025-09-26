import copy

from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import OfflineRuleEvaluator
from crmonitor.monitor import (
    BaseValueMonitorTreeVisitor,
    MonitorCreationRuleTreeVisitor,
    OutputType,
    RtamtRuleMonitorNode,
)
from crmonitor.monitor.proposition_robustness import PropositionRobustnessMonitor
from crmonitor.rule.rule_node import PredicateNode, RuleAstNode


class PropositionMonitorRuleTreeVisitor(MonitorCreationRuleTreeVisitor):
    def visit_rule_node(self, rule_node: RuleAstNode, *ctx):
        children = [c.visit(self, *ctx) for c in rule_node.children]
        monitor = PropositionRobustnessMonitor.create_from_rule_node(
            rule_node, self.dt, self.output_type
        )
        return RtamtRuleMonitorNode(rule_node.name, children, monitor)


class PropositionCollectorMonitorTreeVisitor(BaseValueMonitorTreeVisitor):
    @staticmethod
    def visit_rule_node(rule_node: "RtamtRuleMonitorNode", *ctx):
        return list(rule_node.monitor.ast_node_values.items())

    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        raise NotImplementedError()


class PropositionRuleEvaluator(OfflineRuleEvaluator):
    def __init__(
        self,
        rule: RuleAstNode,
        ego_id: int,
        world: World,
        start_time_step=None,
        use_boolean: bool = False,
        output_type: OutputType = OutputType.STANDARD,
    ):
        monitor_creation_visitor = PropositionMonitorRuleTreeVisitor(world.dt, output_type)
        self.proposition_collector = PropositionCollectorMonitorTreeVisitor()
        super().__init__(
            rule,
            ego_id,
            world,
            start_time_step,
            use_boolean,
            output_type,
            monitor_creation_visitor,
        )
        monitor_copied = copy.copy(self._monitor)
        while not isinstance(monitor_copied, RtamtRuleMonitorNode):
            if isinstance(monitor_copied, list):
                monitor_copied = copy.copy(monitor_copied[0])
            else:
                monitor_copied = monitor_copied.children
        self.rule_str_original = monitor_copied.monitor._rule

    def get_propositions(self):
        """
        Calculates the proposition robustness (mainly used for trajectory repairing)
        Calculations are done for the non-ego vehicle that conforms to the rule with the lowest feasibility.

        Returns:
        props (dict{prop, value}): Robustness values of each proposition, obtained using _props attribute of the
        RtamtStlMonitor, set using the RtamtStlMonitor.collect_prop_rob method. If quantifier nodes exist, the Monitor
        that monitors the ego vehicle against the worst-case non-ego vehicle is used.
        other_id (int): The vehicle against which the values were obtained. Ego if the rule concerns the ego vehicle.
        time (int): Timestep at which the values were obtained.
        """
        other_id = (
            self._eval_visitor.other_ids[-1]
            if len(self._eval_visitor.other_ids) > 0
            else self.ego_vehicle.id
        )
        if hasattr(self._monitor, "monitors"):
            other_id = self._eval_visitor.other_ids[-1]
            props = self._monitor.monitors[other_id].monitor._propositions
        else:
            if any(hasattr(child, "monitors") for child in self._monitor.children):
                other_id = self._eval_visitor.other_ids[-1]
                props = self._monitor.monitor._propositions
            else:
                props = self._monitor.monitor._propositions

        return props, other_id, self._last_evaluation_time_step

    def get_propositions_all(self):
        # New dictionary structure
        transformed_all_props_all_ids = {}

        if not self._eval_visitor.all_props_all_ids:
            # Iterate over all vehicle IDs stored in all_values_all_ids
            veh_id = self.ego_vehicle.id
            vehicle_props = self._monitor.monitor._propositions

            # Populate the props dictionary with proposition names as keys
            for prop_name, robustness_value in vehicle_props.items():
                if prop_name not in transformed_all_props_all_ids:
                    transformed_all_props_all_ids[prop_name] = {}
                transformed_all_props_all_ids[prop_name][veh_id] = robustness_value

        else:
            # Iterate over the original dictionary
            for time_step, props in self._eval_visitor.all_props_all_ids.items():
                for v_id, value in props.items():
                    # If the feature is not in the new dictionary, initialize it with an empty dictionary
                    if v_id not in transformed_all_props_all_ids:
                        transformed_all_props_all_ids[v_id] = {}
                    # Add the time_step and its corresponding value to the feature's dictionary
                    transformed_all_props_all_ids[v_id][time_step] = value

        # Determine the violation other_id used
        other_id = (
            self._eval_visitor.other_ids[-1]
            if self._eval_visitor.other_ids
            else self.ego_vehicle.id
        )

        # Collect the props for the other_id separately
        if other_id == self.ego_vehicle.id:
            other_id_props = self._monitor.monitor._propositions
        else:
            other_id_props = {
                prop_name: robustness_value.get(other_id, None)
                for prop_name, robustness_value in transformed_all_props_all_ids.items()
            }

        return (
            transformed_all_props_all_ids,
            other_id_props,
            self._eval_visitor.all_values_all_ids,
            self._last_evaluation_time_step,
        )
