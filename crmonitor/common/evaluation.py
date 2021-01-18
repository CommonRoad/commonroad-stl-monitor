import itertools
import math
from typing import List, Tuple, Dict

from crmonitor.common.helper import gather
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world_state import WorldState
from crmonitor.monitor.rtamt_monitor_stl import TrafficRuleMonitorForwardSTL
from crmonitor.predicates.python.predicate_value import PredicateValue, \
    PredicateValueCollection
from crmonitor.predicates.python.rule import Rule


def get_valid_time_interval(vehicle_a: Vehicle, vehicle_b: Vehicle):
    start = max(vehicle_a.start_time, vehicle_b.start_time)
    end = min(vehicle_a.end_time, vehicle_b.end_time)
    return start, end


def evaluate_rule(world_state: WorldState, o_id, rule) -> Tuple[
    List[Tuple[float, float]], Dict[float, List[Tuple[str, float]]]]:
    """
    Evaluates a rule for the ego agent of the world_state and the given other agent stepwise.

    :param world_state: Initial world state
    :param o_id: ID of the other agent
    :param rule: Rule to evaluate
    :return: First element: List of tuples of time step and robustness value
            Second element: Dictionary where the key is the time_step and the value is a list of tuples,
            containing the predicate string and the corresponding robustness value.
    """
    all_pred_values = PredicateValueCollection()
    # Collect predicates
    assignment = (world_state.ego_vehicle.id, o_id)
    pred_values = PredicateValueCollection()
    start, end = get_valid_time_interval(world_state.ego_vehicle,
                                         world_state.vehicle_by_id(o_id))
    world_state.time_step = start
    while world_state.time_step <= end:
        for pred_assign in rule.predicate_assignment:
            predicate_ids = gather(assignment, pred_assign.agent_placeholders)
            value = PredicateValue(pred_assign.base_name, predicate_ids,
                                   world_state.time_step)
            if value not in pred_values:
                value.value = pred_assign.evaluator.evaluate_robustness(
                        world_state, predicate_ids)
                all_pred_values.append(value)
            pred_values.append(all_pred_values[value])
        world_state.step()

    # Evaluate rule
    monitor = TrafficRuleMonitorForwardSTL(rule,
                                           output_type="standard")
    monitor_values = {}
    world_state.time_step = start
    while world_state.time_step <= end:
        l = []
        for pred_assign in rule.predicate_assignment:
            ids = gather(assignment, pred_assign.agent_placeholders)
            v = pred_values.by_time_step(world_state.time_step).by_name(
                    pred_assign.base_name).by_ids(ids).get_single_value().value
            l.append((pred_assign.full_name, v))

        monitor_values[world_state.time_step] = l
        world_state.step()
    world_state.time_step = 0
    rob_values = monitor.evaluate_monitor_offline_stepwise(monitor_values)
    return rob_values, monitor_values


def evaluate_necessary_predicates_all_agents(rules: List[Rule],
                                             world_state: WorldState):
    max_vehicle_dependency = max([x.num_dependent_vehicles for x in rules])
    vehicles = world_state.other_vehicles
    vehicle_ids = [v.id for v in vehicles]
    predicate_values = PredicateValueCollection()
    predicate_assignments = set()
    for rule in rules:
        predicate_assignments.update(rule.predicate_assignment)

    for comb in itertools.combinations(vehicle_ids, max_vehicle_dependency):
        ids = (world_state.ego_vehicle.id,) + comb

        # Check if trajectories overlap
        max_start_time = -math.inf
        min_end_time = math.inf
        for id in ids:
            max_start_time = max(
                world_state.vehicle_by_id(id).state_list_cr[0].time_step,
                max_start_time)
            min_end_time = min(
                world_state.vehicle_by_id(id).state_list_cr[-1].time_step,
                min_end_time)
        if max_start_time > min_end_time:
            continue

        for pred in predicate_assignments:
            predicate_vehicle_ids = gather(ids, pred.agent_placeholders)
            value = pred.evaluator.evaluate_robustness(world_state,
                                                       predicate_vehicle_ids)
            pred_value = PredicateValue(pred.full_name, predicate_vehicle_ids[
                                                        :pred.num_dependencies],
                                        world_state.time_step, value)
            predicate_values.append(pred_value)
    return predicate_values
