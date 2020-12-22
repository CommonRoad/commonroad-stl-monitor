import itertools
import math
from typing import List, Set

from crmonitor.common.helper import gather
from crmonitor.common.world_state import WorldState
from crmonitor.predicates.python.rule import Rule


def evaluate_necessary_predicates_all_agents(rules: List[Rule], world_state: WorldState):
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
            max_start_time = max(world_state.vehicle_by_id(id).state_list_cr[0].time_step, max_start_time)
            min_end_time = min(world_state.vehicle_by_id(id).state_list_cr[-1].time_step, min_end_time)
        if max_start_time > min_end_time:
            continue

        for pred in predicate_assignments:
            predicate_vehicle_ids = gather(ids, pred.agent_placeholders)
            value = pred.evaluator.evaluate_robustness(world_state,
                                                       predicate_vehicle_ids)
            pred_value = PredicateValue(pred.pred_str, ids,
                                        world_state.time_step, value)
            predicate_values.append(pred_value)
    return predicate_values


class PredicateValue:
    def __init__(self, predicate_str, vehicle_ids, time_step, value):
        self.predicate_str = predicate_str
        self.vehicle_ids = tuple(vehicle_ids)
        self.time_step = time_step
        self.value = value

    def __hash__(self):
        return hash((self.predicate_str, self.vehicle_ids, self.time_step))

    def __eq__(self, o) -> bool:
        return self.predicate_str == o.predicate_str and self.vehicle_ids == o.vehicle_ids and self.time_step == o.time_step


class PredicateValueCollection:
    def __init__(self, l: Set[PredicateValue] = set()):
        self._predicate_values = l

    @staticmethod
    def _return_collection_or_value(pred):
        if len(pred) == 0:
            raise KeyError()
        elif len(pred) == 1:
            return pred[0]
        else:
            return PredicateValueCollection(pred)

    def by_name(self, name):
        pred = list(filter(lambda x: x.predicate_str == name,
                           self._predicate_values))
        return PredicateValueCollection._return_collection_or_value(pred)

    def by_time_step(self, time_step):
        pred = list(filter(lambda x: x.time_step == time_step,
                           self._predicate_values))
        return PredicateValueCollection._return_collection_or_value(pred)

    def by_ids(self, ids):
        pred = list(
            filter(lambda x: x.vehicle_ids == ids, self._predicate_values))
        return PredicateValueCollection._return_collection_or_value(pred)

    def get_time_steps(self):
        tsteps = set()
        for p in self._predicate_values:
            tsteps.add(p.time_step)
        return tuple(sorted(tsteps))

    def append(self, element):
        self._predicate_values.add(element)

    def extend(self, other):
        self._predicate_values.update(other._predicate_values)
