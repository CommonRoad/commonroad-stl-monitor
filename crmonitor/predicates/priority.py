from enum import Enum
import logging
from typing import List, Set
from crmonitor.common.world import World
from crmonitor.predicates.base import BasePredicateEvaluator
from commonroad.scenario.traffic_sign import TrafficLightState
from commonroad.scenario import lanelet
from commonroad.scenario.traffic_sign import TrafficLight

logger = logging.getLogger(__name__)


class PriorityPredicates(str, Enum):
    SamePriority = "same_priority"
    RelevantTrafficLight = "relevant_traffic_light"
    HasPriority = "has_priority"


class PredSamePriority(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """
    predicate_name = PriorityPredicates.SamePriority
    arity = 4

    # TODO
    # def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:

    # TODO
    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        return self._scale_lat_dist(100)


class PredRelevantTrafficLight(BasePredicateEvaluator):
    """
    evaluates if an upcoming intersection is regulated by traffic lights
    """
    predicate_name = PriorityPredicates.RelevantTrafficLight
    arity = 1

    # TODO
    # def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:

    # TODO
    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        """
        idea 1 (trivial):  
        for l in lanelets_dir(x_k):
            for succ in reachsuc(l):
                if len(active_tls_by_lanelet(succ)) > 0:
                    return 1
        return -1

        def active_tls_by_lanelet(l: lanelet): 
            return list(set(lambda tl : statetl(tl) != inactive, traffic_light(l)))
        """
        """
        idea 2:  
        lanelets_to_tl = calculates_lanelets_away_from_tl(...) #returns -1 if no relevant tl is found
        return 100/lanelets_to_tl if lanelets_to_tl >= 0 else -1
        # robustness is larger if tl is nearer, could have chosen any number instead of 100 but whatever
        """
        return self._scale_lat_dist(100)

    def active_tls_by_lanelet(l: lanelet, time_step):
        """
        returns a set of active traffic lights for a given lanelet
        """
        # TODO
        # how to access traffic_lights of a lanelet
        return set(filter(lambda tl: tl.get_state_at_time_step(time_step) != TrafficLightState.INACTIVE, l.traffic_lights))


class PredHasPriority(BasePredicateEvaluator):
    """
    evaluates if the first vehicle has priority over the second vehicle.
    """
    predicate_name = PriorityPredicates.HasPriority
    arity = 4

    # TODO
    # def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:

    # TODO
    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        return self._scale_lat_dist(100)
