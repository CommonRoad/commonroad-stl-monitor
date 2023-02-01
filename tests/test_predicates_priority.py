import math
import unittest
from pathlib import Path
import numpy as np

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.scenario.obstacle import State, ObstacleType

from crmonitor.common.helper import load_yaml
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import Vehicle, CurvilinearStateManager
from crmonitor.common.world import World
from crmonitor.predicates.priority import (
    PredRelevantTrafficLight,
    PredSamePriority,
    PredHasPriority,
    PredRelevantTrafficLight,
)


class TestPriorityPredicates(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    # TODO
    def test_same_priority(self):
        self.assertEqual(1, 1)

    # TODO
    def test_relevant_traffic_light(self):
        self.assertEqual(1, 1)

    # TODO
    def test_has_priority(self):
        self.assertEqual(1, 1)
