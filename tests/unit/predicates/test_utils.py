from collections.abc import Iterable
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.scenario import LaneletNetwork
from commonroad.scenario.state import CustomState, State
import pytest
import numpy as np

from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world import World
from crmonitor.predicates.utils import distance_to_lanes
from tests.util import parallel_lanes


@pytest.mark.parametrize(
    "exp_result,state,lanelet_ids",
    [
        # Vehicle directly at the center of a lane
        (3.0, CustomState(time_step=0, position=np.array([10.0, 2.0]), orientation=0), [1]),
        # The right border of the vehicle aligns with the right border of the lane. The distance is the width of the vehicle.
        (2.0, CustomState(time_step=0, position=np.array([10.0, 1.0]), orientation=0), [1]),
        # Vehicle is centered on the lane border.
        (1.0, CustomState(time_step=0, position=np.array([10.0, 0.0]), orientation=0), [1]),
        # The right border of the vehicle is exactly on the left border of the lane.
        (0.0, CustomState(time_step=0, position=np.array([10.0, 5.0]), orientation=0), [1]),
        # The vehicle is directly at the center of the lane left of the queried lane.
        (-1.0, CustomState(time_step=0, position=np.array([10.0, 6.0]), orientation=0), [1]),
        # The left border of the vehicle is exactly on the right border of the lane.
        (0.0, CustomState(time_step=0, position=np.array([10.0, 3.0]), orientation=0), [2]),
        # The left border of the vehicle is exactly on the right border of the lane.
        (0.0, CustomState(time_step=0, position=np.array([10.0, 3.0]), orientation=0), [1, 2]),
    ],
)
def test_distance_to_lanes(exp_result: float, state: State, lanelet_ids: Iterable[int]) -> None:
    lanelets = parallel_lanes(num_lanes=4)
    lanelet_network = LaneletNetwork()
    for lanelet in lanelets:
        lanelet_network.add_lanelet(lanelet)

    road_network = RoadNetwork(lanelet_network)
    ego_vehicle = Vehicle(
        id=0,
        obstacle_type=ObstacleType.CAR,
        shape=Rectangle(5, 2),
        states_cr={state.time_step: state},
        lanelet_assignment=None,
        road_network=road_network,
        dt=0.1,
    )

    world = World({ego_vehicle}, road_network)

    result = distance_to_lanes(ego_vehicle, lanelet_ids, world, state.time_step)
    assert np.isclose(result, exp_result)
