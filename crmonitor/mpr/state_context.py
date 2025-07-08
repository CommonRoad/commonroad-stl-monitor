import functools
import inspect
from collections.abc import Iterable
from typing import Callable, ParamSpec, TypedDict, TypeVar

from commonroad.scenario.obstacle import Obstacle
from commonroad.scenario.scenario import Scenario
from typing_extensions import Unpack

from crmonitor.common import ScenarioType, Vehicle, World
from crmonitor.common.vehicle import CurvilinearVehicleState, StateLateral, StateLongitudinal


class ParseError(Exception):
    pass


class StateContextError(Exception): ...


class StateContextKwargs(TypedDict, total=False):
    """
    TypedDict for StateContext keyword arguments.
    Makes kwargs type-safe while allowing partial specification.
    """

    world: World | None
    scenario: Scenario | None
    time_step: int
    vehicles: Iterable[Vehicle] | None
    vehicle_ids: Iterable[int] | None
    scenario_type: ScenarioType
    states: list[CurvilinearVehicleState] | None


class StateContext:
    def __init__(
        self,
        *,
        time_step: int,
        world: World | None = None,
        scenario: Scenario | None = None,
        vehicles: Iterable[Vehicle] | None = None,
        vehicle_ids: Iterable[int] | None = None,
        states: Iterable[CurvilinearVehicleState] | None = None,
        scenario_type: ScenarioType = ScenarioType.INTERSTATE,
    ):
        self._world = world
        self._scenario = scenario
        self._time_step = time_step
        self._vehicles = list(vehicles) if vehicles is not None else None
        self._states = list(states) if states is not None else None
        self._vehicle_ids = list(vehicle_ids) if vehicle_ids is not None else None
        self._scenario_type = scenario_type

    @classmethod
    def from_kwargs(cls, **kwargs: Unpack[StateContextKwargs]) -> "StateContext":
        """
        Create a StateContext from keyword arguments.

        Args:
            **kwargs: Keyword arguments matching StateContextKwargs

        Returns:
            StateContext: A new StateContext instance

        Raises:
            StateContextError: If required arguments are missing
        """
        if "time_step" not in kwargs:
            raise StateContextError("time_step is required")

        return cls(
            time_step=kwargs["time_step"],
            world=kwargs.get("world"),
            scenario=kwargs.get("scenario"),
            vehicles=kwargs.get("vehicles"),
            vehicle_ids=kwargs.get("vehicle_ids"),
            states=kwargs.get("states"),
            scenario_type=kwargs.get("scenario_type", ScenarioType.INTERSTATE),
        )

    @property
    def world(self) -> World:
        if self._world is None:
            if self._scenario is not None:
                self._world = World.create_from_scenario(self._scenario)
            else:
                raise StateContextError

        return self._world

    @property
    def scenario(self) -> Scenario:
        if self._scenario is None:
            if self._world is not None:
                self._scenario = self._world.scenario
            else:
                raise StateContextError

        return self._scenario

    @property
    def scenario_type(self) -> ScenarioType:
        return self._scenario_type

    @property
    def time_step(self) -> int:
        return self._time_step

    def lon_state(self, vehicle_index: int) -> StateLongitudinal:
        return self.vehicle(vehicle_index).get_lon_state(self.time_step)

    def lat_state(self, vehicle_index: int) -> StateLateral:
        return self.vehicle(vehicle_index).get_lat_state(self.time_step)

    @property
    def vehicles(self) -> list[Vehicle]:
        if self._vehicles is None:
            if self._vehicle_ids is not None:
                self._vehicles = [
                    self.world.vehicle_by_id(vehicle_id) for vehicle_id in self._vehicle_ids
                ]
            else:
                raise StateContextError

        return self._vehicles

    # TODO: Maybe an enum can be used to access the vehicles, instead of the index.
    def vehicle(self, vehicle_index: int) -> Vehicle:
        if vehicle_index >= len(self.vehicles):
            raise StateContextError(
                f"Failed to get vehicle at index {vehicle_index} from available vehicles {self.vehicle_ids}"
            )
        return self.vehicles[vehicle_index]

    @property
    def vehicle_ids(self) -> list[int]:
        """Get all vehicle IDs, deriving them if necessary."""
        if self._vehicle_ids is None:
            if self._vehicles is not None:
                self._vehicle_ids = [vehicle.id for vehicle in self._vehicles]
            else:
                raise StateContextError("Cannot derive vehicle IDs from available context")

        return self._vehicle_ids

    def obstacle(self, vehicle_index: int) -> Obstacle:
        """Get the obstacle corresponding to the vehicle at the specified index."""
        return self.scenario.obstacle_by_id(self.vehicle_ids[vehicle_index])

    def reversed(self) -> "StateContext":
        return StateContext(
            time_step=self._time_step,
            world=self._world,
            scenario=self._scenario,
            vehicles=reversed(self._vehicles) if self._vehicles is not None else None,
            vehicle_ids=reversed(self._vehicle_ids) if self._vehicle_ids is not None else None,
            scenario_type=self._scenario_type,
        )


P = ParamSpec("P")
R = TypeVar("R")


def with_state_context(func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator that automatically creates a StateContext from function arguments.

    This allows functions to be written either with explicit StateContext parameter
    or with individual parameters that get converted to a StateContext.

    Args:
        func: The function to decorate

    Returns:
        Decorated function that handles both StateContext and individual parameters
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        sig = inspect.signature(func)
        parameters = sig.parameters

        # Check if the first parameter is annotated as StateContext
        has_context_param = False
        context_param_name = None

        for name, param in parameters.items():
            if param.annotation == StateContext:
                has_context_param = True
                context_param_name = name
                break

        if not has_context_param:
            # If no StateContext parameter, just call the original function
            return func(*args, **kwargs)

        # Check if a StateContext is already provided
        bound_args = sig.bind_partial(*args, **kwargs)
        if context_param_name in bound_args.arguments:
            # StateContext already provided, just call the function
            return func(*args, **kwargs)

        # Extract parameters needed for StateContext
        context_kwargs = {}
        for key in [
            "time_step",
            "world",
            "scenario",
            "vehicles",
            "vehicle_ids",
            "states",
            "scenario_type",
        ]:
            if key in kwargs:
                context_kwargs[key] = kwargs.pop(key)

        if "time_step" not in context_kwargs:
            raise StateContextError("time_step is required for StateContext")

        # Create StateContext
        context = StateContext(**context_kwargs)

        # Add StateContext to kwargs
        kwargs[context_param_name] = context

        # Call the original function
        return func(*args, **kwargs)

    return wrapper
