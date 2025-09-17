import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Generator, Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto

import commonroad_dc.feasibility.feasibility_checker as feasibility_checker
import numpy as np
from commonroad_clcs.clcs import CurvilinearCoordinateSystem
from commonroad_clcs.util import compute_pathlength_from_polyline
from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics
from vehiclemodels.vehicle_parameters import VehicleParameters

from crmonitor.common import ScenarioType
from crmonitor.common.vehicle import (
    CUSTOM_DEFAULT_VEHICLE_DYNAMICS,
    CurvilinearVehicleState,
    CurvilinearVehicleTrajectory,
    StateLateral,
    StateLongitudinal,
)
from crmonitor.common.world import World
from crmonitor.mpr.prediction.error import SamplingError
from crmonitor.mpr.prediction.polynomial import Polynomial
from crmonitor.mpr.prediction.sampling_x_dimensional import (
    DimensionData,
    LonLatData,
    Sampling1DParams,
    SamplingDimension,
    SamplingOrder,
    SamplingXD,
    SamplingXDConfig,
    SamplingXDParams,
    XDimensionalIterator,
)
from crmonitor.mpr.state_context import StateContext

_LOGGER = logging.getLogger(__name__)


class VelocityMode(Enum):
    """
    The velocity modes are used to select how the lateral velocities are determined
    from the combined velocity vector.
    """

    LOW_VELOCITY_MODE = auto()
    HIGH_VELOCITY_MODE = auto()


@dataclass
class LonLatState(LonLatData[float]):
    @classmethod
    def long_lat_state_from_curvilinear_state(
        cls,
        state: tuple[StateLongitudinal, StateLateral],
        clcs: CurvilinearCoordinateSystem,
        velocity_mode: VelocityMode = VelocityMode.HIGH_VELOCITY_MODE,
    ) -> "LonLatState":
        state_lon, state_lat = state
        s, d = state_lon.s, state_lat.d

        kappa_r = _compute_curvature_in_clcs(clcs, s)
        kappa_r_p = _compute_curvature_prime_in_clcs(clcs, s)

        one_krd = 1 - kappa_r * d
        cos_dtheta = np.cos(state_lat.theta)
        tan_dtheta = np.tan(state_lat.theta)

        # Geometric derivation for rate of arc length change.
        d_p = one_krd * tan_dtheta
        krpd_krdp = kappa_r_p * d + kappa_r * d_p
        d_pp = -krpd_krdp * tan_dtheta + (one_krd / (cos_dtheta**2)) * (
            state_lat.kappa * one_krd / cos_dtheta - kappa_r
        )

        s_d = state_lon.v * cos_dtheta / one_krd
        s_dd = state_lon.a
        s_dd -= (s_d**2 / cos_dtheta) * (
            one_krd * tan_dtheta * (state_lat.kappa * one_krd / cos_dtheta - kappa_r) - krpd_krdp
        )
        s_dd /= one_krd / cos_dtheta

        if velocity_mode == VelocityMode.HIGH_VELOCITY_MODE:
            d_d = state_lon.v * np.sin(state_lat.theta)
            d_dd = s_dd * d_p + s_d**2 * d_pp
        else:
            # Makes only sense if s_d ~= 1.0
            d_d = d_p
            d_dd = d_pp

        return LonLatState(
            lon=DimensionData(position=s, velocity=s_d, acceleration=s_dd),
            lat=DimensionData(position=d, velocity=d_d, acceleration=d_dd),
        )

    def get_dimension_boundaries(self, dimension: SamplingDimension) -> list[float | None]:
        boundaries = []
        for order in SamplingOrder:
            boundaries.append(self.get_dimension(dimension).get_order(order))

        return boundaries


_DEFAULT_SAMPLING_SIZE = LonLatData(
    lon=DimensionData(position=12, velocity=12, acceleration=12),
    lat=DimensionData(position=12, velocity=12, acceleration=12),
)
_DEFAULT_SAMPLING_ORDERS_LON = frozenset({SamplingOrder.VELOCITY})
_DEFAULT_SAMPLING_ORDERS_LAT = frozenset({SamplingOrder.POSITION, SamplingOrder.VELOCITY})


@dataclass(kw_only=True)
class EndStateOptions:
    number: int
    """Number of samples for monte-carlo prediction."""

    d_radius: float
    """Allowed lateral radius around the initial lateral position."""

    d_dot_radius: float
    """Absolute allowed lateral velocity interval [-d_dot_radius, d_dot_radius]."""

    size: LonLatData[float] = field(default_factory=lambda: _DEFAULT_SAMPLING_SIZE)
    """The size of the grid during grid sampling."""

    """The selected sampling orders for each sampling dimension."""
    lon_orders: frozenset[SamplingOrder] = field(
        default_factory=lambda: deepcopy(_DEFAULT_SAMPLING_ORDERS_LON)
    )
    lat_orders: frozenset[SamplingOrder] = field(
        default_factory=lambda: deepcopy(_DEFAULT_SAMPLING_ORDERS_LAT)
    )

    s_d_max: float = float("inf")

    v_delta: float = 1.0

    def iter_orders(self) -> Generator[tuple[SamplingDimension, SamplingOrder], None, None]:
        for order in self.lon_orders:
            yield SamplingDimension.LON, order

        for order in self.lat_orders:
            yield SamplingDimension.LAT, order


DEFAULT_HIGH_VELOCITY_END_STATE_OPTIONS_INTERSTATE = EndStateOptions(
    number=1000,
    d_radius=5,
    d_dot_radius=3,
)

DEFAULT_LOW_VELOCITY_END_STATE_OPTIONS_INTERSECTION = EndStateOptions(
    number=1000,
    d_radius=1.56,
    d_dot_radius=0.2,
)

DEFAULT_HIGH_VELOCITY_END_STATE_OPTIONS_INTERSECTION = EndStateOptions(
    number=1500,
    d_radius=1.5,
    d_dot_radius=3,
)


@dataclass(kw_only=True)
class SwitchableEndStateOptions:
    low_velocity_mode: EndStateOptions | None = None
    high_velocity_mode: EndStateOptions | None = None

    mode_switch_threshold: float | None = None

    def __post_init__(self) -> None:
        has_both_modes = self.high_velocity_mode is not None and self.low_velocity_mode is not None
        has_mode_switch_threshold = self.mode_switch_threshold is not None

        if has_mode_switch_threshold and not has_both_modes:
            raise ValueError("If ")

        if has_both_modes and not has_mode_switch_threshold:
            raise ValueError()

    @classmethod
    def default_for_scenario_type(cls, scenario_type: ScenarioType) -> "SwitchableEndStateOptions":
        if scenario_type == ScenarioType.INTERSTATE:
            return cls(high_velocity_mode=DEFAULT_HIGH_VELOCITY_END_STATE_OPTIONS_INTERSTATE)
        else:
            return cls(
                low_velocity_mode=DEFAULT_LOW_VELOCITY_END_STATE_OPTIONS_INTERSECTION,
                high_velocity_mode=DEFAULT_HIGH_VELOCITY_END_STATE_OPTIONS_INTERSECTION,
                mode_switch_threshold=4.0,
            )

    def get_velocity_mode_for_state(self, state: LonLatState) -> VelocityMode:
        """
        Determine the appropriate velocity mode based on current longitudinal velocity.

        The velocity mode affects which sampling parameters are used. Different modes
        may have different sampling strategies for low-speed vs high-speed scenarios.

        Args:
            state: Current multi-dimensional state containing velocity information.

        Returns:
            The determined velocity mode.
        """
        has_multiple_velocity_modes = self.mode_switch_threshold is not None
        if not has_multiple_velocity_modes:
            velocity_mode = (
                VelocityMode.HIGH_VELOCITY_MODE
                if self.high_velocity_mode is not None
                else VelocityMode.LOW_VELOCITY_MODE
            )
            return velocity_mode

        if state.lon.velocity <= self.mode_switch_threshold:
            return VelocityMode.LOW_VELOCITY_MODE
        else:
            return VelocityMode.HIGH_VELOCITY_MODE

    def get_end_state_options_for_state(self, state: LonLatState) -> EndStateOptions:
        """
        Get the end state options appropriate for the given state's velocity mode.

        Use this instead of directly retrieving the end state options from the sampler config.

        Args:
            state: Current multi-dimensional state

        Returns:
            Sampling parameters for the appropriate velocity mode.
        """
        velocity_mode = self.get_velocity_mode_for_state(state)
        if velocity_mode == VelocityMode.HIGH_VELOCITY_MODE:
            return self.high_velocity_mode
        else:
            return self.low_velocity_mode


@dataclass
class FutureStateSamplerConfig:
    # The setup for `end_state_options` was adopted from the original code, with only slight modifications.
    # It is suboptimal, since it is not very easy for users to selectively override options from the defaults.
    # TODO: Find a better solution to provide the `end_state_options` alongside easy to configure defaults.
    end_state_options: SwitchableEndStateOptions
    """Options for the sampled end states. The sampling boundaries will be derived from these options."""

    samplingxd: SamplingXDConfig = field(default_factory=SamplingXDConfig)

    time_horizon_sec: float = 1.5
    """Specify the sampling time horizon in seconds."""

    eps: float = 1e-6

    @classmethod
    def default_config_for_scenario_type(
        cls, scenario_type: ScenarioType = ScenarioType.INTERSTATE
    ) -> "FutureStateSamplerConfig":
        return cls(SwitchableEndStateOptions.default_for_scenario_type(scenario_type))


class StateBasedSamplingResult:
    """
    Result of state sampling.

    Provides a simple iterator interface, which can be used to retrieve all cartesian sampled states.
    Additionally, the curvilinear time-series trajectories can be accessed, if no
    cartesian commonroad trajectory is required.
    """

    def __init__(
        self,
        sampled_trajectories: Iterable[CurvilinearVehicleTrajectory],
        clcs: CurvilinearCoordinateSystem,
    ) -> None:
        self.sampled_trajectories = sampled_trajectories
        self._clcs = clcs

    def __iter__(self) -> Generator[CurvilinearVehicleState]:
        for trajectory in self.sampled_trajectories:
            for time_step in range(trajectory.initial_time_step, trajectory.final_time_step + 1):
                yield trajectory.state_at_time_step(time_step, self._clcs)


class FutureStateSampler:
    """
    Sampler to simulate the future states of a vehicle.

    Args:
        scenario_type: Specify the scenario type for which this sampler will be used.
                       The concrete sampler implementation will be choosen for the given scenario type.
        config: Optionally provide a sampler config.
                If None is provided, the default config for the scenario type is used.
    """

    def __init__(
        self,
        scenario_type: ScenarioType = ScenarioType.INTERSTATE,
        config: FutureStateSamplerConfig | None = None,
    ) -> None:
        if config is None:
            config = FutureStateSamplerConfig.default_config_for_scenario_type(scenario_type)

        if scenario_type == ScenarioType.INTERSTATE:
            self._sampler = InterstateFutureStateSampler(config)
        else:
            raise NotImplementedError
            # self._sampler = IntersectionFutureStateSampler(config)

    def sample(self, world: World, time_step: int, vehicle_id: int) -> StateBasedSamplingResult:
        """
        Sample the

        Args:
            world: The world in which this will sample.
            time_step: Initial time step from which new states will be sampled.
            vehicle_id: The vehicle from `world` for which the future states will be sampled.

        Returns:
            The sampling result object.

        Raises:
            SamplingError: If the sampling fails.
        """

        # TODO: Should the ctx be passed from the outside?
        ctx = StateContext(time_step=time_step, world=world, vehicle_ids=(vehicle_id,))
        try:
            return self._sampler.sample_from_state(ctx, CUSTOM_DEFAULT_VEHICLE_DYNAMICS)
        except Exception as e:
            raise SamplingError(
                f"State sampling for vehicle {vehicle_id} in scenario {world.scenario.scenario_id} at time step {time_step} failed: {e}"
            ) from e


class AbstractFutureStateSampler(ABC):
    """
    Abstract state sampler which should implement
    """

    def __init__(
        self,
        config: FutureStateSamplerConfig,
    ):
        self._config = config

        self._xd_sampler: SamplingXD = SamplingXD(
            config.samplingxd.distribution, config.samplingxd.simulation
        )

        # variables for selecting sampling parameters experiment
        self.num_all_feasible = 0
        self.num_all_sampling = 0

    @abstractmethod
    def sample_from_state(
        self,
        ctx: StateContext,
        vehicle_dynamics: VehicleDynamics,
    ) -> StateBasedSamplingResult:
        """
        Sample future vehicle trajectories from a given initial state.

        This is the main interface method that concrete implementations must provide.
        It should generate a collection of possible future trajectories based on the
        initial state and vehicle dynamics.

        Args:
            initial_state: Tuple containing initial longitudinal and lateral states.
            time_step: Initial time step from which future states are sampled.
            dt: Time step size of the scenario.
            lane_clcs: Curvilinear coordinate system which is used as the reference for the state. Should be the current lane of the vehicle.
            vehicle_dynamics: Vehicle dynamics model to which the sampled trajectories must conform to.

        Yields:
            CurvilinearVehicleTrajectory: Generated vehicle trajectories in curvilinear coordinates.
        """
        ...

    @abstractmethod
    def _get_sampling_1d_params_for_dimension(
        self,
        state: LonLatState,
        vehicle_dynamics: VehicleDynamics,
        dimension: SamplingDimension,
        order: SamplingOrder,
    ) -> Sampling1DParams:
        """
        Get the sampling parameters for a derivative order in a dimension.

        The sampling parameters might depend on the vehicle dynamics and the configured
        end state options.

        Args:
            state: The initial state from which future states are sampled.
            vehicle_dynamics: Vehicle dynamics model that may influence sampling parameters.
            dimension: The specific dimension (LONG/LAT) for sampling.
            order: The derivative order (POSITION/VELOCITY/ACCELERATION) for sampling.

        Returns:
            Sampling params for the order and dimension.

        Raises:
            NotImplementedError: If the dimension and order combination is not supported.
        """
        ...

    def _get_sampling_xd_params_for_state(
        self, state: LonLatState, vehicle_dynamics: VehicleDynamics
    ) -> SamplingXDParams:
        """
        Generate multi-dimensional sampling parameters for the given state.

        This method orchestrates the creation of sampling parameters across all
        dimensions and orders specified in the configuration. It requires the abstract
        method _get_sampling_1d_params_for_dimension for each required combination.

        Args:
            state: Current multi-dimensional state to sample from.
            vehicle_dynamics: Vehicle dynamics which might influence the sampling parameters.

        Returns:
            SamplingXDParams: Complete set of multi-dimensional sampling parameters

        Raises:
            SamplingError: If sampling parameters cannot be generated for any
                          required dimension/order combination
        """
        end_state_options = self._config.end_state_options.get_end_state_options_for_state(state)

        sampling_xd_params = defaultdict(dict)
        for sampling_dimension, sampling_order in end_state_options.iter_orders():
            try:
                sampling_1d_params = self._get_sampling_1d_params_for_dimension(
                    state, vehicle_dynamics, sampling_dimension, sampling_order
                )
            except ValueError as e:
                raise SamplingError(
                    f"Failed to get 1d sampling params for dimension '{sampling_dimension}' and order '{sampling_order}': {e}"
                ) from e
            sampling_xd_params[sampling_dimension][sampling_order] = sampling_1d_params

        return SamplingXDParams(
            sample_number=end_state_options.number, sampling_dimensions=sampling_xd_params
        )

    def _is_valid_end_state(
        self, end_state: LonLatState, vehicle_dynamics: VehicleDynamics
    ) -> bool:
        """
        Test whether a sampled end state is valid for the vehicle dynamics.

        Arguments:
            end_state: The sampled state.
            vehicle_dynamics: Vehicle dynamics which the end state must adhere to.

        Returns:
            Whether the state is a valid end state or not.
        """
        vehicle_params: VehicleParameters = vehicle_dynamics.parameters

        if end_state.lon.has_order(SamplingOrder.VELOCITY) and end_state.lat.has_order(
            SamplingOrder.VELOCITY
        ):
            long_velocity = end_state.lon.velocity
            lat_velocity = end_state.lat.velocity

            # no backwards
            # TODO: Shouldn't backward driving be allowed?
            if long_velocity <= 0.001:
                return False

            velocity = np.linalg.norm([long_velocity, lat_velocity])
            if velocity > vehicle_params.longitudinal.v_max:
                return False

        if end_state.lon.has_order(SamplingOrder.ACCELERATION) and end_state.lat.has_order(
            SamplingOrder.ACCELERATION
        ):
            long_acceleration = end_state.lon.acceleration
            lat_acceleration = end_state.lat.acceleration
            acceleration = np.linalg.norm([long_acceleration, lat_acceleration])
            if acceleration > vehicle_params.longitudinal.a_max:
                return False

        return True

    def _end_state_sample(
        self, sampling_xd_params: SamplingXDParams, vehicle_dynamics: VehicleDynamics
    ) -> Iterable[LonLatState]:
        """
        Generate and validate end state samples using multi-dimensional sampling.

        This method performs the actual sampling using the configured XD sampler,
        validates that all dimensions have consistent sample counts, and filters
        the results to only yield states which adhere to the vehicle dynamics.

        Args:
            sampling_xd_params: Multi-dimensional sampling parameters.
            vehicle_dynamics: Vehicle dynamics for end state validation.

        Yields:
            XDimensionalState: Valid sampled end states that conform to the vehicle dynamics.

        Raises:
            SamplingError: If no samples are produced or sample arrays have inconsistent lengths.
        """
        sampling_results = self._xd_sampler.sample(sampling_xd_params)

        sampling_lens = [
            len(sample_array) for _, _, sample_array in XDimensionalIterator(sampling_results)
        ]

        if len(sampling_lens) == 0:
            raise SamplingError("Sampler produced no sampled states")

        if min(sampling_lens) != max(sampling_lens):
            raise SamplingError("")

        min_len = min(sampling_lens)
        for i in range(min_len):
            state_prototype = defaultdict(dict)
            for dimension, order, samples in XDimensionalIterator(sampling_results):
                state_prototype[dimension][order] = samples[i]

            x_dimensional_state = LonLatState.from_dict(state_prototype)
            if self._is_valid_end_state(x_dimensional_state, vehicle_dynamics):
                yield x_dimensional_state


class InterstateFutureStateSampler(AbstractFutureStateSampler):
    """
    State sampler for interstate scenarios.

    For generic sampling, prefer to use `FututreStateSampler` instead of using this directly.
    """

    def sample_from_state(
        self,
        ctx: StateContext,
        vehicle_dynamics: VehicleDynamics,
    ) -> StateBasedSamplingResult:
        ref_lane = ctx.vehicle(0).lane_at_time_step(ctx.time_step)
        start_long_lat_state = LonLatState.long_lat_state_from_curvilinear_state(
            (ctx.lon_state(0), ctx.lat_state(0)), ref_lane.clcs
        )

        # sample end states
        sample_params = self._get_sampling_xd_params_for_state(
            start_long_lat_state, vehicle_dynamics
        )

        start_long_boundaries = start_long_lat_state.get_dimension_boundaries(SamplingDimension.LON)
        start_lat_boundaries = start_long_lat_state.get_dimension_boundaries(SamplingDimension.LAT)
        results = []
        num_unfeasible_trajectories = 0
        _LOGGER.debug(
            "Sampling for vehicle %s from time step %s in %s with start boundaries: lon %s; lat %s",
            ctx.vehicle(0).id,
            ctx.time_step,
            ctx.scenario.scenario_id,
            start_long_boundaries,
            start_lat_boundaries,
        )
        for end_long_lat_state in self._end_state_sample(sample_params, vehicle_dynamics):
            long_fun = Polynomial.from_boundary(
                start_long_boundaries,
                end_long_lat_state.get_dimension_boundaries(SamplingDimension.LON),
                self._config.time_horizon_sec,
            )  # TODO I dint verify the .from_boundary method.

            lat_fun = Polynomial.from_boundary(
                start_lat_boundaries,
                end_long_lat_state.get_dimension_boundaries(SamplingDimension.LAT),
                self._config.time_horizon_sec,
            )

            num_ts = int(self._config.time_horizon_sec / ctx.scenario.dt + self._config.eps)
            final_time_step = int(ctx.time_step + num_ts)
            curvilinear_trajectory = _create_trajectory_from_polynomials(
                long_fun,
                lat_fun,
                ctx.scenario.dt,
                ctx.time_step,
                final_time_step,
                ref_lane.clcs,
            )

            commonroad_input_trajectory = (
                curvilinear_trajectory.convert_to_commonroad_input_trajectory(
                    ref_lane.clcs, initial_time_step=ctx.time_step + 1
                )
            )

            # check feasibility
            initial_state = ctx.obstacle(0).state_at_time(ctx.time_step)
            feasible, _ = feasibility_checker.input_vector_feasibility(
                initial_state,
                commonroad_input_trajectory,
                vehicle_dynamics,
                ctx.scenario.dt,
            )

            if not feasible:
                num_unfeasible_trajectories += 1
                continue

            results.append(curvilinear_trajectory)

        num_feasible_trajectories = len(results)
        _LOGGER.debug(
            "Out of %s sampled trajectories for vehicle %s at time step %s in %s, %s trajectories are feasible while %s trajectories are infeasible.",
            num_feasible_trajectories + num_unfeasible_trajectories,
            ctx.vehicle(0).id,
            ctx.time_step,
            ctx.scenario.scenario_id,
            num_feasible_trajectories,
            num_unfeasible_trajectories,
        )

        return StateBasedSamplingResult(results, ref_lane.clcs)

    def _get_sampling_1d_params_for_dimension(
        self,
        state: LonLatState,
        vehicle_dynamics: VehicleDynamics,
        dimension: SamplingDimension,
        order: SamplingOrder,
    ) -> Sampling1DParams:
        vehicle_params: VehicleParameters = vehicle_dynamics.parameters
        end_state_options = self._config.end_state_options.get_end_state_options_for_state(state)

        dimension_value = state.get_dimension(dimension).get_order(order)
        if dimension_value is None:
            raise SamplingError(
                f"State value for order '{order}' in dimension '{dimension}' is None. Expected a valid value."
            )

        grid_size = end_state_options.size.get_dimension(dimension).get_order(order)
        if grid_size is None:
            raise SamplingError(
                f"Grid size for order '{order}' in dimension '{dimension}' is None. Expected a valid value."
            )

        match dimension:
            case SamplingDimension.LON:
                if order != SamplingOrder.VELOCITY:
                    raise NotImplementedError(
                        "Only 'velocity' is supported for longitudinal sampling"
                    )
                return Sampling1DParams(
                    min_val=dimension_value
                    - vehicle_params.longitudinal.a_max * self._config.time_horizon_sec,
                    max_val=dimension_value
                    + vehicle_params.longitudinal.a_max * self._config.time_horizon_sec,
                    size=grid_size,
                )

            case SamplingDimension.LAT:
                match order:
                    case SamplingOrder.POSITION:
                        return Sampling1DParams(
                            min_val=dimension_value - end_state_options.d_radius,
                            max_val=dimension_value + end_state_options.d_radius,
                            size=grid_size,
                        )
                    case SamplingOrder.VELOCITY:
                        return Sampling1DParams(
                            min_val=-end_state_options.d_dot_radius,
                            max_val=end_state_options.d_dot_radius,
                            size=grid_size,
                        )
                    case _:
                        raise NotImplementedError(
                            "Only 'position' and 'velocity' are supported for lateral sampling"
                        )
            case _:
                raise NotImplementedError(
                    f"Dimension {dimension} is not a supported sampling dimension"
                )


# TODO: finish intersection sampling.
# class IntersectionFutureStateSampler(AbstractFutureStateSampler):
#     def sample_from_state(
#         self,
#         initial_state: CurvilinearVehicleState,
#         time_step: int,
#         dt: float,
#         lane_clcs: CurvilinearCoordinateSystem,
#         vehicle_dynamics: VehicleDynamics,
#     ) -> Iterable[CurvilinearVehicleTrajectory]:
#         """yields sampled vehicle states"""
#         ...

#     def get_start_state_intersection(
#         self, lane, mode_switch_threshold
#     ) -> tuple[list[float], list[float]]:
#         return self.vehicle.trajectory_persp.get_long_lat_state_intersection(
#             self.initial_time_step, lane, mode_switch_threshold
#         )

#     def get_possible_lanes(self):
#         lanes_from_start = self.world_state.road_network.find_lanes_by_lanelets(
#             {
#                 self.vehicle.lanelets_dir[0],
#             }
#         )
#         occupied_lanes = self.vehicle.trajectory_persp.get_occupied_lanes(self.initial_time_step)
#         possible_lanes = list(lanes_from_start.intersection(occupied_lanes))
#         if len(possible_lanes) == 1:
#             return possible_lanes
#         selected_lanes = list()
#         # TODO: can remove because lanes are the longest when creation
#         for lane in possible_lanes:
#             subset_find = False
#             for index, selected_lane in enumerate(selected_lanes):
#                 if set(selected_lane.segment_ids).issubset(lane.segment_ids):
#                     subset_find = True
#                     selected_lanes[index] = lane
#                     break
#                 elif set(lane.segment_ids).issubset(selected_lane.segment_ids):
#                     subset_find = True
#                     break
#                 else:
#                     subset_find = False
#             if not subset_find:
#                 selected_lanes.append(lane)
#         if len(selected_lanes) == 0:
#             for lane in lanes_from_start:
#                 if len(set(lane.segment_ids).intersection(self.vehicle.lanelets_dir[1:])) != 0:
#                     selected_lanes.append(lane)
#         return selected_lanes

#     def generate_intersection_cache(self) -> None:
#         lanes = self.get_possible_lanes()
#         result = []
#         # reset number of samples
#         self.num_all_sampling = 0
#         self.num_all_feasible = 0

#         for lane in lanes:
#             try:
#                 long_left, lat_left = self.get_start_state_intersection(
#                     lane, self._mode_switch_threshold
#                 )
#             except Exception:
#                 continue
#             # TODO: velocity mode switching
#             # sample end states
#             sample_params, low_velocity_mode = self.get_sample_params_intersection(
#                 long_left, lat_left
#             )
#             for long_right, lat_right in self.end_state_sample(sample_params):
#                 self.num_all_sampling += 1
#                 # interpolate
#                 long_fun = Polynomial.from_boundary(
#                     long_left, long_right, self._config.time_horizon_sec
#                 )
#                 if low_velocity_mode:
#                     t = np.arange(0, self.dt * self.num_ts, self.dt)
#                     s_goal = long_fun(t[-1]) - long_fun(0)
#                     lat_fun = Polynomial.from_boundary(lat_left, lat_right, s_goal)
#                 else:
#                     lat_fun = Polynomial.from_boundary(
#                         lat_left, lat_right, self._config.time_horizon_sec
#                     )
#                 trajectory_persp = TrajectoryPerspective.create_from_functions_intersection(
#                     long_fun,
#                     lat_fun,
#                     low_velocity_mode,
#                     self.vehicle,
#                     self.initial_time_step,
#                     self.num_ts,
#                     lane,
#                 )

#                 trajectory_persp.assign_reference_lane_intersection(lane)
#                 trajectory_persp.assign_lanelets_dir(lane.segment_ids)

#                 initial_state = self.world_state.scenario.obstacle_by_id(
#                     self.vehicle.id
#                 ).state_at_time(self.initial_time_step)
#                 commonroad_input = trajectory_persp.convert_to_commonroad_input_trajectory()

#                 # check feasibility
#                 self.vehicle.vehicle_dynamics.parameters.steering.v_max = (
#                     self.is_valid_end_state.v_delta
#                 )
#                 self.vehicle.vehicle_dynamics.parameters.steering.v_min = (
#                     self._end_state_options.v_delta
#                 )
#                 feasible, _ = feasibility_checker.input_vector_feasibility(
#                     initial_state,
#                     commonroad_input,
#                     self.vehicle.vehicle_dynamics,
#                     self.dt,
#                 )

#                 if feasible:
#                     result.append(trajectory_persp)
#                     self.num_all_feasible += 1

#         # store result
#         self.vehicle.append_cache(self.hash, StateBasedSamplingResult(result))

#     def _get_sampling_1d_params_for_dimension(
#         self,
#         state: XDimensionalState,
#         vehicle_dynamics: VehicleDynamics,
#         dimension: SamplingDimension,
#         order: SamplingOrder,
#     ) -> Sampling1DParams:
#         vehicle_params: VehicleParameters = vehicle_dynamics.parameters
#         end_state_options = self._get_end_state_options_for_state(state)

#         match dimension:
#             case SamplingDimension.LONG:
#                 if order != SamplingOrder.VELOCITY:
#                     raise ValueError()

#                 long_velocity = state.dimensions[dimension][order]
#                 if end_state_options.s_d_max == float("inf"):
#                     s_d_max = (
#                         long_velocity
#                         + vehicle_params.longitudinal.a_max * self._config.time_horizon_sec
#                     )
#                 else:
#                     s_d_max = end_state_options.s_d_max

#                 return Sampling1DParams(
#                     min_val=max(
#                         0.0,
#                         long_velocity
#                         - vehicle_params.longitudinal.a_max
#                         + self._config.time_horizon_sec,
#                     ),
#                     max_val=s_d_max,
#                 )

#             case SamplingDimension.LAT:
#                 match order:
#                     case SamplingOrder.POSITION:
#                         lat_position = state.dimensions[dimension][order]
#                         return Sampling1DParams(
#                             min_val=lat_position - end_state_options.d_radius,
#                             max_val=lat_position + end_state_options.d_radius,
#                         )

#                     case SamplingOrder.VELOCITY:
#                         return Sampling1DParams(
#                             min_val=end_state_options.d_dot_radius,
#                             max_val=end_state_options.d_dot_radius,
#                         )
#                     case _:
#                         raise ValueError()


def _compute_curvature_in_clcs(
    clcs: CurvilinearCoordinateSystem, position: np.ndarray
) -> np.ndarray:
    path_length = compute_pathlength_from_polyline(clcs.ref_path)
    return np.interp(position, path_length, clcs.ref_curv)


def _compute_curvature_prime_in_clcs(
    clcs: CurvilinearCoordinateSystem, position: np.ndarray
) -> np.ndarray:
    path_length = compute_pathlength_from_polyline(clcs.ref_path)
    return np.interp(position, path_length, clcs.ref_curv_d)


def _create_trajectory_from_polynomials(
    long_fun: np.polynomial.Polynomial,
    lat_fun: np.polynomial.Polynomial,
    dt: float,
    initial_time_step: int,
    final_time_step: int,
    clcs: CurvilinearCoordinateSystem,
) -> CurvilinearVehicleTrajectory:
    t = np.arange(0, dt * (final_time_step - initial_time_step + 1), dt)
    s = long_fun(t)
    s_d = long_fun.deriv(1)(t)
    if not (s_d > 0.001).all():
        raise SamplingError("The vehicle is at low speed, which is not supported")
    s_dd = long_fun.deriv(2)(t)
    d = lat_fun(t)
    d_d = lat_fun.deriv(1)(t)
    d_dd = lat_fun.deriv(2)(t)
    d_p = d_d / s_d
    d_pp = (d_dd - d_p * s_dd) / (s_d) ** 2

    kappa_r = _compute_curvature_in_clcs(clcs, s)
    kappa_r_p = _compute_curvature_prime_in_clcs(clcs, s)
    one_krd = 1 - kappa_r * d
    delta_theta = np.arctan2(d_p, one_krd)
    one_krd = 1 - kappa_r * d
    krpd_krdp = kappa_r_p * d + kappa_r * d_p

    cos_dtheta = np.cos(delta_theta)
    tan_dtheta = np.tan(delta_theta)

    kappa_gl = (d_pp + kappa_r * d_p * tan_dtheta) * cos_dtheta * (cos_dtheta / one_krd) ** 2 + (
        cos_dtheta / one_krd
    ) * kappa_r
    delta_theta_p = kappa_gl * one_krd / cos_dtheta - kappa_r
    a = s_dd * one_krd / cos_dtheta + s_d**2 / cos_dtheta * (
        one_krd * tan_dtheta * delta_theta_p - krpd_krdp
    )
    v = s_d * one_krd / cos_dtheta

    kappa = (
        ((d_pp + krpd_krdp * tan_dtheta) * cos_dtheta**2 / one_krd + kappa_r) * cos_dtheta / one_krd
    )

    trajectory = CurvilinearVehicleTrajectory(
        initial_time_step, final_time_step, dt, s, d, v, a, kappa
    )

    return trajectory
