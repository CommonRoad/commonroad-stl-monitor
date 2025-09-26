import itertools
import logging
import math
import multiprocessing
import traceback
from collections.abc import Iterable
from pathlib import Path
from typing import Generator, Tuple

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.util import Interval
from commonroad.scenario.obstacle import DynamicObstacle, TrajectoryPrediction
from commonroad.scenario.scenario import Scenario

from crmonitor.common import ScenarioType, World
from crmonitor.mpr.learning.data_loader import DataLoader, EntryId, FeatureEntry, PredicateEntry
from crmonitor.mpr.learning.feature_extractor import (
    FeatureExtractor,
    FeatureVariablesValueCollection,
)
from crmonitor.mpr.mpr_predicate_evaluator import (
    MprPredicateEvaluationResult,
    MprPredicateEvaluator,
)
from crmonitor.predicates.base import AbstractPredicate, PredicateName

_LOGGER = logging.getLogger(__name__)

_DataEntryType = tuple[
    EntryId, FeatureVariablesValueCollection, dict[PredicateName, MprPredicateEvaluationResult]
]


class DataGenerator:
    """
    Generates learning data for gaussian process, by evaluating predicates and extracting features from scenarios.
    """

    def __init__(
        self,
        predicates: Iterable[AbstractPredicate],
        scenarios_path: Path,
        dt: float,
        output_path: Path,
        vehicle_pair_steps: int = 20,
        vehicle_pair_combinations_limit: int = 20,
        state_sampling_time_horizon: float = 1.5,
        time_steps_per_scenario: int = 1,
        scenario_type: ScenarioType = ScenarioType.INTERSTATE,
        snapshot_frequency: int | None = None,
    ) -> None:
        """
        Initializes the DataGenerator with scenario configuration and sampling parameters.

        Args:
            predicate_names (List[str]): List of predicate names to evaluate.
            scenarios_path (Path): Directory containing scenario files.
            dt (float): Time difference between steps.
            output_path (Path): The file where the learning data should be written to. If snapshots are enabled, the snapshots will also be written to this file.
            vehicle_pair_steps (int, optional): Sampling frequency for vehicle pairs.
            vehicle_pair_combinations_limit (int, optional): Maximum number of vehicle pair combinations to process.
            state_sampling_time_horizon (float, optional): Time horizon for state-based sampling. Defaults to 1.5.
            time_steps_per_scenario (int, optional): Number of time steps to process per scenario. Defaults to 1.
            scenario_type (ScenarioType, optional): Type of scenario. Defaults to ScenarioType.INTERSTATE.
            snapshot_frequency (int, optional): The number of data entries, after which a new snapshot will be written.
        """
        self._predicates = predicates
        self._scenarios_path = scenarios_path
        self._output_path = output_path
        self._time_steps_per_scenario = time_steps_per_scenario
        self._state_sampling_ts = state_sampling_time_horizon / dt + 1e-6
        self._vehicle_pair_steps = vehicle_pair_steps
        self._vehicle_pair_combinations_limit = vehicle_pair_combinations_limit

        self._snapshot_frequency = snapshot_frequency
        self._entries_since_last_snapshot = 0

        self._data_loader = DataLoader.create_empty(
            [str(predicate.predicate_name) for predicate in predicates]
        )
        self._feature_extractor = FeatureExtractor.for_scenario_type(scenario_type)
        self._mpr_evaluator = MprPredicateEvaluator(predicates, scenario_type)

    def _vehicle_ids_iter(
        self, scenario: Scenario, time_step: int
    ) -> Generator[tuple[int, ...], None, None]:
        """
        Iterates over valid pairs of vehicle IDs in a scenario at a given time step.

        Args:
            scenario (Scenario): The scenario containing dynamic obstacles.
            time_step (int): Current time step for processing.

        Yields:
            Tuple[int, ...]: A pair of vehicle IDs meeting the filtering criteria.
        """
        id_weight_dict: dict[int, float] = dict()  # vehicle_id: weight
        for dynamic_obstacle in scenario.dynamic_obstacles:
            assert isinstance(dynamic_obstacle.prediction, TrajectoryPrediction)
            if not (
                dynamic_obstacle.initial_state.time_step <= time_step
                and dynamic_obstacle.prediction.final_time_step
                >= (time_step + self._state_sampling_ts)
            ):
                continue

            if dynamic_obstacle.state_at_time(time_step).velocity <= 0.001:
                continue

            id_weight_dict[dynamic_obstacle.obstacle_id] = vehicle_weight_for_sampling(
                dynamic_obstacle
            )

        vehicle_ids_list = np.array(
            list(itertools.combinations(id_weight_dict, 2)), dtype="int,int"
        )
        vehicle_ids_weights = np.array(
            [
                id_weight_dict[ego_id] + id_weight_dict[other_id]
                for ego_id, other_id in vehicle_ids_list
            ]
        )
        if len(vehicle_ids_list) == 0 or len(vehicle_ids_weights) == 0:
            _LOGGER.warning(
                f"Found no qualifing vehicles in scenario {scenario.scenario_id} at time step {time_step}."
            )
            return
        vehicle_ids_weights = vehicle_ids_weights / np.sum(vehicle_ids_weights)  # normalize weights
        vehicle_ids_list = np.random.choice(
            vehicle_ids_list,
            size=math.ceil(len(vehicle_ids_list) / self._vehicle_pair_steps),
            p=vehicle_ids_weights,
            replace=False,
        )  # apply vehicle pair steps -- random is used to avoid bias (previously: i % self._vehicle_pair_steps == 0)
        if len(vehicle_ids_list) > self._vehicle_pair_combinations_limit:
            _LOGGER.info(
                f"Reducing number of vehicle pairs from {len(vehicle_ids_list)} to {self._vehicle_pair_combinations_limit}"
            )
            vehicle_ids_list = np.random.choice(
                vehicle_ids_list, self._vehicle_pair_combinations_limit, replace=False
            )

        yield from vehicle_ids_list

    def _process_vehicles(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> _DataEntryType:
        """
        Processes a pair of vehicles by evaluating predicates and extracting features.

        Args:
            vehicle_ids (Tuple[int, ...]): Pair of vehicle IDs.
            time_step (int): Time step at which processing occurs.
            world_mpr (World): World state wrapper created from the scenario.

        Returns:
            Tuple[dict, dict, dict]: A tuple containing:
                - dict_entry_id: Identifiers including scenario ID, time step, and vehicle IDs.
                - features_dict: Extracted features for the vehicles.
                - predicates_dict: Evaluated predicate values.
        """
        _LOGGER.debug(
            f"Processing the vehicles {vehicle_ids} in scenario {world.scenario.scenario_id} at time step {time_step}"
        )

        evaluation_result = self._mpr_evaluator.evaluate(world, time_step, vehicle_ids)
        feature_values = self._feature_extractor.extract_feature_values(
            world, time_step, vehicle_ids
        )

        entry_id = EntryId(
            scenario_id=str(world.scenario.scenario_id),
            time_step=time_step,
            ego_id=vehicle_ids[0],
            other_id=vehicle_ids[1],
        )

        return (entry_id, feature_values, evaluation_result)

    def _process_scenario(self, scenario: Scenario) -> list[_DataEntryType]:
        """
        Processes a single scenario file.

        Loads the scenario, iterates over specified time steps and vehicle pairs,
        and collects data entries by processing vehicle pairs two times.

        Args:
            scenario_path (Path): Path to the scenario file.
            result_pipe (Connection): Connection object to transfer resuting data entries.

        Raises:
            RuntimeError: If the scenario cannot be processed.
        """
        current_time_step = None
        current_vehicle_ids = None
        try:
            world = World.create_from_scenario(scenario)

            end_time = _get_scenario_final_time_step(scenario) - self._state_sampling_ts - 1
            data_entries = []

            for time_step in np.linspace(0, end_time, self._time_steps_per_scenario, dtype=int):
                # Bind to outer scope variable, for better error reporting.
                current_time_step = int(time_step)
                for vehicle_ids in self._vehicle_ids_iter(scenario, current_time_step):
                    current_vehicle_ids = tuple(vehicle_ids)
                    data_entry = self._process_vehicles(
                        world,
                        current_time_step,
                        current_vehicle_ids,
                    )
                    data_entries.append(data_entry)

                    data_entry = self._process_vehicles(
                        world, current_time_step, tuple(reversed(current_vehicle_ids))
                    )
                    data_entries.append(data_entry)

            return data_entries

        except Exception as exp:
            _LOGGER.debug(traceback.format_exc())
            time_step_hint = (
                f" at time step {current_time_step}" if current_time_step is not None else ""
            )
            vehicle_id_hint = (
                f" vehicles {current_vehicle_ids}" if current_vehicle_ids is not None else ""
            )
            raise RuntimeError(
                f"Failed to process scenario {scenario.scenario_id}{time_step_hint}{vehicle_id_hint}: {exp}"
            ) from exp

    def _handle_result_data_entry(self, data_entry: _DataEntryType) -> None:
        """
        Record a data entry result from scenario processing and optionally writes a snapshot.

        Args:
            data_entry (Tuple[dict, dict, dict]): The data entry result from `_process_scenario`.
        """
        try:
            entry_id, feature_value_collection, mpr_results = data_entry

            feature_entries = []
            for agent_combination, feature_variables in feature_value_collection.as_dict().items():
                for feature_name, value in feature_variables.items():
                    feature_entry = FeatureEntry(
                        agent_combination=agent_combination.value,
                        feature_name=feature_name,
                        value=value,
                    )
                    feature_entries.append(feature_entry)

            predicate_entries = []
            for predicate_name, mpr_results in mpr_results.items():
                for metric, value in mpr_results.as_dict().items():
                    predicate_entry = PredicateEntry(
                        predicate_name=predicate_name, metric=metric, value=value
                    )
                    predicate_entries.append(predicate_entry)

            self._data_loader.append(entry_id, feature_entries, predicate_entries)
            self._entries_since_last_snapshot += 1
        except Exception as e:
            # Catch all exceptions, because if an exception occurs here, this will crash the data generation.
            _LOGGER.debug(traceback.format_exc())
            _LOGGER.error(
                "Error occured in success callback, while processing data entry for %s: %s",
                data_entry[0],
                e,
            )

        if (
            self._snapshot_frequency is not None
            and self._entries_since_last_snapshot >= self._snapshot_frequency
        ):
            _LOGGER.info(f"Writing data snapshot to {self._output_path}")
            self._data_loader.save_data(self._output_path)
            self._entries_since_last_snapshot = 0

    def _process_scenario_success_callback(self, data_entries: list[_DataEntryType]) -> None:
        """
        Callback invoked upon successful processing of a scenario.

        Args:
            data_entries (List[Tuple[dict, dict, dict]]): Processed data entries from a scenario.
        """
        if len(data_entries) == 0:
            _LOGGER.warning(
                "No learning data was generated for a scenario. If you see alot messages of this kind, it indicates a problem with your data generation parameters."
            )
        for data_entry in data_entries:
            self._handle_result_data_entry(data_entry)

    def _process_scenario_error_callback(self, exp: BaseException) -> None:
        """
        Callback invoked if processing a scenario fails.

        Args:
            exp (BaseException): Exception raised during scenario processing.
        """
        _LOGGER.error("Failed to process scenario: %s", str(exp))

    def generate_data(self, workers: int | None = None, limit: int | None = None) -> None:
        """
        Generates data by processing all scenarios using a multiprocessing pool.

        Initializes an empty dataframe in the data loader, then applies asynchronous
        processing to each scenario file in the provided directory.

        Args:
            workers (int, optional): Number of parallel worker processes to use. Defaults to the number of CPU cores.
        """

        if workers is None:
            workers = multiprocessing.cpu_count()

        pool = multiprocessing.Pool(workers)

        total_scenarios = 0
        scenarios_skipped = 0

        for i, scenario_path in enumerate(self._scenarios_path.iterdir()):
            if i % 100 == 0:
                _LOGGER.info(f"Checked out {i} scenarios so far.")
                _LOGGER.info(
                    f"Ratio of skipped scenarios: {scenarios_skipped / (scenarios_skipped + total_scenarios + 1e-7):.3f}"
                )

            scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)
            min_vel, max_vel, min_acc, max_acc = min_max_vel_acc_in_scenario(scenario)
            # 10 % bounds for min_vel, max_vel, min_acc, max_acc
            # rand: Even if the scenario is not interesting, we still want to process it with a probability of 10%
            if (
                min_vel > 20.1
                and max_vel < 43.7
                and min_acc > -1.51
                and max_acc < 1.08
                and np.random.rand() > 0.1
            ):
                scenarios_skipped += 1
                continue

            _ = pool.apply_async(
                self._process_scenario,
                (scenario,),
                callback=self._process_scenario_success_callback,
                error_callback=self._process_scenario_error_callback,
            )
            total_scenarios += 1

            if limit is not None and total_scenarios >= limit:
                _LOGGER.info(
                    f"Reached scenario limit of {limit}. Stopping adding new scenarios as processing jobs."
                )
                break

        pool.close()
        pool.join()

    def save_data(self, file_path: Path | None = None) -> None:
        """
        Finalizes the data by normalizing robustness and saves the results to a CSV file.

        Args:
            file_path (Path, optional): Destination file path for the CSV output. If None, the output path from the `DataGenerator`s construct is used.
        """
        if file_path is None:
            file_path = self._output_path
        self._data_loader.save_data(file_path)


def _get_scenario_final_time_step(scenario: Scenario) -> int:
    """
    Determines the maximum time step in a scenario. This is usefull, to determine the length of a scenario.

    :param scenario: The scenario to analyze.

    :return: The final time step in the scenario, or 0 if no obstacles are in the scenario.
    """
    max_time_step = 0
    for dynamic_obstacle in scenario.dynamic_obstacles:
        if dynamic_obstacle.prediction is None:
            max_time_step = max(max_time_step, dynamic_obstacle.initial_state.time_step)
            continue

        max_time_step = max(max_time_step, dynamic_obstacle.prediction.final_time_step)

    if isinstance(max_time_step, Interval):
        return int(max_time_step.end)
    else:
        return max_time_step


def min_max_vel_acc_in_scenario(scenario: Scenario) -> tuple[float, float, float, float]:
    min_vel = np.inf
    max_vel = -np.inf
    min_acc = np.inf
    max_acc = -np.inf
    for vehicle in scenario.dynamic_obstacles:
        velocities = [state.velocity for state in vehicle.prediction.trajectory.state_list]
        accelerations = [state.acceleration for state in vehicle.prediction.trajectory.state_list]
        min_vel = min(min_vel, min(velocities))
        max_vel = max(max_vel, max(velocities))
        min_acc = min(min_acc, min(accelerations))
        max_acc = max(max_acc, max(accelerations))
    return min_vel, max_vel, min_acc, max_acc


def min_max_vel_acc_of_vehicle(vehicle: DynamicObstacle) -> Tuple[float, float, float, float]:
    velocities = [state.velocity for state in vehicle.prediction.trajectory.state_list]
    accelerations = [state.acceleration for state in vehicle.prediction.trajectory.state_list]
    return min(velocities), max(velocities), min(accelerations), max(accelerations)


def vehicle_weight_for_sampling(vehicle: DynamicObstacle) -> float:
    """
    Returns a weight for a vehicle to be sampled, based on the velocity and acceleration of the vehicle.

    Args:
        vehicle (DynamicObstacle): The vehicle to compute the weight for.

    Returns:
        float: The weight of the vehicle.
    """
    min_vel, max_vel, min_acc, max_acc = min_max_vel_acc_of_vehicle(vehicle)
    # using the 10% bounds for min_vel, max_vel, min_acc, max_acc
    # and their span for balancing vel and acc
    weight = max(
        5.0,  # minimum weight
        20.1 - min_vel,  # 20.1 is the 10% minimum bound
        max_vel - 43.7,  # 43.7 is the 10% maximum bound
        (-1.51 - min_acc) * 9,  # -1.51 is the 10% minimum bound
        (max_acc - 1.08) * 9,  # 1.08 is the 10% maximum bound
    )
    return weight
