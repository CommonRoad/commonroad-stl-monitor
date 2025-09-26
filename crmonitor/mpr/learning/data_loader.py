import logging
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd

from crmonitor.common import ScenarioType
from crmonitor.predicates.base import AbstractPredicate

from .feature_variables import (
    default_feature_variable_classes_for_scenario_type,
)

_LOGGER = logging.getLogger(__name__)

# Originally, those values were loaded from the config files.
# The values are the same, and there is currently no need to adjust them, especially since they are hard coded in the data generator.
# TODO: centralize the entry ids across DataLoader and DataGenerator, so that this const is not needed.
ENTRY_ID_NAMES = ("scenario_id", "time_step", "ego_id", "other_id")

# The key for the reference robustness. Normally, this is the value returned by the data generator.
# TODO: This is carried over from the config files. These keys should be managed centrally for DataLoader and DataGenerator.
DATA_LOADER_REFERENCE_NAMES = (
    "count_valid",
    "count_true",
    "count_error",
    "bool",
    "robustness",
    "mfr_robustness",
)

# The key which the dataloader will associate the normalized robustness with.
# TODO: This is carried over from the config files. Can this be solved more elegantly?
DATA_LOADER_OUTPUT_NAME = "normalized_robustness"

# MIN_VALID_COUNT = 10000
MIN_VALID_COUNT = 100


@dataclass(frozen=True, slots=True)
class EntryId:
    """Strongly typed entry identifier replacing tuple-based IDs."""

    scenario_id: str
    time_step: int
    ego_id: int
    other_id: int

    @classmethod
    def from_dict(cls, data: dict[str, str | int]) -> "EntryId":
        """Create EntryId from dictionary."""
        return cls(
            scenario_id=data["scenario_id"],
            time_step=data["time_step"],
            ego_id=data["ego_id"],
            other_id=data["other_id"],
        )

    def to_tuple(self) -> tuple[str, int, int, int]:
        """Convert to tuple for pandas MultiIndex compatibility."""
        return (self.scenario_id, self.time_step, self.ego_id, self.other_id)

    @property
    def names(self) -> tuple[str, ...]:
        """Field names for pandas MultiIndex."""
        return ("scenario_id", "time_step", "ego_id", "other_id")


@dataclass(frozen=True, slots=True)
class DataLoaderKey:
    """Strongly typed key for data access."""

    category: str  # "inputs" or "predicates"
    subcategory: str  # agent combination or predicate name
    name: str  # feature name or reference name

    def to_tuple(self) -> tuple[str, str, str]:
        """Convert to tuple for pandas MultiIndex compatibility."""
        return (self.category, self.subcategory, self.name)


@dataclass
class FeatureEntry:
    agent_combination: str
    feature_name: str
    value: float

    def to_tuple(self) -> tuple[str, str, str]:
        return ("inputs", self.agent_combination, self.feature_name)


@dataclass
class PredicateEntry:
    predicate_name: str
    metric: str
    value: float

    def to_tuple(self) -> tuple[str, str, str]:
        return ("predicates", self.predicate_name, self.metric)


def project(x: "np.ndarray", min: float, max: float) -> "np.ndarray":
    if len(x) <= 1:
        # CAUTION: even an empty list was given, will return [max].
        return [max]
    x_min = np.min(x)
    x_max = np.max(x)
    assert np.sign(x_min) == np.sign(x_max) or x_min == 0.0 or x_max == 0.0, (
        f"The values should have the same sign. x_min={x_min}, x_max={x_max}"
    )
    if x_min == x_max:
        return [max] * len(x)
    norm = (x - x_min) / (x_max - x_min) * (max - min) + min
    return np.round(norm.astype(float), 6)  # TODO why round?


def normalize(robustness: "np.ndarray") -> "np.ndarray":
    mask = robustness > 0
    normalized_robustness = np.array(robustness)
    normalized_robustness[mask] = project(normalized_robustness[mask], 0, 1)
    normalized_robustness[~mask] = project(normalized_robustness[~mask], -1, 0)
    return normalized_robustness


def _keys_for_predicate(predicate_name: str) -> list[tuple[str, str, str]]:
    """Generate keys for predicate references."""
    keys = []
    for reference in list(DATA_LOADER_REFERENCE_NAMES) + [DATA_LOADER_OUTPUT_NAME]:
        keys.append(("predicates", predicate_name, reference))
    return keys


class DataLoader:
    def __init__(
        self,
        predicate_names: list[str],
        data: pd.DataFrame,
        scenario_type: ScenarioType = ScenarioType.INTERSTATE,
    ) -> None:
        self.predicate_names = predicate_names
        self._scenario_type = scenario_type
        self.data = data

    @classmethod
    def create_empty(
        cls, predicate_names: list[str], scenario_type: ScenarioType = ScenarioType.INTERSTATE
    ) -> "DataLoader":
        index = pd.MultiIndex.from_tuples([], names=ENTRY_ID_NAMES)

        input_columns = []
        for (
            agent_combination,
            feature_variables,
        ) in default_feature_variable_classes_for_scenario_type(scenario_type).items():
            for feature_variable in feature_variables:
                input_columns.append(("inputs", agent_combination.value, feature_variable.name))

        predicate_columns = []
        for predicate_name in predicate_names:
            predicate_columns.extend(_keys_for_predicate(predicate_name))

        columns = pd.MultiIndex.from_tuples(input_columns + predicate_columns)
        data = pd.DataFrame(index=index, columns=columns)

        return cls(predicate_names, data)

    def Xy(
        self,
        predicate: type[AbstractPredicate],
        index_features: list[tuple[str, str, str]],
        mfr_data: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """get the input X and output y of the predicate for training and testing regression models.

        Args:
            predicate_name (str): name of the predicate
            flag_info (bool): information for evaluating the predicates

        Returns:
            X : inputs
            y : outputs
            Z : information
        """

        if predicate.arity == 1:
            # reset index
            data = self.data.reset_index(ENTRY_ID_NAMES[-1])
            # remove duplicates
            data = data[~data.index.duplicated(keep="first")]
        else:
            data = deepcopy(self.data)  # TODO necessary?

        cleaned_data = self._clean_data(data, predicate.predicate_name, index_features)
        X = cleaned_data.loc[:, index_features]
        # TODO: fix round
        X = X.round(6)
        y = cleaned_data.predicates[predicate.predicate_name][DATA_LOADER_OUTPUT_NAME]
        # TODO: fix round
        y = y.round(6)
        if mfr_data:
            y_mfr = data.predicates[predicate.predicate_name].mfr_robustness
            y_mfr = y_mfr.round(6)
            return X.values, y.values, y_mfr.values
        else:
            return X.values, y.values

    def _clean_data(
        self, data: pd.DataFrame, predicate_name: str, feature_indices: list[tuple[str, str, str]]
    ) -> pd.DataFrame:
        """
        Clean the data by removing NaN values and invalid samples.

        Args:
            data: DataFrame to clean
            predicate_name: Name of the predicate
            feature_indices: Feature indices to check for -inf values

        Returns:
            Cleaned DataFrame
        """
        # Drop rows with NaN
        data = data.dropna()

        # Drop rows with too few samples
        count_valid_idx = ("predicates", predicate_name, "count_valid")
        invalid_mask = data.loc[:, count_valid_idx] <= MIN_VALID_COUNT

        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            total_count = len(data)
            _LOGGER.debug(
                f"Removed {invalid_count} out of {total_count} rows that don't meet "
                f"the criterion of count_valid > {MIN_VALID_COUNT}."
            )
            data = data[~invalid_mask]

        # Drop rows with -inf values
        for idx_feature in feature_indices:
            try:
                inf_mask = data.loc[:, idx_feature] == -np.inf
            except KeyError as e:
                raise KeyError(
                    f"Failed to find feature variable {idx_feature} in learning data. Maybe this is a new feature variable, which was not included in the data generation yet?"
                ) from e
            if inf_mask.any():
                data = data[~inf_mask]

        return data

    @staticmethod
    def load_data_from_file(file_path: Path):
        data = pd.read_csv(
            file_path,
            header=[0, 1, 2],
            index_col=list(range(len(ENTRY_ID_NAMES))),
        )
        return data

    @classmethod
    def create_from_file(cls, file_path: Path) -> "DataLoader":
        data = cls.load_data_from_file(file_path)
        predicate_names = data.predicates.columns.get_level_values(0).unique().to_list()
        return cls(predicate_names, data)

    def limit_size(self, size: int, rand_state: int = 12345):
        self.data = self.data.sample(size, random_state=rand_state)

    def save_data(self, file_path: Path):
        self.generate_normalize_robustness()
        self.data.to_csv(file_path)

    def append(
        self,
        entry_id: EntryId,
        feature_entries: Iterable[FeatureEntry],
        predicate_entries: Iterable[PredicateEntry],
    ) -> None:
        entry_id_tuple = entry_id.to_tuple()
        self.data.loc[entry_id_tuple, :] = None

        for feature_entry in feature_entries:
            self.data.at[entry_id_tuple, feature_entry.to_tuple()] = feature_entry.value

        for predicate_entry in predicate_entries:
            self.data.at[entry_id_tuple, predicate_entry.to_tuple()] = predicate_entry.value

    def generate_normalize_robustness(self) -> None:
        for predicate_name in self.predicate_names:
            robustness = self.data.predicates.loc[
                :, (predicate_name, DATA_LOADER_REFERENCE_NAMES[-1])
            ].values
            self.data.loc[:, ("predicates", predicate_name, DATA_LOADER_OUTPUT_NAME)] = normalize(
                robustness
            )
