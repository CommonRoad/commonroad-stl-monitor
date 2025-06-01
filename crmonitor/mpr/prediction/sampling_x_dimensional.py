"""
Module that implements the low-level sampling for
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from typing import Generic, TypeVar

import numpy as np
from scipy.stats import norm, uniform

# TODO: The original implementation had this hard coded value for monte-carlo size.
_DEFAULT_MONTE_CARLO_SIZE = 1


@dataclass
class Sampling1DParams:
    """Parameters for 1D sampling operations."""

    min_val: float
    max_val: float

    # TODO: The size option does not really belong here.
    # It is specific to this dimension, but it is dependent on the sampling simulation.
    # This creates a higher then necessary coupling with the specific sampling implementations,
    # because they must carry over this value.
    # It would be better, if this option is instead provided directly from SamplingXD.
    size: int
    eps: float = 0.0015

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.max_val <= self.min_val:
            raise ValueError(
                f"max_val ({self.max_val}) must be greater than min_val ({self.min_val})"
            )

        if not (0 < self.eps < 0.5):
            raise ValueError(f"eps ({self.eps}) must be between 0 and 0.5")

        if not isinstance(self.size, int):
            raise TypeError(f"size must be an integer, got {type(self.size).__name__}")

        if self.size < 1:
            raise ValueError(f"size ({self.size}) must be greater than or equal to 1")


class Sampling1D(ABC):
    """Base class for each sampling implementation which samples in one dimension."""

    @classmethod
    @abstractmethod
    def grid_sampling(cls, params: Sampling1DParams) -> np.ndarray:
        pass

    @classmethod
    @abstractmethod
    def monte_carlo_sampling(cls, params: Sampling1DParams) -> float:
        pass


class UniformSampling1D(Sampling1D):
    @classmethod
    def grid_sampling(cls, params: Sampling1DParams) -> np.ndarray:
        if params.size == 1:
            return np.array([(params.min_val + params.max_val) / 2])
        else:  # size > 1
            return np.linspace(params.min_val, params.max_val, params.size)

    @classmethod
    def monte_carlo_sampling(cls, params: Sampling1DParams) -> float:
        return uniform.rvs(
            loc=params.min_val,
            scale=(params.max_val - params.min_val),
            size=_DEFAULT_MONTE_CARLO_SIZE,
        )


class NormalSampling1D(Sampling1D):
    @classmethod
    def grid_sampling(cls, params: Sampling1DParams) -> np.ndarray:
        cdfs = np.linspace(params.eps, 1 - params.eps, params.size)
        x = norm.ppf(
            cdfs,
            loc=(params.min_val + params.max_val) / 2,
            scale=1 / -norm.ppf(params.eps) * (params.max_val - params.min_val) / 2,
        )
        # Handle small overflow
        if x[0] < params.min_val:
            x[0] = params.min_val
        if x[-1] > params.max_val:
            x[-1] = params.max_val
        return x

    @classmethod
    def monte_carlo_sampling(cls, params: Sampling1DParams) -> float:
        return norm.rvs(
            loc=(params.min_val + params.max_val) / 2,
            scale=1 / -norm.ppf(params.eps) * (params.max_val - params.min_val) / 2,
            size=_DEFAULT_MONTE_CARLO_SIZE,
        )


class SamplingOrder(IntEnum):
    """
    Kinematic sampling orders for each sampling dimension.

    Represents the hierarchical order of kinematic quantities, where each successive
    order is the time derivative of the previous one. The integer values correspond
    to the derivative order with respect to time.
    """

    POSITION = 0
    VELOCITY = 1
    ACCELERATION = 2

    def __str__(self) -> str:
        match self:
            case SamplingOrder.POSITION:
                return "position"
            case SamplingOrder.VELOCITY:
                return "velocity"
            case SamplingOrder.ACCELERATION:
                return "acceleration"


class SamplingDimension(Enum):
    """
    Specify dimension in which

    Usually used in combination with `SamplingOrder` to denote the
    """

    LONG = "long"
    LAT = "lat"

    def __str__(self) -> str:
        return self.value


class SamplingDistribution(Enum):
    UNIFORM = auto()
    NORMAL = auto()


class SamplingSimulation(Enum):
    GRID = auto()
    MONTE_CARLO = auto()


@dataclass(kw_only=True)
class SamplingXDParams:
    sample_number: int = 1000
    sampling_dimensions: dict[SamplingDimension, Sampling1DParams]


_T = TypeVar("_T")


XDimensionalData = dict[SamplingDimension, dict[SamplingOrder, _T]]


class XDimensionalIterator(Generic[_T]):
    """
    Helper class to make it easier to iterate over `XDimensionalData` by unpacking it into individual tuples of dimension, order and value.
    """

    def __init__(self, dimension_data: XDimensionalData[_T]) -> None:
        self._dimension_data = dimension_data

    def __iter__(self) -> Iterator[tuple[SamplingDimension, SamplingOrder, _T]]:
        for dimension, inner_dict in self._dimension_data.items():
            for order, item in inner_dict.items():
                yield (dimension, order, item)


class SamplingXD:
    def __init__(self, distribution: SamplingDistribution, simulation: SamplingSimulation) -> None:
        self._distribution = distribution
        self._simulation = simulation

        if distribution == SamplingDistribution.UNIFORM:
            sampler1d = UniformSampling1D
        else:
            sampler1d = NormalSampling1D

        if simulation == SamplingSimulation.GRID:
            self._sample_function = sampler1d.grid_sampling
        else:  # 'monte-carlo'
            self._sample_function = sampler1d.monte_carlo_sampling

    def sample(self, params: SamplingXDParams) -> XDimensionalData[np.ndarray]:
        samples = defaultdict(dict)
        for dimension, order, sampling_1d_params in XDimensionalIterator(
            params.sampling_dimensions
        ):
            if self._simulation == SamplingSimulation.GRID:
                samples[dimension][order] = self._sample_function(sampling_1d_params)
            else:  # 'monte-carlo'
                result = []
                for _ in range(params.sample_number):
                    result.extend(self._sample_function(sampling_1d_params))
                samples[dimension][order] = np.array(result)
        return samples
