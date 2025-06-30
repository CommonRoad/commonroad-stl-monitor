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
from typing_extensions import Self

from crmonitor.mpr.prediction.error import SamplingError

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

    LON = "long"
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
    sampling_dimensions: dict[SamplingDimension, dict[SamplingOrder, Sampling1DParams]]


_T = TypeVar("_T")


class DimensionData(Generic[_T]):
    def __init__(
        self,
        *,
        position: _T | None = None,
        velocity: _T | None = None,
        acceleration: _T | None = None,
    ) -> None:
        self._position = position
        self._velocity = velocity
        self._acceleration = acceleration

    @property
    def position(self) -> _T:
        if self._position is None:
            raise SamplingError(
                "Tried to access order 'position' of state, but 'position' is not available for the dimension!"
            )
        return self._position

    @property
    def velocity(self) -> _T:
        if self._velocity is None:
            raise SamplingError(
                "Tried to access order 'velocity' of state, but 'velocity' is not available for the dimension!"
            )
        return self._velocity

    @property
    def acceleration(self) -> _T:
        if self._acceleration is None:
            raise SamplingError(
                "Tried to access order 'acceleration' of state, but 'acceleration' is not available for the dimension!"
            )
        return self._acceleration

    @classmethod
    def from_dict(cls, dict_: dict[SamplingOrder, _T]) -> Self:
        return cls(
            position=dict_.get(SamplingOrder.POSITION),
            velocity=dict_.get(SamplingOrder.VELOCITY),
            acceleration=dict_.get(SamplingOrder.ACCELERATION),
        )

    def has_order(self, order: SamplingOrder) -> bool:
        return self.get_order(order) is not None

    def get_order(self, order: SamplingOrder) -> _T | None:
        if order == SamplingOrder.POSITION:
            return self._position
        elif order == SamplingOrder.VELOCITY:
            return self._velocity
        else:
            return self._acceleration


@dataclass
class LonLatData(Generic[_T]):
    lon: DimensionData[_T]
    lat: DimensionData[_T]

    @classmethod
    def from_dict(cls, dict_: dict[SamplingDimension, dict[SamplingOrder, _T]]) -> Self:
        return cls(
            lon=DimensionData.from_dict(dict_[SamplingDimension.LON]),
            lat=DimensionData.from_dict(dict_[SamplingDimension.LAT]),
        )

    def get_dimension(self, dimension: SamplingDimension) -> DimensionData[_T]:
        if dimension == SamplingDimension.LON:
            return self.lon
        else:
            return self.lat


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


@dataclass
class SamplingXDConfig:
    distribution: SamplingDistribution = SamplingDistribution.UNIFORM
    simulation: SamplingSimulation = SamplingSimulation.MONTE_CARLO


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
