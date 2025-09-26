import math
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from typing_extensions import override


class IRobustnessScaler(metaclass=ABCMeta):
    @property
    @abstractmethod
    def max(self) -> float:
        """
        The maximum robustness value. Should be used for clipping and as default value, instead of hardcoding.
        """
        ...

    @property
    @abstractmethod
    def min(self) -> float:
        """
        The minimal robustness value. Should be used for clipping and as default value, instead of hardcoding.
        """
        ...

    @abstractmethod
    def scale_speed(self, x: float) -> float:
        pass

    @abstractmethod
    def scale_acc(self, x: float) -> float:
        pass

    @abstractmethod
    def scale_lon_dist(self, x: float) -> float:
        pass

    @abstractmethod
    def scale_lat_dist(self, x: float) -> float:
        pass

    @abstractmethod
    def scale_angle(self, x: float) -> float:
        pass


@dataclass(frozen=True)
class RobustnessScalingConstants:
    MAX_LONG_DIST: float = 200.0
    MAX_LAT_DIST: float = 20.0
    MAX_SPEED: float = 250.0 / 3.6
    MAX_ACC: float = 10.5
    MAX_ANGLE: float = math.pi


class RobustnessScaler(IRobustnessScaler):
    def __init__(
        self,
        scale: bool = True,
        scale_constants: Optional[RobustnessScalingConstants] = None,
    ):
        self._scale_constants = scale_constants or RobustnessScalingConstants()
        self.scale = scale

    def _scale(self, x: float, max_value: float) -> float:
        return np.clip(x / max_value, self.min, self.max) if self.scale else x

    @property
    @override
    def max(self) -> float:
        return 1.0 if self.scale else float("inf")

    @property
    @override
    def min(self) -> float:
        return -1.0 if self.scale else float("-inf")

    @override
    def scale_speed(self, x: float) -> float:
        return self._scale(x, self._scale_constants.MAX_SPEED)

    @override
    def scale_acc(self, x: float) -> float:
        return self._scale(x, self._scale_constants.MAX_ACC)

    @override
    def scale_lon_dist(self, x: float) -> float:
        return self._scale(x, self._scale_constants.MAX_LONG_DIST)

    @override
    def scale_lat_dist(self, x: float) -> float:
        return self._scale(x, self._scale_constants.MAX_LAT_DIST)

    @override
    def scale_angle(self, x: float) -> float:
        return self._scale(x, self._scale_constants.MAX_ANGLE)
