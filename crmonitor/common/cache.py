from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from typing_extensions import override

_K = TypeVar(name="_K")
_V = TypeVar("_V")


class TimeStepCache(Generic[_K, _V], ABC):
    @abstractmethod
    def set_at_time_step(self, time_step: int, key: _K, value: _V) -> None: ...

    @abstractmethod
    def get_at_time_step(self, time_step: int, key: _K) -> _V | None: ...

    @abstractmethod
    def invalidate(self) -> None: ...


class BasicTimeStepCache(TimeStepCache[_K, _V]):
    def __init__(self) -> None:
        self._cache = {}

    @override
    def set_at_time_step(self, time_step: int, key: _K, value: _V) -> None:
        self._cache[(time_step, key)] = value

    @override
    def get_at_time_step(self, time_step: int, key: _K) -> _V | None:
        return self._cache.get((time_step, key))

    def invalidate(self) -> None:
        del self._cache
        self._cache = {}


class LinearTimeStepCache(TimeStepCache[_K, _V]):
    """
    Special time step cache, which assumes linear ordering of time steps.

    This cache is useful for online evaluations, where time steps are strictly increasing.
    For offline evaluations this is currently not the case, so this caching approach is not beneficial.
    """

    def __init__(self) -> None:
        self._cache = {}
        self._last_time_step = None

    @override
    def set_at_time_step(self, time_step: int, key: _K, value: _V) -> None:
        self._invalidate_if_time_step_advanced(time_step)

        self._cache[key] = value

    @override
    def get_at_time_step(self, time_step: int, key: _K) -> _V | None:
        self._invalidate_if_time_step_advanced(time_step)

        return self._cache.get(key)

    def _invalidate_if_time_step_advanced(self, time_step: int) -> None:
        if self._last_time_step is None:
            return

        if time_step != self._last_time_step:
            self.invalidate()

    @override
    def invalidate(self) -> None:
        del self._cache
        self._cache = {}
        self._last_time_step = None
