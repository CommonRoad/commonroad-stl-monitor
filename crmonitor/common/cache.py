from abc import ABC, abstractmethod
from typing import Generic, TypeVar

_K = TypeVar(name="_K")
_V = TypeVar("_V")


class TimeStepCache(Generic[_K, _V], ABC):
    @abstractmethod
    def set_at_time_step(self, time_step: int, key: _K, value: _V) -> None: ...

    @abstractmethod
    def get_at_time_step(self, time_step: int, key: _K) -> _V | None: ...


class BasicTimeStepCache(TimeStepCache[_K, _V]):
    def __init__(self) -> None:
        self._cache = {}


class LinearTimeStepCache(TimeStepCache[_K, _V]):
    def __init__(self) -> None:
        self._cache = {}
        self._last_time_step = None

    def set_at_time_step(self, time_step: int, key: _K, value: _V) -> None:
        self._invalidate_if_time_step_advanced(time_step)

        self._cache[key] = value

    def get_at_time_step(self, time_step: int, key: _K) -> _V | None:
        self._invalidate_if_time_step_advanced(time_step)

        return self._cache.get(key)

    def _invalidate_if_time_step_advanced(self, time_step: int) -> None:
        if self._last_time_step is None:
            return

        if time_step > self._last_time_step:
            del self._cache
            self._cache = {}
            self._last_time_step = time_step
        elif time_step < self._last_time_step:
            raise ValueError()
