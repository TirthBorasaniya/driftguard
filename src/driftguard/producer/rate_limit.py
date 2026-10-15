"""Pacing for the replay producer: token bucket and event-time scaling (SPEC 3.4)."""

import time
from collections.abc import Callable

# ============= Constants =============

TOKEN_EPSILON = 1e-9

# ============= Pacers =============


class TokenBucket:
    """
    Token bucket limiting emission to `rate` events per second with a burst allowance.

    Parameters
    ----------
    rate : float
        Sustained events per second.
    burst : float
        Bucket capacity in events.
    clock : callable
        Monotonic time source in seconds.
    sleep : callable
        Sleep function, injectable for tests.
    """

    def __init__(
        self,
        rate: float,
        burst: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if rate <= 0 or burst <= 0:
            raise ValueError("rate and burst must be positive")
        self.rate = float(rate)
        self.capacity = float(burst)
        self.tokens = float(burst)
        self._clock = clock
        self._sleep = sleep
        self._last = clock()

    def _refill(self) -> None:
        now = self._clock()
        self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.rate)
        self._last = now

    def acquire(self, n: float = 1.0) -> float:
        """
        Block until `n` tokens are available and consume them.

        Returns
        -------
        waited : float
            Seconds spent sleeping.
        """
        waited = 0.0
        self._refill()
        # tolerance stops float absorption from looping on a deficit of a few ulps
        while self.tokens < n - TOKEN_EPSILON:
            delay = (n - self.tokens) / self.rate
            self._sleep(delay)
            waited += delay
            self._refill()
        self.tokens = max(0.0, self.tokens - n)
        return waited


class RealtimePacer:
    """
    Paces emission so that event-time gaps are replayed divided by `factor`.

    Parameters
    ----------
    factor : float
        Speed-up factor; 60 replays one minute of event time per wall-clock second.
    clock, sleep : callable
        Injectable time functions.
    """

    def __init__(
        self,
        factor: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if factor <= 0:
            raise ValueError("factor must be positive")
        self.factor = float(factor)
        self._clock = clock
        self._sleep = sleep
        self._origin_wall: float | None = None
        self._origin_event_ms: int | None = None

    def wait_until(self, event_time_ms: int) -> float:
        """Sleep until the wall-clock time matching `event_time_ms`; return seconds slept."""
        if self._origin_wall is None:
            self._origin_wall, self._origin_event_ms = self._clock(), event_time_ms
            return 0.0
        target = self._origin_wall + (event_time_ms - self._origin_event_ms) / 1000.0 / self.factor
        delay = target - self._clock()
        if delay > 0:
            self._sleep(delay)
            return delay
        return 0.0
