"""Tumbling windows over two streams (features and scores): event counts or calendar months."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

# ============= Helpers =============


def month_id(event_time_ms: int) -> str:
    """Return the UTC calendar month `YYYY-MM` of an event time."""
    return datetime.fromtimestamp(event_time_ms / 1000, tz=UTC).strftime("%Y-%m")


@dataclass
class ClosedWindow:
    """A completed window with its feature matrix and champion scores."""

    window_id: str
    features: np.ndarray
    scores: np.ndarray
    end_event_time_ms: int


# ============= Count windows =============


@dataclass
class CountWindows:
    """
    Tumbling windows of `size` events per stream (CICIDS2017, SPEC 5.1).

    The k-th window holds flows k*size .. (k+1)*size-1 from the flow stream and the same index
    range from the prediction stream; it closes once both streams have filled it.
    """

    size: int
    _features: list[list[float]] = field(default_factory=list)
    _scores: list[float] = field(default_factory=list)
    _times: list[int] = field(default_factory=list)
    index: int = 0

    def add_flows(self, rows: list[tuple[int, list[float]]]) -> None:
        """Append (event_time_ms, feature vector) rows."""
        for event_time, vector in rows:
            self._features.append(vector)
            self._times.append(event_time)

    def add_scores(self, rows: list[tuple[int, float]]) -> None:
        """Append (event_time_ms, score) rows."""
        self._scores.extend(score for _, score in rows)

    def ready(self) -> list[ClosedWindow]:
        """Pop every window that both streams have filled."""
        closed = []
        while len(self._features) >= self.size and len(self._scores) >= self.size:
            closed.append(
                ClosedWindow(
                    window_id=f"window_{self.index:05d}",
                    features=np.asarray(self._features[: self.size], dtype=np.float64),
                    scores=np.asarray(self._scores[: self.size], dtype=np.float64),
                    end_event_time_ms=int(max(self._times[: self.size])),
                )
            )
            del self._features[: self.size], self._scores[: self.size], self._times[: self.size]
            self.index += 1
        return closed


# ============= Month windows =============


@dataclass
class MonthWindows:
    """
    Calendar-month windows (Kyoto, SPEC 5.1).

    Month M closes once each stream has seen `close_after` events from later months; this
    absorbs the interleaving of six partitions. Events arriving for a closed month are counted
    as late and excluded.
    """

    close_after: int = 1000
    _features: dict[str, list[list[float]]] = field(default_factory=lambda: defaultdict(list))
    _scores: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _later_flows: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _later_scores: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    closed: set[str] = field(default_factory=set)
    late_events: int = 0

    def _count_later(self, month: str, counter: dict[str, int]) -> None:
        for open_month in list(self._features.keys() | self._scores.keys()):
            if open_month < month:
                counter[open_month] += 1

    def add_flows(self, rows: list[tuple[int, list[float]]]) -> None:
        """Append (event_time_ms, feature vector) rows."""
        for event_time, vector in rows:
            month = month_id(event_time)
            if month in self.closed:
                self.late_events += 1
                continue
            self._features[month].append(vector)
            self._count_later(month, self._later_flows)

    def add_scores(self, rows: list[tuple[int, float]]) -> None:
        """Append (event_time_ms, score) rows."""
        for event_time, score in rows:
            month = month_id(event_time)
            if month in self.closed:
                self.late_events += 1
                continue
            self._scores[month].append(score)
            self._count_later(month, self._later_scores)

    def ready(self) -> list[ClosedWindow]:
        """Pop every month that both streams have moved past."""
        closed = []
        for month in sorted(self._features.keys() | self._scores.keys()):
            if (
                self._later_flows[month] >= self.close_after
                and self._later_scores[month] >= self.close_after
            ):
                features = self._features.pop(month, [])
                scores = self._scores.pop(month, [])
                self.closed.add(month)
                if features and scores:
                    year, mon = (int(part) for part in month.split("-"))
                    end = datetime(year + mon // 12, mon % 12 + 1, 1, tzinfo=UTC)
                    closed.append(
                        ClosedWindow(
                            window_id=month,
                            features=np.asarray(features, dtype=np.float64),
                            scores=np.asarray(scores, dtype=np.float64),
                            end_event_time_ms=int(end.timestamp() * 1000) - 1,
                        )
                    )
        return closed
