"""Immutable flight data; no Qt, filesystem or controller dependencies."""
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

GAP_FACTOR = 5
CURSOR_TOLERANCE_S = .1


@dataclass(frozen=True)
class Series:
    times: np.ndarray
    values: np.ndarray

    def nearest(self, time_s, tolerance_s=CURSOR_TOLERANCE_S):
        index = int(np.searchsorted(self.times, time_s))
        candidates = [i for i in (index - 1, index) if 0 <= i < len(self.times)]
        if not candidates:
            return None
        best = min(candidates, key=lambda i: abs(self.times[i] - time_s))
        return best if abs(self.times[best] - time_s) <= tolerance_s else None

    def plot(self, column):
        """Insert NaNs so plotting never bridges long missing intervals."""
        if len(self.times) < 2:
            return self.times, self.values[:, column]
        dt = np.diff(self.times)
        gaps = np.flatnonzero(dt > GAP_FACTOR * np.median(dt)) + 1
        return np.insert(self.times, gaps, np.nan), np.insert(self.values[:, column], gaps, np.nan)


def series(rows, columns=1):
    # Last sample wins at a repeated simulation time (for example, a stopped update).
    samples = {}
    for t, values in rows:
        try:
            sample = tuple(float(v) for v in values)
            if len(sample) == columns and np.isfinite(float(t)) and all(np.isfinite(v) for v in sample):
                samples[float(t)] = sample
        except (ValueError, TypeError, OverflowError):
            continue
    times = np.array(sorted(samples), dtype=float)
    values = np.array([samples[t] for t in times], dtype=float).reshape((-1, columns))
    times.setflags(write=False)
    values.setflags(write=False)
    return Series(times, values)


@dataclass(frozen=True)
class FrameRef:
    topic: str
    log_time_ns: int
    record_sequence: int
    identity: tuple
    width: int
    height: int
    simulation_time_s: float | None = None


@dataclass(frozen=True)
class Frame:
    width: int
    height: int
    rgb: bytes


@dataclass(frozen=True)
class Event:
    time_s: float | None
    end_s: float | None
    arrival_ns: int
    severity: str
    component: str
    reason: str
    phase: str
    details: Mapping
    frame: FrameRef | None = None
    bbox: tuple | None = None


@dataclass(frozen=True)
class Summary:
    outcome: str
    duration_s: float
    max_altitude_m: float | None
    y_range_m: tuple | None
    first_fault: str | None
    warnings: int
    errors: int
    integrity: str


@dataclass(frozen=True)
class Episode:
    key: tuple[str, str]
    trajectory: Series
    trajectory_origin: str
    velocity: Series
    barometer_velocity: Series
    attitude: Series
    feedback_attitude: Series
    msp_attitude: Series
    msp_attitude_estimate: Series
    thrust: Series
    tracking_time_s: float | None
    barometer: Series
    goal: Series
    margins: Series
    margins_origin: str
    events: tuple[Event, ...]
    summary: Summary
    notices: tuple[str, ...]
    context_times: tuple[float, ...]
    contexts: tuple[Mapping, ...]
    frames: tuple[FrameRef, ...]
    boxes: Mapping

    def context_at(self, time_s):
        index = bisect_left(self.context_times, time_s)
        candidates = [i for i in (index - 1, index) if 0 <= i < len(self.context_times)]
        if not candidates:
            return {}
        best = min(candidates, key=lambda i: abs(self.context_times[i] - time_s))
        return self.contexts[best] if abs(self.context_times[best] - time_s) <= CURSOR_TOLERANCE_S else {}


@dataclass(frozen=True)
class FlightData:
    path: Path
    size: int
    modified_ns: int
    episodes: tuple[Episode, ...]
    default_episode: int
    notices: tuple[str, ...]
    camera_image_count: int | None = None
