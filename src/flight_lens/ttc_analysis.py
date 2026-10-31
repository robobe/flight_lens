"""Production-faithful optical TTC replay for FlightLens investigation."""
from collections import Counter, deque
from dataclasses import dataclass
from math import exp, isfinite, pi, sqrt, tan


EPSILON = 1e-12
DEFAULT_TTC_WINDOW_S = 0.5
DEFAULT_TTC_MIN_SPAN_S = 0.2
DEFAULT_TTC_MIN_GROWTH_PX = 1.0
DEFAULT_TTC_MAX_RELATIVE_RESIDUAL = 0.15
DEFAULT_TTC_MIN_SAMPLES = 8
DEFAULT_MAX_IMAGE_AGE_S = 0.1

RESET_REASONS = frozenset({'recovery', 'clipped_target', 'stale_image', 'no_target', 'not_observed'})


@dataclass(frozen=True)
class TtcParameters:
    window_s: float = DEFAULT_TTC_WINDOW_S
    min_span_s: float = DEFAULT_TTC_MIN_SPAN_S
    min_growth_px: float = DEFAULT_TTC_MIN_GROWTH_PX
    max_relative_residual: float = DEFAULT_TTC_MAX_RELATIVE_RESIDUAL
    min_samples: int = DEFAULT_TTC_MIN_SAMPLES
    max_image_age_s: float = DEFAULT_MAX_IMAGE_AGE_S

    def validate(self):
        values = (self.window_s, self.min_span_s, self.min_growth_px,
                  self.max_relative_residual, self.max_image_age_s)
        if not all(isfinite(value) and value > 0 for value in values):
            raise ValueError('TTC parameters must be finite and positive')
        if self.min_span_s > self.window_s:
            raise ValueError('minimum TTC span cannot exceed history window')
        if type(self.min_samples) is not int or self.min_samples < 2:
            raise ValueError('minimum TTC samples must be an integer >= 2')
        return self


@dataclass(frozen=True)
class TtcPoint:
    time_s: float
    value_s: float | None
    valid: bool
    reason: str
    relative_residual: float | None = None


@dataclass(frozen=True)
class TtcReplay:
    points: tuple[TtcPoint, ...]
    counts: dict[str, int]


@dataclass(frozen=True)
class TtcComparison:
    hit_time_s: float | None
    oracle: tuple[tuple[float, float], ...]
    sample_count: int
    mean_absolute_error_s: float | None
    root_mean_square_error_s: float | None
    bias_s: float | None
    coverage: float


def hit_time(episode):
    candidates = [event.time_s for event in episode.events
                  if event.time_s is not None and event.reason == 'target_hit']
    return min(candidates) if candidates else None


def compare_to_hit_time(episode, points):
    """Compare qualified replay TTC with retrospective target-hit TTC."""
    end_time = hit_time(episode)
    samples = [point for point in points if point.time_s <= end_time] if end_time is not None else []
    oracle = tuple((point.time_s, end_time - point.time_s) for point in samples)
    valid = [(point, value) for point, (_, value) in zip(samples, oracle) if point.valid]
    if end_time is None:
        return TtcComparison(None, (), 0, None, None, None, 0.)
    errors = [point.value_s - value for point, (_, value) in zip(samples, oracle) if point.valid]
    if not errors:
        return TtcComparison(end_time, oracle, 0, None, None, None, 0.)
    absolute = [abs(error) for error in errors]
    return TtcComparison(end_time, oracle, len(errors), sum(absolute) / len(errors),
                         sqrt(sum(error * error for error in errors) / len(errors)),
                         sum(errors) / len(errors), len(errors) / max(1, len(samples)))


def butterworth_filter(points, cutoff_hz=1.0, order=2):
    """Apply an offline, zero-phase Butterworth low-pass to qualified points.

    Invalid samples remain gaps and are never bridged by the filter.
    """
    if not isfinite(cutoff_hz) or cutoff_hz <= 0 or order != 2:
        raise ValueError('Butterworth cutoff must be positive and order must be 2')
    filtered = [None] * len(points)
    index = 0
    while index < len(points):
        if not points[index].valid:
            index += 1
            continue
        end = index
        while end < len(points) and points[end].valid:
            end += 1
        segment = points[index:end]
        if len(segment) >= 3:
            times = [point.time_s for point in segment]
            values = [point.value_s for point in segment]
            dt_values = [b - a for a, b in zip(times, times[1:]) if b > a]
            if dt_values:
                sample_rate = 1. / (sum(dt_values) / len(dt_values))
                normalized_cutoff = min(cutoff_hz, sample_rate * .49)
                ita = 1. / tan(pi * normalized_cutoff / sample_rate)
                root_two = 2. ** .5
                scale = 1. / (1. + root_two * ita + ita * ita)
                b0, b1, b2 = scale, 2. * scale, scale
                a1 = 2. * (1. - ita * ita) * scale
                a2 = (1. - root_two * ita + ita * ita) * scale

                def apply(values_to_filter):
                    output = []
                    x1 = x2 = y1 = y2 = values_to_filter[0]
                    for value in values_to_filter:
                        current = b0 * value + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
                        output.append(current)
                        x2, x1 = x1, value
                        y2, y1 = y1, current
                    return output

                forward = apply(values)
                backward = list(reversed(apply(list(reversed(forward)))))
                for offset, value in enumerate(backward, index):
                    filtered[offset] = value
        index = end
    return tuple(filtered)


def low_pass_filter(points, alpha=.25):
    """Apply a causal exponential low-pass to qualified points without bridging gaps."""
    if not isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError('low-pass alpha must be in (0, 1]')
    filtered = [None] * len(points)
    previous = None
    for index, point in enumerate(points):
        if not point.valid:
            previous = None
            continue
        previous = point.value_s if previous is None else previous + alpha * (point.value_s - previous)
        filtered[index] = previous
    return tuple(filtered)


def causal_low_pass_filter(points, time_constant_s=.15):
    """Apply a time-aware causal EMA without bridging invalid samples."""
    if not isfinite(time_constant_s) or time_constant_s <= 0:
        raise ValueError('low-pass time constant must be positive')
    filtered = [None] * len(points)
    previous = None
    previous_time = None
    for index, point in enumerate(points):
        value = point.value_s
        if not point.valid or value is None or not isfinite(value):
            previous = None
            previous_time = None
            continue
        if previous is None or previous_time is None or point.time_s <= previous_time:
            previous = value
        else:
            alpha = 1. - exp(-(point.time_s - previous_time) / time_constant_s)
            previous += alpha * (value - previous)
        previous_time = point.time_s
        filtered[index] = previous
    return tuple(filtered)


def apparent_size(bbox):
    if bbox is None or len(bbox) != 4:
        return None
    width, height = bbox[2], bbox[3]
    if not all(isfinite(float(value)) for value in (width, height)) or width <= 0 or height <= 0:
        return None
    return sqrt(width) * sqrt(height)


def recalculate(samples, parameters=None):
    """Recalculate reciprocal-apparent-size TTC over tracking samples."""
    parameters = (parameters or TtcParameters()).validate()
    history = deque()
    points = []
    for sample in samples:
        reason = sample.ttc_reason or 'insufficient_history'
        if reason in RESET_REASONS:
            history.clear()
            points.append(TtcPoint(sample.time_s, None, False, reason))
            continue
        size = apparent_size(sample.bbox)
        if size is None or sample.bbox_age_s is not None and sample.bbox_age_s > parameters.max_image_age_s:
            history.clear()
            reason = 'stale' if size is not None else 'not_observed'
            points.append(TtcPoint(sample.time_s, None, False, reason))
            continue
        history.append((sample.time_s, size))
        while history and sample.time_s - history[0][0] > parameters.window_s + EPSILON:
            history.popleft()
        value = None
        residual_value = None
        reason = 'insufficient_history'
        if len(history) >= parameters.min_samples:
            span = sample.time_s - history[0][0]
            if span >= parameters.min_span_s:
                times = [time_s - sample.time_s for time_s, _ in history]
                inverse_sizes = [1 / size for _, size in history]
                mean_t = sum(times) / len(times)
                mean_y = sum(inverse_sizes) / len(inverse_sizes)
                variance = sum((time_s - mean_t) ** 2 for time_s in times)
                if variance <= EPSILON or mean_y <= 0:
                    reason = 'unresolved_growth'
                else:
                    slope = sum((time_s - mean_t) * (y - mean_y)
                                for time_s, y in zip(times, inverse_sizes)) / variance
                    intercept = mean_y - slope * mean_t
                    if slope >= -EPSILON or intercept <= 0:
                        reason = 'not_closing'
                    elif history[-1][1] - history[0][1] < parameters.min_growth_px:
                        reason = 'unresolved_growth'
                    else:
                        residual = sqrt(sum((y - intercept - slope * time_s) ** 2
                                            for time_s, y in zip(times, inverse_sizes)) / len(times))
                        residual_value = residual / mean_y
                        if residual_value > parameters.max_relative_residual:
                            reason = 'noisy_growth'
                        else:
                            value = -(intercept + slope * 0.) / slope
                            reason = 'qualified' if isfinite(value) and value > 0 else 'model_elapsed'
                            if reason != 'qualified':
                                value = None
        points.append(TtcPoint(sample.time_s, value, reason == 'qualified', reason, residual_value))
    return TtcReplay(tuple(points), dict(Counter(point.reason for point in points)))
