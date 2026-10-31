from types import SimpleNamespace

import pytest

from flight_lens.ttc_analysis import (TtcParameters, TtcPoint, butterworth_filter,
                                      causal_low_pass_filter, compare_to_hit_time,
                                      low_pass_filter, recalculate)


def samples(sizes, *, start=0.0, dt=.04, reason='qualified'):
    return tuple(SimpleNamespace(time_s=start + index * dt, bbox=(100., 100., size, size),
                                bbox_age_s=0., ttc_reason=reason)
                 for index, size in enumerate(sizes))


def test_recalculate_qualifies_consistent_growth():
    result = recalculate(samples([10., 11., 12., 13., 14., 16., 18., 20.]))
    assert result.points[-1].valid
    assert result.points[-1].reason == 'qualified'


def test_recalculate_reports_non_closing_with_bbox_present():
    result = recalculate(samples([12.] * 10))
    assert all(point.value_s is None for point in result.points)
    assert result.counts['not_closing'] > 0 or result.counts['unresolved_growth'] > 0


def test_recalculate_resets_on_recovery():
    before = samples([10., 11., 12., 13., 14., 16., 18., 20.])
    recovery = SimpleNamespace(time_s=.4, bbox=(100., 100., 20., 20.), bbox_age_s=0., ttc_reason='recovery')
    after = samples([21., 22., 24.], start=.44)
    result = recalculate((*before, recovery, *after))
    assert result.points[8].reason == 'recovery'
    assert all(not point.valid for point in result.points[9:])


def test_parameters_validate_span_and_sample_count():
    with pytest.raises(ValueError):
        TtcParameters(window_s=.1, min_span_s=.2).validate()
    with pytest.raises(ValueError):
        TtcParameters(min_samples=1).validate()


def test_butterworth_filter_preserves_invalid_gaps():
    result = recalculate(samples([10., 11., 12., 13., 14., 16., 18., 20., 22., 24.]))
    filtered = butterworth_filter(result.points, cutoff_hz=1.)
    assert filtered[0] is None
    assert any(value is not None for value in filtered)


def test_butterworth_filter_rejects_invalid_cutoff():
    with pytest.raises(ValueError):
        butterworth_filter((), cutoff_hz=0.)


def test_low_pass_filter_resets_across_invalid_samples():
    points = (TtcPoint(0., 1., True, 'qualified'), TtcPoint(1., 3., True, 'qualified'),
              TtcPoint(2., None, False, 'recovery'), TtcPoint(3., 5., True, 'qualified'))
    filtered = low_pass_filter(points, alpha=.5)
    assert filtered[0] == 1.
    assert filtered[1] == 2.
    assert filtered[2] is None
    assert filtered[3] == 5.


def test_low_pass_filter_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        low_pass_filter((), alpha=0.)


def test_causal_low_pass_uses_elapsed_time_and_resets():
    points = (TtcPoint(0., 1., True, 'qualified'),
              TtcPoint(.1, 3., True, 'qualified'),
              TtcPoint(.5, 5., True, 'qualified'),
              TtcPoint(.6, None, False, 'recovery'),
              TtcPoint(.7, 7., True, 'qualified'))
    filtered = causal_low_pass_filter(points, time_constant_s=.1)
    assert filtered[0] == 1.
    assert filtered[1] == pytest.approx(1. + (1. - 2.718281828459045 ** -1) * 2.)
    expected = (1. + (1. - 2.718281828459045 ** -1) * 2.)
    expected += (1. - 2.718281828459045 ** -4) * (5. - expected)
    assert filtered[2] == pytest.approx(expected)
    assert filtered[3] is None
    assert filtered[4] == 7.


def test_causal_low_pass_rejects_invalid_time_constant():
    with pytest.raises(ValueError):
        causal_low_pass_filter((), time_constant_s=0.)


def test_hit_time_comparison_reports_oracle_metrics():
    replay = SimpleNamespace(points=tuple(TtcPoint(index * .1, 1.5 - index * .05, True, 'qualified')
                                          for index in range(10)))
    episode = SimpleNamespace(events=(SimpleNamespace(time_s=2., reason='target_hit'),))
    comparison = compare_to_hit_time(episode, replay.points)
    assert comparison.hit_time_s == 2.
    assert comparison.sample_count > 0
    assert comparison.mean_absolute_error_s is not None
    assert comparison.coverage > 0.


def test_hit_time_comparison_is_unavailable_without_target_hit():
    replay = recalculate(samples([10., 11., 12., 13., 14., 16., 18., 20.]))
    comparison = compare_to_hit_time(SimpleNamespace(events=()), replay.points)
    assert comparison.hit_time_s is None
    assert comparison.oracle == ()
    assert comparison.sample_count == 0
