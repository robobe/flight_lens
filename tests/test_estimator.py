import numpy as np

from flight_lens.estimator import estimate_attitude


def test_kalman_attitude_is_sampled_at_100_hz_and_bridges_gaps():
    times, values = estimate_attitude(((0., (0., 0., 0.)), (.2, (1., 2., 3.))))

    np.testing.assert_allclose(np.diff(times), .01)
    assert len(times) == 21
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values[0], (0., 0., 0.), atol=.01)


def test_kalman_attitude_unwraps_yaw_at_180_degrees():
    times, values = estimate_attitude(((0., (0., 0., 179.)),
                                       (.1, (0., 0., -179.))))

    assert times[-1] == .1
    assert values[-1, 2] > 180.
