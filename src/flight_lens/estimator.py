"""Small offline attitude estimator for sparse Betaflight MSP samples."""
import numpy as np

ESTIMATOR_HZ = 100.0
MEASUREMENT_NOISE_DEG = 1.5
ANGULAR_ACCELERATION_NOISE_DEG_S2 = 180.0


def _step(state, covariance, dt):
    transition = np.array(((1., dt), (0., 1.)))
    dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
    process = ANGULAR_ACCELERATION_NOISE_DEG_S2 ** 2 * np.array(((dt4 / 4., dt3 / 2.), (dt3 / 2., dt2)))
    return transition @ state, transition @ covariance @ transition.T + process


def estimate_attitude(rows, hz=ESTIMATOR_HZ):
    """Return a constant-velocity Kalman estimate sampled at ``hz``."""
    if not rows:
        return np.array([], dtype=float), np.empty((0, 3), dtype=float)
    times = np.array([float(t) for t, _ in rows], dtype=float)
    measurements = np.unwrap(np.radians(np.array([values for _, values in rows], dtype=float)), axis=0)
    measurements = np.degrees(measurements)
    step = 1. / hz
    output_times = np.arange(times[0], times[-1] + step * .5, step)
    estimates = np.empty((len(output_times), 3), dtype=float)
    states = [np.array((measurements[0, axis], 0.)) for axis in range(3)]
    covariances = [np.diag((MEASUREMENT_NOISE_DEG ** 2, 90. ** 2)) for _ in range(3)]
    measurement_index = 0
    current_time = times[0]
    for output_index, output_time in enumerate(output_times):
        while measurement_index < len(times) and times[measurement_index] <= output_time + 1e-12:
            measurement_time = times[measurement_index]
            dt = max(0., measurement_time - current_time)
            for axis in range(3):
                states[axis], covariances[axis] = _step(states[axis], covariances[axis], dt)
                innovation = measurements[measurement_index, axis] - states[axis][0]
                gain = covariances[axis][:, 0] / (covariances[axis][0, 0] + MEASUREMENT_NOISE_DEG ** 2)
                states[axis] += gain * innovation
                covariances[axis] -= np.outer(gain, covariances[axis][0])
            current_time = measurement_time
            measurement_index += 1
        dt = max(0., output_time - current_time)
        for axis in range(3):
            states[axis], covariances[axis] = _step(states[axis], covariances[axis], dt)
            estimates[output_index, axis] = states[axis][0]
        current_time = output_time
    return output_times, estimates
