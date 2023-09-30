'''
Runs the Kalman filter for a simple 1D motion model.
'''

import numpy as np
import matplotlib.pyplot as plt

from kalman_filter import KF

# pylint: disable=invalid-name
# pylint: disable=unused-variable


def apply_kalman_filter(kf: KF, NUM_STEPS: int) -> tuple:
    '''
    Returns the actual states, predicted states, and covariance matrices
    after running the Kalman filter for `NUM_STEPS` steps.
    '''

    TIME_STEP = 0.1
    MEASURMENT_INTERVAL = 20

    actual_position = 0.0
    meas_variance = 0.1 ** 2
    actual_velocity = 0.5

    predicted_states = []
    covariance_matrices = []
    actual_positions = []
    actual_velocities = []

    for step in range(NUM_STEPS):
        if step > 500:
            actual_velocity *= 0.9

        covariance_matrices.append(kf.covariance)
        predicted_states.append(kf.state)

        actual_position = actual_position + TIME_STEP * actual_velocity

        kf.predict(change_in_time=TIME_STEP)

        if step != 0 and step % MEASURMENT_INTERVAL == 0:
            meas_value = actual_position + np.random.randn() * np.sqrt(meas_variance)
            kf.update(meas_value=meas_value, meas_variance=meas_variance)

        actual_positions.append(actual_position)
        actual_velocities.append(actual_velocity)

    actual_states = (actual_positions, actual_velocities)

    return actual_states, predicted_states, covariance_matrices


def main() -> None:
    '''Le Main'''

    kf = KF(initial_state=[initial_position := 0.0, initial_velocity := 1.0],
            acceleration_variance=0.1)

    result = apply_kalman_filter(kf=kf, NUM_STEPS=1000)
    actual_states, predicted_states, covariance_matrices = result
    actual_positions, actual_velocities = actual_states

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.title('Position')
    plt.plot(actual_positions, 'b')
    plt.plot([q[0] for q in predicted_states], 'r')
    plt.plot([q[0] - 2*np.sqrt(cov[0, 0])
             for q, cov in zip(predicted_states, covariance_matrices)], 'y--')
    plt.plot([q[0] + 2*np.sqrt(cov[0, 0])
             for q, cov in zip(predicted_states, covariance_matrices)], 'y--')

    plt.subplot(2, 1, 2)
    plt.title('Velocity')
    plt.plot(actual_velocities, 'b')
    plt.plot([q[1] for q in predicted_states], 'r')
    plt.plot([q[1] - 2*np.sqrt(cov[1, 1])
             for q, cov in zip(predicted_states, covariance_matrices)], 'y--')
    plt.plot([q[1] + 2*np.sqrt(cov[1, 1])
             for q, cov in zip(predicted_states, covariance_matrices)], 'y--')

    plt.show()


if __name__ == '__main__':
    main()
