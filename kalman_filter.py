'''
A simple Kalman filter implementation representing a particle moving in 1D.
'''

import numpy as np


# pylint: disable=invalid-name
# pylint: disable=unused-variable


class KF:
    '''A simple Kalman filter implementation.'''

    def __init__(self,
                 initial_state: list[float],
                 acceleration_variance: float) -> None:

        # state vector
        self._state = np.array(initial_state)

        # number of state variables
        self._num_state_vars = len(initial_state)

        # variance in the acceleration
        self._acceleration_variance = acceleration_variance

        # uncertainty in the state
        self._covariance = np.eye(self._num_state_vars)

    def predict(self, change_in_time: float) -> None:
        '''
        Predict the next state given the current state and the change in time.

        P(state k|measurement k-1) = N(μ, P)\n
        μ = A_k-1 μ_k-1\n
        P = A_k-1 P_k-1 A_k-1t + Q_k-1\n
        '''

        trans_matrix = np.eye(self._num_state_vars)
        trans_matrix[POSITION_IDX := 0][VELOCITY_IDX := 1] = change_in_time
        # trans_matrix[VELOCITY_IDX][ACCELERATION_IDX] = change_in_time

        process_noise_matrix = np.array(
            [[0.5 * change_in_time**2], [change_in_time]])

        # x = F x
        self._state = trans_matrix.dot(self._state)

        # P = F P Ft + G Gt a
        self._covariance = (
            trans_matrix.dot(self._covariance).dot(trans_matrix.T)
            + process_noise_matrix.dot(process_noise_matrix.T)
            * self._acceleration_variance)

    def update(self, meas_value: float, meas_variance: float) -> None:
        '''
        Update the state given a measurement and the variance in the
        measurement.

        P(state|measurement) = N(μ, P)\n
        μ = μ' + K (z - H μ')\n
        P = (I - K H) P'\n
        S = H P' Ht + R\n
        K = P' Ht S^-1\n
        '''

        # meas: measurement

        meas_matrix = np.array([[1, 0]])  # H = [1, 0]

        meas = np.array([meas_value])  # z
        meas_noise = np.array([meas_variance])  # R

        # y = z - H x
        meas_residual = meas - meas_matrix.dot(self._state)

        # S = H P Ht + R
        meas_covariance = self._covariance.dot(meas_matrix.T)
        noisy_meas_covariance = meas_matrix.dot(meas_covariance) + meas_noise

        # K = P Ht S^-1
        # Kalman gain determines how much weight to give predictions and
        # measurements
        kalman_gain = meas_covariance.dot(np.linalg.inv(noisy_meas_covariance))

        # x = x + K y
        self._state += kalman_gain.dot(meas_residual)

        # P = (I - K H) P
        self._covariance = np.dot(
            np.eye(self._num_state_vars) - kalman_gain.dot(meas_matrix),
            self._covariance)

    @property
    def state(self) -> np.array:
        '''Returns the state.'''
        return self._state

    @property
    def covariance(self) -> np.array:
        '''Returns the covariance matrix of the state.'''
        return self._covariance

    @property
    def position(self) -> float:
        '''Returns the position of the state.'''
        return self._state[POSITION_IDX := 0]

    @property
    def velocity(self) -> float:
        '''Returns the velocity of the state.'''
        return self._state[VELOCITY_IDX := 1]
