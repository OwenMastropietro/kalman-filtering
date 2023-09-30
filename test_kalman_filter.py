'''
Unit tests for the KalmanFilter class.
'''

import unittest
import numpy as np
from kalman_filter import KF


# pylint: disable=unused-variable


class TestKalmanFilter(unittest.TestCase):
    '''Unit tests for the KalmanFilter class.'''

    def test_can_construct_with_initial_state(self):
        '''
        Tests that the KalmanFilter can be constructed with an initial state.
        '''

        initial_state = [initial_position := 0.0, initial_velocity := 1.0]

        kalman_filter = KF(initial_state=initial_state,
                           acceleration_variance=0.1)

        self.assertEqual(kalman_filter.state.tolist(), initial_state)

    def test_shape_of_state_and_covariance_after_prediction(self):
        '''
        Tests that the state and covariance are of the right shape after
        calling predict.
        '''

        initial_state = [initial_position := 0.2, initial_velocity := 2.3]

        kalman_filter = KF(initial_state=initial_state,
                           acceleration_variance=1.2)
        kalman_filter.predict(0.1)

        self.assertEqual(kalman_filter.covariance.shape, (2, 2))
        self.assertEqual(kalman_filter.state.shape, (2, ))

    def test_increased_uncertainty_after_prediction(self):
        '''
        Tests that the uncertainty in the state increases after calling
        predict.
        '''

        initial_state = [initial_position := 0.2, initial_velocity := 2.3]

        kalman_filter = KF(initial_state=initial_state,
                           acceleration_variance=1.2)

        for _ in range(10):
            det_before = np.linalg.det(kalman_filter.covariance)
            kalman_filter.predict(change_in_time=0.1)
            det_after = np.linalg.det(kalman_filter.covariance)

            self.assertGreater(det_after, det_before)
            # print(det_before, det_after)

    def test_decreased_uncertainty_after_update(self):
        '''
        Tests that the uncertainty in the state decreases after calling
        update.
        '''

        initial_state = [initial_position := 0.2, initial_velocity := 2.3]

        kalman_filter = KF(initial_state=initial_state,
                           acceleration_variance=1.2)

        det_before = np.linalg.det(kalman_filter.covariance)
        kalman_filter.update(meas_value=0.1, meas_variance=0.01)
        det_after = np.linalg.det(kalman_filter.covariance)

        self.assertLess(det_after, det_before)


if __name__ == '__main__':
    unittest.main()
