import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
from tf_supervised_inference.distributions import \
    BayesianLinearRegressionDistribution
import numpy as np


class BayesianLinearRegressionDistributionTest(tf.test.TestCase):
    def setUp(self):
        tf.set_random_seed(42)
        np.random.seed(42)

    def test_all(self):
        num_features = 2
        x = np.random.normal(0, 1, [1000, num_features]).astype('float32')
        w = np.random.normal(0, 1, [num_features, 1]).astype('float32')
        y = x @ w
        for (patient, mean, log_precision, log_ig_mode,
             log_ig_shape) in BayesianLinearRegressionDistribution.all(x, y):
            map = patient.maximum_a_posteriori_estimate()

            self.assertAllClose(w, map.weights[0], atol=1e-3, rtol=1e-2)

    def test_all_with_fixed_measurement_error(self):
        num_features = 2
        x = np.random.normal(0, 1, [1000, num_features]).astype('float32')
        w = np.random.normal(0, 1, [num_features, 1]).astype('float32')
        y = x @ w

        x_errors = [
            (1e-6, 1e-5),
            (1e-5, 1e-4),
            (1e-5, 1e-4),
            (1e-4, 1e-3),
            (1e-4, 1e-3),
            (1e-4, 1e-3),
            (1e-3, 1e-2),
        ]
        x_errors.reverse()
        for (patient, mean, log_precision, log_ig_mode,
             log_ig_shape) in BayesianLinearRegressionDistribution.all(
                 x, y, log_ig_shape_params=[np.inf]):
            map = patient.maximum_a_posteriori_estimate()
            x_atol, x_rtol = x_errors.pop()

            self.assertAllClose(w, map.weights[0], atol=x_atol, rtol=x_rtol)


if __name__ == '__main__':
    tf.test.main()
