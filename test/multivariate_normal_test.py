import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
from tf_supervised_inference.distributions import MultivariateNormal
import numpy as np


class MultivariateNormalTest(tf.test.TestCase):
    def setUp(self):
        tf.set_random_seed(42)
        np.random.seed(42)

    def test_from_shared_mean_and_log_precision(self):
        patient = MultivariateNormal.from_shared_mean_and_log_precision(
            0, tf.log(4.0), 1)
        self.assertAllEqual(patient.means, tf.zeros([1, 1]))
        self.assertAllEqual(patient.precision, 4 * tf.eye(1))

    def test_sample(self):
        patient = MultivariateNormal.from_shared_mean_and_log_precision(
            0, tf.log(2.0), 1)
        self.assertAllClose(patient.sample(), [[0.231555]])

    def test_next(self):
        patient = MultivariateNormal.from_shared_mean_and_log_precision(
            0, tf.log(2.0), 1)

        num_features = 2
        x = np.random.normal(0, 1, [10, num_features]).astype('float32')
        y = np.random.normal(0, 1, [10, 1]).astype('float32')
        weighted_feature_sums = x.T @ y
        empirical_precision = x.T @ x
        patient = patient.next(weighted_feature_sums, empirical_precision)

        self.assertAllClose(patient.means,
                            [[-0.20955732464790344], [0.12006434053182602]])
        self.assertAllClose(patient.precision,
                            [[10.534350395202637, 5.616731643676758],
                             [5.616731643676758, 11.563950538635254]])


if __name__ == '__main__':
    tf.test.main()
