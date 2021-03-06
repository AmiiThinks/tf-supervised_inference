import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
from tf_supervised_inference.distributions import ImproperMultivariateNormal
import numpy as np


class ImproperMultivariateNormalTest(tf.test.TestCase):
    def setUp(self):
        tf.set_random_seed(42)
        np.random.seed(42)

    def test_init(self):
        patient = ImproperMultivariateNormal(
            tf.zeros([2, 1]), tf.zeros([2, 2]))
        self.assertAllEqual(patient.means, tf.zeros([2, 1]))
        self.assertAllEqual(patient.precision, tf.zeros([2, 2]))

    def test_next(self):
        patient = ImproperMultivariateNormal(
            tf.zeros([2, 1]), tf.zeros([2, 2]))

        num_features = 2
        x = np.random.normal(0, 1, [10, num_features]).astype('float32')
        y = np.random.normal(0, 1, [10, 1]).astype('float32')
        weighted_feature_sums = x.T @ y
        empirical_precision = x.T @ x
        patient = patient.next(weighted_feature_sums, empirical_precision)

        self.assertAllClose(patient.means, [[-0.225088], [0.107223]])
        self.assertAllClose(patient.precision,
                            [[8.534351, 3.616732], [3.616732, 9.563951]])


if __name__ == '__main__':
    tf.test.main()
