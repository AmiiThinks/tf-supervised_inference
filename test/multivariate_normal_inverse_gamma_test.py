import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
from tf_supervised_inference.distributions import \
    MultivariateNormal, InverseGamma, MultivariateNormalInverseGamma
import numpy as np


class MultivariateNormalInverseGammaTest(tf.test.TestCase):
    def setUp(self):
        tf.set_random_seed(42)
        np.random.seed(42)

    def test_init(self):
        ig = InverseGamma.from_log_mode_and_log_shape(
            tf.log(1.0), tf.log(2.1))  # Broad prior around 1
        self.assertAllClose(ig.mode(), 1.0)
        self.assertAllClose(ig.variance(), 7.22013523666416)

        normal = MultivariateNormal.from_shared_mean_and_log_precision(
            mean=0, log_precision=tf.log(0.1),
            num_dims=2)  # Broad prior around 0
        self.assertAllClose(0.1, normal.precision[0, 0])
        self.assertAllClose(0.1, normal.precision[1, 1])

        patient = MultivariateNormalInverseGamma(normal, ig)
        assert patient is not None

    def test_sample(self):
        ig = InverseGamma.from_log_mode_and_log_shape(
            tf.log(1.0), tf.log(2.1))  # Broad prior around 1
        self.assertAllClose(ig.mode(), 1.0)
        self.assertAllClose(ig.variance(), 7.22013523666416)

        normal = MultivariateNormal.from_shared_mean_and_log_precision(
            mean=0, log_precision=tf.log(0.1),
            num_dims=2)  # Broad prior around 0
        self.assertAllEqual([2, 1], tf.shape(normal.means))
        self.assertAllEqual([2, 2], tf.shape(normal.precision))
        self.assertAllClose(0.1, normal.precision[0, 0])
        self.assertAllClose(0.1, normal.precision[1, 1])

        patient = MultivariateNormalInverseGamma(normal, ig)
        self.assertAllClose(patient.sample(), [[0.292883], [-2.993716]])

    def test_next(self):
        ig = InverseGamma.from_log_mode_and_log_shape(
            tf.log(1.0), tf.log(2.1))  # Broad prior around 1
        self.assertAllClose(ig.mode(), 1.0)
        self.assertAllClose(ig.variance(), 7.22013523666416)

        normal = MultivariateNormal.from_shared_mean_and_log_precision(
            mean=0, log_precision=tf.log(0.1),
            num_dims=2)  # Broad prior around 0
        self.assertAllEqual([2, 1], tf.shape(normal.means))
        self.assertAllEqual([2, 2], tf.shape(normal.precision))
        self.assertAllClose(0.1, normal.precision[0, 0])
        self.assertAllClose(0.1, normal.precision[1, 1])

        patient = MultivariateNormalInverseGamma(normal, ig)

        num_features = 2
        x = np.random.normal(0, 1, [10, num_features]).astype('float32')
        y = np.random.normal(0, 1, [10, 1]).astype('float32')
        patient = patient.next(x, y)

        self.assertAllClose([[-0.201125], [-0.117803]], patient.sample())
        self.assertAllClose([[-0.03634], [-0.094946]], patient.sample())
        self.assertAllClose([[-0.215406], [0.160784]], patient.sample())


if __name__ == '__main__':
    tf.test.main()
