import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
from tf_supervised_inference.distributions import InverseGamma
import numpy as np


class InverseGammaTest(tf.test.TestCase):
    def setUp(self):
        tf.set_random_seed(42)
        np.random.seed(42)

    def test_init(self):
        patient = InverseGamma(shape=1.0, scale=2.0)
        assert patient.shape == 1.0
        assert patient.scale == 2.0

    def test_sample(self):
        patient = InverseGamma(shape=1.0, scale=2.0)
        self.assertAllClose(patient.sample().numpy(), 1.154769)

    def test_summary_stats(self):
        params = [
            (1.0, 0.1), (1.0, 1.0), (1.0, 2.0), (1.0, 0.1), (1.0, 10.0),
            (9999.0, 15000.0)
        ]  # yapf: disable

        expected = [
            [466, 19, 6, 4, 2, 1, 0, 2, 0, 0],
            [187, 95, 66, 30, 17, 17, 9, 5, 5, 69],
            [67, 106, 86, 46, 48, 19, 11, 19, 10, 88],
            [451, 19, 10, 3, 4, 3, 1, 2, 0, 7],
            [0, 3, 9, 27, 37, 26, 32, 18, 16, 332],
            [0, 500, 0, 0, 0, 0, 0, 0, 0, 0]
        ]  # yapf: disable

        for i in range(len(params)):
            with self.subTest():
                x = expected[i]
                shape, scale = params[i]
                patient = InverseGamma(shape, scale)

                samples = tf.stack([patient.sample() for _ in range(500)])
                hist = tf.histogram_fixed_width(samples, [0.0, 10.0], nbins=10)
                self.assertAllClose(hist, x)

    def test_next(self):
        x = np.random.normal(0, 1, [100])
        shape, scale = (1.0, 2.0)  # Mostly flat prior
        patient = InverseGamma(shape, scale)
        patient = patient.next(len(x), 12.0)

        self.assertAllClose(patient.shape, 1 + 100 / 2.0)
        self.assertAllClose(patient.scale, 2 + 12.0 / 2.0)

    def test_from_log_mode_and_log_shape(self):
        params = [
            (1.0, 0.1), (1.0, 1.0), (1.0, 2.0), (1.0, 0.1), (1.0, 10.0),
            (9999.0, 15000.0)
        ]  # yapf: disable
        for shape, scale in params:
            mode = scale / (shape + 1.0)
            patient = InverseGamma.from_log_mode_and_log_shape(
                tf.log(mode), tf.log(shape))

            self.assertAllClose(patient.shape, shape)
            self.assertAllClose(patient.scale, scale)
            self.assertAllClose(patient.mode(), mode)

            if shape > 1:
                self.assertAllClose(patient.mean(), scale / (shape - 1.0))
            else:
                self.assertIsNone(patient.mean())

            if shape > 2:
                self.assertAllClose(patient.variance(), scale**2.0 /
                                    ((shape - 1.0)**2.0 * (shape - 1.0)))
            else:
                self.assertIsNone(patient.variance())


if __name__ == '__main__':
    tf.test.main()
