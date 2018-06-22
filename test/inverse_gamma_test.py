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
        self.assertAlmostEquals(patient.sample().numpy(), 1.155, places=3)

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

        self.assertAlmostEquals(patient.shape, 1 + 100 / 2.0)
        self.assertAlmostEquals(patient.scale, 2 + 12.0 / 2.0)

    def test_from_mode_and_shape(self):
        params = [
            (1.0, 0.1), (1.0, 1.0), (1.0, 2.0), (1.0, 0.1), (1.0, 10.0),
            (9999.0, 15000.0)
        ]  # yapf: disable
        for shape, scale in params:
            mode = scale / (shape + 1.0)
            patient = InverseGamma.from_mode_and_shape(mode, shape)

            self.assertAlmostEquals(patient.shape, shape)
            self.assertAlmostEquals(patient.scale, scale)
            self.assertAlmostEquals(patient.mode(), mode)

            if shape > 1:
                self.assertAlmostEquals(patient.mean(), scale / (shape - 1.0))
            else:
                self.assertAlmostEquals(patient.mean(), None)

            if shape > 2:
                self.assertAlmostEquals(patient.variance(), scale**2.0 /
                                        ((shape - 1.0)**2.0 * (shape - 1.0)))
            else:
                self.assertAlmostEquals(patient.variance(), None)


if __name__ == '__main__':
    tf.test.main()
