import tensorflow as tf
import numpy as np
from collections import namedtuple


def normal_fourier_affine_params(shape, mean=0, gaussian_kernel_param=1.0):
    w = tf.random_normal(
        mean=mean, stddev=2 * gaussian_kernel_param, shape=shape)
    b = tf.random_uniform([], minval=-np.pi, maxval=np.pi)
    return w, b


def fourier_basis(x, w, b):
    return tf.cos(x @ w + b)


def poly_basis(x, degree):
    return tf.concat(
        [1.0 / (i + 1) * x**(i + 1) for i in range(degree)], axis=-1)


def multi_predict(inputs, *functions):
    return tf.stack([f(inputs) for f in functions], axis=-1)


def empirical_predictive_distribution(inputs, *functions):
    ys = multi_predict(inputs, *functions)
    counts, sums, sum_of_squares, _ = tf.nn.sufficient_statistics(
        ys, axes=[-1])
    means = sums / counts
    var = (sum_of_squares - (sums**2 / counts)) / (counts - 1)
    return means, var


class SyntheticData(object):
    @classmethod
    def from_finite_generator(cls, data_generator, **kwargs):
        phi, y = zip(*data_generator)
        phi = np.stack(phi)
        y = np.stack(y)
        return cls(phi, y, **kwargs)

    def __init__(self, phi, y, threshold=0):
        self.threshold = threshold
        self.phi = phi
        self.y = y
        self._partition_examples()

    def num_examples(self):
        return self.phi.shape[0]

    def num_features(self):
        return self.phi.shape[1]

    def num_outputs(self):
        return self.y.shape[1]

    def _partition_examples(self):
        self.bad_examples = self.y < self.threshold
        self.good_examples = np.logical_not(self.bad_examples)
        return self

    def add_bias(self, b):
        self.y += b
        return self._partition_examples()

    def with_noise(self, stddev=1.0):
        return self.__class__(
            self.phi,
            (
                self.y
                + np.random.normal(
                    loc=0,
                    scale=stddev,
                    size=self.y.shape
                ).astype('float32')
            )
        )  # yapf: disable

    def good_y(self, j=0):
        return self.y[self.good_examples[:, j], j:j + 1]

    def bad_y(self, j=0):
        return self.y[self.bad_examples[:, j], j:j + 1]

    def good_phi(self, j=0):
        return self.phi[self.good_examples[:, j], :]

    def bad_phi(self, j=0):
        return self.phi[self.bad_examples[:, j], :]

    def clone(self, phi=lambda x: x):
        return self.__class__(phi(self.phi), self.y, threshold=self.threshold)


class NoisyNoiselessSyntheticDataPair(
        namedtuple('_NoisyNoiselessSyntheticDataPair',
                   ['noiseless', 'noisy'])):
    @classmethod
    def from_stddev(cls, data, stddev=1.0):
        return cls(data, data.with_noise(stddev))

    def clone(self, phi=lambda x: x):
        return self.__class__(self.noiseless.clone(phi), self.noisy.clone(phi))
