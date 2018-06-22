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


class Data(namedtuple('_Data', ['phi', 'y'])):
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
    )

  def clone(self, phi=lambda x: x):
    return self.__class__(phi(self.phi), self.y)

  def num_examples(self):
    return self.phi.shape[0]

  def num_features(self):
    return self.phi.shape[1]

  def num_outputs(self):
    return self.y.shape[1]


class TrainingValidationDataPair(
    namedtuple('_TrainingValidationDataPair', ['t', 'v'])
):
  def all(self):
    return Data(
        tf.concat([self.t.phi, self.v.phi], axis=0),
        tf.concat([self.t.y, self.v.y], axis=0))

  def clone(self, phi=lambda x: x):
    return self.__class__(self.t.clone(phi), self.v.clone(phi))
