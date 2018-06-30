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


class Data(object):
    def __init__(self, phi, y):
        self._data = (tf.convert_to_tensor(phi), tf.convert_to_tensor(y))

    def __iter__(self):
        return iter(self._data)

    def random_training_and_validation_sets(self,
                                            num_shuffles=1,
                                            training_proportion=0.5):
        for shuffle in range(num_shuffles):
            shuffle_indices = np.random.randint(
                self.num_examples(), size=[self.num_examples()])

            shuffled_data = Data(
                tf.gather(self.phi, shuffle_indices),
                tf.gather(self.y, shuffle_indices))

            num_training_examples = max(
                1,
                min(
                    int(np.ceil(self.num_examples() * training_proportion)),
                    self.num_examples() - 1))

            tdata, vdata = shuffled_data.split([
                num_training_examples,
                self.num_examples() - num_training_examples
            ])

            yield tdata, vdata

    @property
    def phi(self):
        return self._data[0]

    @property
    def y(self):
        return self._data[1]

    def with_noise(self, stddev=1.0):
        return self.__class__(self.phi, (self.y + tf.random_normal(
            mean=0, stddev=stddev, shape=self.y.shape)))

    def clone(self, phi=lambda x: x):
        return self.__class__(phi(self.phi), self.y)

    def num_examples(self):
        n = self.phi.shape[0].value
        return tf.shape(self.phi.shape)[0] if n is None else n

    def num_features(self):
        return self.phi.shape[1].value

    def num_outputs(self):
        return self.y.shape[1].value

    def split(self, num_or_size_splits):
        return [
            self.__class__(phi, y)
            for phi, y in zip(
                tf.split(self.phi, num_or_size_splits),
                tf.split(self.y, num_or_size_splits))
        ]


def NamedDataSets(**named_data_sets):
    names, data_sets = zip(*named_data_sets.items())

    class NamedDataSets(namedtuple('NamedDataSets', names)):
        def all(self):
            phis, ys = zip(*self)
            return Data(tf.concat(phis, axis=0), tf.concat(ys, axis=0))

        def clone(self, phi=lambda x: tf.identity(x)):
            items = self._asdict().items()
            return self.__class__(
                **{name: data.clone(phi)
                   for name, data in items})

    return NamedDataSets(*data_sets)
