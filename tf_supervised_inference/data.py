import numpy as np
from collections import namedtuple


class SyntheticData(object):
    @classmethod
    def from_finite_generator(cls, data_generator, **kwargs):
        phi, y = zip(*data_generator)
        phi = np.concatenate(phi, axis=0)
        y = np.concatenate(y, axis=0)
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