import tensorflow as tf
from .__init__ import LinearModel


class InverseGamma(object):
    @classmethod
    def from_mode_and_shape(cls, mode, shape):
        scale = mode * (shape + 1)
        return cls(shape, scale)

    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale
        self._inverse_distribution = tf.distributions.Gamma(
            self.shape, self.scale)

    def sample(self):
        return 1.0 / self._inverse_distribution.sample()

    def next(self, n, sse_estimate):
        return self.__class__(self.shape + n / 2.0,
                              self.scale + sse_estimate / 2.0)

    def mode(self):
        return self.scale / (self.shape + 1)

    def mean(self):
        return self.scale / (self.shape - 1) if self.shape > 1 else None

    def variance(self):
        return (self.mean()**2.0 / (self.shape - 1.0)
                if self.shape > 2 else None)


class MultivariateNormal(object):
    @classmethod
    def from_shared_mean_measurement_variance_and_effective_sample_size(
            cls, mean, m_var, ess, num_dims):
        precision = ess / m_var
        return cls(
            tf.constant(mean / precision, shape=[num_dims, 1]),
            precision * tf.eye(num_dims))

    def __init__(self, unscaled_means, precision, normal_prior=None):
        unscaled_means = tf.convert_to_tensor(unscaled_means)
        precision = tf.convert_to_tensor(precision)
        if normal_prior is None:
            self.precision = precision
        else:
            self.precision = normal_prior.precision + precision

        if normal_prior is not None:
            unscaled_means += normal_prior.weighted_precision_sums

        L = tf.cholesky(self.precision)
        self.means = tf.cholesky_solve(L, unscaled_means)

        self.covariance_scale = tf.matrix_triangular_solve(
            L, tf.eye(L.shape[0].value))

        self.weighted_precision_sums = self.precision @ self.means
        self.quadratic_form = tf.matmul(
            self.means, self.weighted_precision_sums, transpose_a=True)

        self.distribution = tf.contrib.distributions.MultivariateNormalTriL(
            tf.transpose(self.means), self.covariance_scale)

    def sample(self):
        return tf.transpose(self.distribution.sample())

    def next(self, weighted_feature_sums, empirical_precision):
        return self.__class__(
            weighted_feature_sums, empirical_precision, normal_prior=self)

    def covariance(self):
        return self.covariance_scale @ tf.transpose(self.covariance_scale)

    def maximum_a_posteriori_estimate(self):
        return LinearModel(self.means)


class MultivariateNormalInverseGamma(object):
    def __init__(self, normal_prior, ig_prior):
        self.normal_prior = normal_prior
        self.ig_prior = ig_prior

    def sample(self):
        scale = self.ig_prior.sample()
        d = tf.contrib.distributions.MultivariateNormalTriL(
            tf.transpose(self.normal_prior.means),
            scale * self.normal_prior.covariance_scale)
        return tf.transpose(d.sample())

    def next(self, x, y):
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)

        x_T = tf.transpose(x)
        weighted_feature_sums = x_T @ y
        empirical_precision = x_T @ x

        normal_posterior = self.normal_prior.next(weighted_feature_sums,
                                                  empirical_precision)

        yty = tf.matmul(y, y, transpose_a=True, name='yty')
        sse_estimate = tf.squeeze(self.normal_prior.quadratic_form + yty -
                                  normal_posterior.quadratic_form)
        ig_posterior = self.ig_prior.next(
            tf.cast(tf.shape(x)[0], tf.float32), sse_estimate)

        return self.__class__(normal_posterior, ig_posterior)

    def maximum_a_posteriori_estimate(self):
        return self.normal_prior.maximum_a_posteriori_estimate()


class BayesianLinearRegressionDistribution(object):
    def __init__(self, prior):
        self.prior = prior
        self.posterior = self.prior

    def train(self, phi, y):
        self.posterior = self.prior.next(phi, y)
        return self

    def sample(self, n=1):
        return [LinearModel(self.posterior.sample()) for _ in range(n)]

    def maximum_a_posteriori_estimate(self):
        return self.posterior.maximum_a_posteriori_estimate()
