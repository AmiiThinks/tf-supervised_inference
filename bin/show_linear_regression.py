#!/usr/bin/env python

import tensorflow as tf
tf.enable_eager_execution()
from tf_supervised_inference.distributions import \
    MultivariateNormal, InverseGamma, MultivariateNormalInverseGamma
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import fire
from simple_pytimer import Timer


# Broad IG prior around 1
# Broad normal prior around 0
def main(ig_mode=1.0, ig_shape=2.1, mean=0, ess=0.1, num_dims=2, num_lines=10):
    tf.set_random_seed(42)
    np.random.seed(42)

    ig_timer = Timer('Create IG prior')
    with ig_timer:
        ig = InverseGamma.from_mode_and_shape(float(ig_mode), float(ig_shape))
    print(str(ig_timer))

    normal_timer = Timer('Create normal prior')
    with normal_timer:
        normal = MultivariateNormal.from_shared_mean_measurement_variance_and_effective_sample_size(
            mean=float(mean),
            m_var=ig.mode(),
            ess=float(ess),
            num_dims=int(num_dims))
    print(str(normal_timer))

    mnig = Timer('Create MNIG prior')
    with mnig:
        patient = MultivariateNormalInverseGamma(normal, ig)
    print(str(mnig))

    data_timer = Timer('Create data')
    with data_timer:
        x = np.ones([10, 2]).astype('float32')
        points = np.random.normal(0, 1, [10]).astype('float32')
        x[:, 0] = points
        w = np.random.normal(0, 1, [2, 1]).astype('float32')
        y = x @ w
    print(str(data_timer))

    inference_timer = Timer('Do inference')
    with inference_timer:
        patient = patient.next(x, y)
    print(str(inference_timer))

    plt.figure()
    plt.plot(points, y, '.')

    plt.plot(points, x @ patient.normal_prior.means)

    sample_timer = Timer('Sample from posterior')
    with sample_timer:
        for _ in range(num_lines):
            plt.plot(points, x @ patient.sample())
    print(str(sample_timer))

    plt.legend(['Data', 'MAP'])
    plt.ylim([-2, 2])
    plt.show()


if __name__ == '__main__':
    fire.Fire(main)
