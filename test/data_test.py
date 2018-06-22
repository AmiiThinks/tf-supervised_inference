import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
import numpy as np
from tf_supervised_inference.data import \
    SyntheticData, \
    NoisyNoiselessSyntheticDataPair


class DataTest(tf.test.TestCase):
    def setUp(self):
        tf.set_random_seed(42)
        np.random.seed(42)

    def test_init_synthetic_data(self):
        patient = SyntheticData(
            np.random.normal(size=[10, 2]), np.random.normal(size=[10, 1]), 0)

        assert patient.num_examples() == 10
        assert patient.num_features() == 2
        assert patient.num_outputs() == 1
        self.assertAllGreater(patient.good_y(), 0)
        self.assertAllLess(patient.bad_y(), 0)
        assert len(patient.good_phi()) == len(patient.good_y())
        assert len(patient.bad_phi()) == len(patient.bad_y())

    def test_init_noisy_noiseless_synthetic_data_pair(self):
        patient = NoisyNoiselessSyntheticDataPair.from_stddev(
            SyntheticData(
                np.random.normal(size=[10, 2]),
                np.random.normal(size=[10, 1]),
                0
            ),
            1
        )  # yapf:disable

        assert patient.noiseless.num_examples() == 10
        assert patient.noiseless.num_features() == 2
        assert patient.noiseless.num_outputs() == 1
        self.assertAllGreater(patient.noiseless.good_y(), 0)
        self.assertAllLess(patient.noiseless.bad_y(), 0)
        assert len(patient.noiseless.good_phi()) == len(patient.noiseless.good_y())
        assert len(patient.noiseless.bad_phi()) == len(patient.noiseless.bad_y())

        assert patient.noisy.num_examples() == 10
        assert patient.noisy.num_features() == 2
        assert patient.noisy.num_outputs() == 1
        self.assertAllGreater(patient.noisy.good_y(), 0)
        self.assertAllLess(patient.noisy.bad_y(), 0)
        assert len(patient.noisy.good_phi()) == len(patient.noisy.good_y())
        assert len(patient.noisy.bad_phi()) == len(patient.noisy.bad_y())


if __name__ == '__main__':
    tf.test.main()
