import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
import numpy as np
from tf_supervised_inference.data import \
    SyntheticData, \
    NoisyNoiselessSyntheticDataPair, \
    poly_basis, \
    normal_fourier_affine_params, \
    fourier_basis


class DataTest(tf.test.TestCase):
    def setUp(self):
        tf.set_random_seed(42)
        np.random.seed(42)

    def test_random_fourier_basis(self):
        x = np.random.uniform(size=[10, 2])

        for gaussian_kernel_param, expected in [
            (
                0.1,
                [
                    [0.3148444592952728, -0.6601547598838806, 0.8492576479911804],
                    [0.999679446220398, -0.9738280177116394, 0.9195378422737122],
                    [0.6647799015045166, 0.985145628452301, 0.5767946839332581],
                    [0.34385326504707336, -0.9961916208267212, 0.8349908590316772],
                    [0.9134759902954102, -0.9973570108413696, 0.9956913590431213],
                    [0.034429971128702164, -0.9410932064056396, 0.6805763840675354],
                    [0.42847099900245667, 0.32812657952308655, 0.29529690742492676],
                    [0.7085629105567932, 0.9999983310699463, 0.6093845963478088],
                    [0.995257556438446, -0.26448750495910645, 0.9598801136016846],
                    [0.7999333739280701, 0.5911393165588379, 0.6590866446495056]
                ]
            ),
            (
                1,
                [
                    [-0.322880357503891, -0.8770470023155212, -0.13028155267238617],
                    [0.36981201171875, -0.11994848400354385, -0.664853572845459],
                    [0.6642372012138367, -0.023072026669979095, -0.4067854881286621],
                    [0.28688815236091614, -0.9556680917739868, 0.9425829648971558],
                    [0.1800398975610733, -0.42274558544158936, -0.9992055892944336],
                    [0.006646182853728533, -0.9958024621009827, 0.9999452829360962],
                    [0.9960185289382935, 0.5724993348121643, 0.38363054394721985],
                    [0.745669424533844, -0.039623063057661057, -0.47564566135406494],
                    [0.8864196538925171, -0.4427263140678406, -0.48591288924217224],
                    [0.9939070343971252, 0.05412252992391586, -0.9976999759674072]
                ]
            ),
            (
                10,
                [
                    [0.4377979040145874, -0.6338453888893127, -0.9975139498710632],
                    [-0.3747597336769104, -0.9806351661682129, -0.6396830081939697],
                    [-0.9934041500091553, -0.9802564978599548, -0.9320948719978333],
                    [0.24222037196159363, -0.4018210172653198, -0.94535893201828],
                    [-0.12396690994501114, -0.9863908886909485, -0.8271189332008362],
                    [0.47293218970298767, -0.1836300790309906, -0.8919642567634583],
                    [-0.9657471776008606, -0.5424955487251282, -0.23954416811466217],
                    [-0.9836463332176208, -0.9827760457992554, -0.9259065389633179],
                    [-0.538953959941864, -0.9709486365318298, -0.9551045894622803],
                    [-0.9029608368873596, -0.9457762837409973, -0.7817054390907288]
                ]
            )
        ]:  # yapf:disable
            with self.subTest(gaussian_kernel_param):
                patient = fourier_basis(x,
                                        *normal_fourier_affine_params([2, 3]))
                assert patient.shape[0].value == 10
                assert patient.shape[1].value == 3
                self.assertAllClose(expected, patient)

    def test_poly_basis(self):
        x = np.random.uniform(size=[10, 1])
        patient = poly_basis(x, 3)
        assert patient.shape[0].value == 10
        assert patient.shape[1].value == 3
        self.assertAllClose(x, patient[:, 0:1])
        self.assertAllClose(x**2 / 2.0, patient[:, 1:2])
        self.assertAllClose(x**3 / 3.0, patient[:, 2:3])

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
        assert len(patient.noiseless.good_phi()) == len(
            patient.noiseless.good_y())
        assert len(patient.noiseless.bad_phi()) == len(
            patient.noiseless.bad_y())

        assert patient.noisy.num_examples() == 10
        assert patient.noisy.num_features() == 2
        assert patient.noisy.num_outputs() == 1
        self.assertAllGreater(patient.noisy.good_y(), 0)
        self.assertAllLess(patient.noisy.bad_y(), 0)
        assert len(patient.noisy.good_phi()) == len(patient.noisy.good_y())
        assert len(patient.noisy.bad_phi()) == len(patient.noisy.bad_y())


if __name__ == '__main__':
    tf.test.main()
