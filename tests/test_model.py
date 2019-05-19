import tensorflow as tf
import numpy as np
from model import CMPS, RhoCMPS, PsiCMPS
from data import get_audio

from tensorflow.contrib.training import HParams


tf.flags.DEFINE_integer("sample_duration", 2**8, "Duration of samples (as integer).")
tf.flags.DEFINE_integer("sample_rate", 16000, "Sampling rate.")
FLAGS = tf.flags.FLAGS

hparams = HParams(minibatch_size=8, bond_dim=8, delta_t=1/FLAGS.sample_rate, sigma=1, initial_rank=None, A=1,
                  h_reg=2/(np.pi * FLAGS.sample_rate)**2, r_reg=2/(np.pi * FLAGS.sample_rate)**2,)


class TestCMPS(tf.test.TestCase):

    def testHIsHermitian(self):

        model = CMPS(hparams)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(model.H, tf.linalg.adjoint(model.H))


class TestRhoCMPS(tf.test.TestCase):

    def testLossNotNaN(self):
        data = get_audio(None, 'damped_sine', hparams)
        model = RhoCMPS(hparams, data_iterator=data)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertNotEqual(model.loss.eval(), np.nan)

    def testRho0IsADensityMatrix(self):

        model = RhoCMPS(hparams)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(model.rho_0, model.rho_0 / tf.trace(model.rho_0))
            self.assertAllClose(model.rho_0, tf.transpose(model.rho_0, conjugate=True))

    def testHInitialization(self):

        test_H_diag = np.random.rand(hparams.bond_dim).astype(dtype=np.float32)
        model = RhoCMPS(hparams, H_in=test_H_diag)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(model.H, np.diag(test_H_diag))

    def testTrivialUpdateOfAncilla(self):
        """
        Update with H=R=0
        """

        test_H_diag = np.zeros([hparams.bond_dim], dtype=np.float32)
        test_R = np.zeros(2*[hparams.bond_dim], dtype=np.float32)
        signal = np.random.rand(hparams.minibatch_size).astype(dtype=np.float32)
        model = RhoCMPS(hparams, H_in=test_H_diag, Rx_in=test_R, Ry_in=test_R)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            stack_rho_0 = tf.stack(hparams.minibatch_size * [model.rho_0])
            updated_rho = model._update_ancilla_rho(stack_rho_0, signal)
            self.assertAllClose(stack_rho_0, updated_rho)

    def testSampling(self):
        """
        Samples from a two-level system
        """

        hparams = HParams(minibatch_size=8, bond_dim=2, delta_t=1 / FLAGS.sample_rate, sigma=1, initial_rank=None, A=1,
                          h_reg=2 / (np.pi * FLAGS.sample_rate) ** 2, r_reg=2 / (np.pi * FLAGS.sample_rate) ** 2, )

        ω = 10
        R = np.array([[0, 1], [0, 0]], dtype=np.float32)
        H = np.array([[ω, 0], [0, -ω]], dtype=np.float32)

        qubit = RhoCMPS(hparams, Rx_in=R, H_in=H)

        waveform = qubit.sample_rho(num_samples=2, length=512)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertEqual(waveform.eval().shape, (2, 512))


if __name__ == '__main__':
      tf.test.main()