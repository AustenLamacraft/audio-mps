import tensorflow as tf
import numpy as np
from model import CMPS, RhoCMPS, PsiCMPS
from data import get_audio

from tensorflow.contrib.training import HParams


tf.flags.DEFINE_integer("sample_duration", 2**8, "Duration of samples (as integer).")
tf.flags.DEFINE_integer("sample_rate", 16000, "Sampling rate.")
FLAGS = tf.flags.FLAGS

hparams = HParams(minibatch_size=8, bond_dim=7, delta_t=1/FLAGS.sample_rate, sigma=0.0001, initial_rank=None, A=1.,
                  h_reg=2/(np.pi * FLAGS.sample_rate)**2, r_reg=2/(np.pi * FLAGS.sample_rate)**2)


class TestCMPS(tf.test.TestCase):

    def testRHasNoDiagonalElements(self):

        model = CMPS(hparams)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(tf.matrix_diag_part(model.R.eval()), hparams.bond_dim * [0.])

if __name__ == '__main__':
      tf.test.main()


class TestRhoCMPS(tf.test.TestCase):

    def testLossNotNaN(self):
        data = get_audio(None, 'damped_sine', hparams)
        model = RhoCMPS(hparams, data_iterator=data)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            assert not np.isnan(model.loss.eval())

    def testRho0IsADensityMatrix(self):

        model = RhoCMPS(hparams)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(model.rho_0, model.rho_0 / tf.trace(model.rho_0))
            self.assertAllClose(model.rho_0, tf.transpose(model.rho_0, conjugate=True))

    def testRhoEvolvedWithDataRemainsNormalized(self):
        data = get_audio(None, 'damped_sine', hparams)
        model = RhoCMPS(hparams, data_iterator=data)
        rho_out = model.rho_evolve_with_data()

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(tf.trace(rho_out), tf.ones_like(rho_out[:,:,0,0]), rtol=1e-5)

    def testRhoEvolvedSamplingRemainsNormalized(self):

        model = RhoCMPS(hparams)
        samples = 5
        rho_out = model.rho_evolve_with_sampling(num_samples=samples, length=FLAGS.sample_duration)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(tf.trace(rho_out), tf.ones([samples, FLAGS.sample_duration]), rtol=1e-4)

    def testTrivialUpdateOfAncilla(self):
        """
        Update with H=R=0
        """

        test_H_diag = np.zeros([hparams.bond_dim], dtype=np.float32)
        test_R = np.zeros(2*[hparams.bond_dim], dtype=np.complex64)
        signal = np.random.rand(hparams.minibatch_size).astype(dtype=np.float32)
        model = RhoCMPS(hparams, freqs_in=test_H_diag, R_in=test_R)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            stack_rho_0 = tf.stack(hparams.minibatch_size * [model.rho_0])
            updated_rho = model._update_ancilla_rho(stack_rho_0, signal, 0.)
            self.assertAllClose(stack_rho_0, updated_rho)

    def testSampling(self):
        """
        Samples from a two-level system
        """

        hparams = HParams(minibatch_size=8, bond_dim=2, delta_t=1 / FLAGS.sample_rate, sigma=1, initial_rank=None, A=1.,
                          h_reg=2 / (np.pi * FLAGS.sample_rate) ** 2, r_reg=2 / (np.pi * FLAGS.sample_rate) ** 2)

        ω = 10
        R = np.array([[0, 1], [0, 0]], dtype=np.complex64)
        freqs = np.array([ω, -ω], dtype=np.float32)

        qubit = RhoCMPS(hparams, R_in=R, freqs_in=freqs)

        waveform = qubit.sample(num_samples=2, length=512)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertEqual(waveform.eval().shape, (2, 512))

class TestPsiCMPS(tf.test.TestCase):

    def testLossNotNaN(self):
        data = get_audio(None, 'damped_sine', hparams)
        model = PsiCMPS(hparams, data_iterator=data)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            assert not np.isnan(model.loss.eval())

    def testPsiEvolvedWithDataRemainsNormalized(self):
        data = get_audio(None, 'damped_sine', hparams)
        model = PsiCMPS(hparams, data_iterator=data)
        psi_out = model.psi_evolve_with_data()

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(tf.norm(psi_out, axis=-1), tf.ones_like(psi_out[:,:,0]), rtol=1e-5)

    def testTrivialUpdateOfAncilla(self):
        """
        Update with H=R=0
        """

        test_H_diag = np.zeros([hparams.bond_dim], dtype=np.float32)
        test_R = np.zeros(2*[hparams.bond_dim], dtype=np.complex64)
        signal = np.random.rand(hparams.minibatch_size).astype(dtype=np.float32)
        model = PsiCMPS(hparams, freqs_in=test_H_diag, R_in=test_R)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            stack_psi_0 = tf.stack(hparams.minibatch_size * [model.psi_0])
            updated_psi = model._update_ancilla_psi(stack_psi_0, signal, 0.)
            self.assertAllClose(stack_psi_0, updated_psi)

    def testSampling(self):
        """
        Samples from a two-level system
        """

        hparams = HParams(minibatch_size=8, bond_dim=2, delta_t=1 / FLAGS.sample_rate, sigma=1, initial_rank=None, A=1.,
                          h_reg=2 / (np.pi * FLAGS.sample_rate) ** 2, r_reg=2 / (np.pi * FLAGS.sample_rate) ** 2)

        ω = 10
        R = np.array([[0, 1], [0, 0]], dtype=np.complex64)
        freqs = np.array([ω, -ω], dtype=np.float32)

        qubit = PsiCMPS(hparams, R_in=R, freqs_in=freqs)

        waveform = qubit.sample(num_samples=2, length=512)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertEqual(waveform.eval().shape, (2, 512))