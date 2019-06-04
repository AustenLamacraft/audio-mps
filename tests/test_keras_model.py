import tensorflow as tf
import numpy as np
from keras_model import CMPSCell, PsiCMPSCell, StochasticSchrodinger, SchrodingerRNN
from utils import normalize
from data import get_audio

from tensorflow.contrib.training import HParams


tf.flags.DEFINE_integer("sample_duration", 2**8, "Duration of samples (as integer).")
tf.flags.DEFINE_integer("sample_rate", 16000, "Sampling rate.")
FLAGS = tf.flags.FLAGS

hps = HParams(minibatch_size=8, bond_dim=7, delta_t=1 / FLAGS.sample_rate, sigma=0.0001, initial_rank=None, A=100.,
              h_reg=2/(np.pi * FLAGS.sample_rate)**2, r_reg=2/(np.pi * FLAGS.sample_rate),
              h_scale=2/(np.pi * FLAGS.sample_rate)**2, r_scale=2/(np.pi * FLAGS.sample_rate))


class TestCMPSCell(tf.test.TestCase):

    def testRHasNoDiagonalElements(self):

        model = CMPSCell(hps)
        model.build(0)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(tf.matrix_diag_part(model.R.eval()), hps.bond_dim * [0.])

    def testFreqsCorrectlyInitializedGivenInitialValues(self):

        test_freqs = np.random.rand(hps.bond_dim).astype(dtype=np.float32)
        test_R = np.random.rand(hps.bond_dim, hps.bond_dim).astype(dtype=np.float32)

        cell = CMPSCell(hps, freqs_in=test_freqs, R_in=test_R)
        cell.build(0)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllEqual(test_freqs, cell.freqs.eval())
            # self.assertAllEqual(test_R, cell.R.eval())

    def testRCorrectlyInitializedGivenInitialValues(self):

        test_freqs = np.random.rand(hps.bond_dim).astype(dtype=np.float32)
        test_R = np.random.rand(hps.bond_dim, hps.bond_dim).astype(dtype=np.float32)
        # Remove diagonal
        test_R -= np.diag(np.diag(test_R))

        cell = CMPSCell(hps, freqs_in=test_freqs, R_in=test_R)
        cell.build(0)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllEqual(test_R, cell.R.eval())

class TestPsiCMPSCell(tf.test.TestCase):

    def testInitialStateHasCorrectShapeAndNormalization(self):

        cell = PsiCMPSCell(hps)
        psi_0, _ = cell.get_initial_state(batch_size=hps.minibatch_size)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertEqual(psi_0.shape, (hps.minibatch_size, hps.bond_dim))
            self.assertAllClose(psi_0, normalize(psi_0, axis=1))
            norms = tf.reduce_sum(tf.square(tf.abs(psi_0)), axis=1)
            self.assertAllClose(norms, tf.ones([hps.minibatch_size], dtype=tf.complex64))

    def testTrivialUpdatOfAncilla(self):

        test_freqs = np.zeros([hps.bond_dim], dtype=np.float32)
        test_R = np.zeros(2 * [hps.bond_dim], dtype=np.complex64)
        signal = np.random.rand(hps.minibatch_size).astype(dtype=np.float32)
        time = np.random.rand(hps.minibatch_size).astype(dtype=np.float32)
        input = tf.stack([signal, time], axis=1)

        cell = PsiCMPSCell(hps, freqs_in=test_freqs, R_in=test_R)
        psi_0 = cell.get_initial_state(batch_size=hps.minibatch_size)[0]
        cell.build(0)

    def testExpectationOfIdentityCloseToZero(self):
        # Note that it's zero because we remove diagonal
        test_freqs = np.random.rand(hps.bond_dim).astype(dtype=np.float32)
        test_R = np.identity(hps.bond_dim, dtype=np.complex64)
        time = np.random.rand(hps.minibatch_size).astype(dtype=np.float32)

        cell = PsiCMPSCell(hps, freqs_in=test_freqs, R_in=test_R)
        cell.build(0)

        psi_0 = cell.get_initial_state(batch_size=hps.minibatch_size)[0]
        exp = cell._expectation(psi_0, time)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(exp, tf.zeros([hps.minibatch_size], dtype=tf.float32))

    def testRegularizerLosses(self):
        cell = PsiCMPSCell(hps)
        cell.build(0)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            Rx_reg, Ry_reg, freqs_reg = cell.losses
            # This doesn't work for R because we have removed diagonal elements
            self.assertAllClose(freqs_reg, hps.h_reg * tf.reduce_sum(tf.square(cell.freqs)))


class TestStochasticSchrodinger(tf.test.TestCase):

    def testOutputCorrectShape(self):
        rnn = StochasticSchrodinger(hps)
        signal = np.random.rand(hps.minibatch_size, FLAGS.sample_duration).astype(dtype=np.float32)
        output = rnn(signal)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            output_eval = output.eval()
            self.assertEqual(output_eval.shape, (hps.minibatch_size, FLAGS.sample_duration))


class TestSchrodingerRNN(tf.test.TestCase):

    def testCallGivesCorrectShape(self):
        model = SchrodingerRNN(hps)
        signal = np.random.rand(hps.minibatch_size, FLAGS.sample_duration).astype(dtype=np.float32)
        # Note that keras handles sessions, initialization, and so on...
        output = model(signal)
        self.assertEqual(output.shape, (hps.minibatch_size, FLAGS.sample_duration))

    def testSampleGivesCorrectShape(self):
        num_samples = 4
        model = SchrodingerRNN(hps)
        # Note that keras handles sessions, initialization, and so on...
        output = model.sample(num_samples=num_samples, sample_duration=FLAGS.sample_duration)
        self.assertEqual(output.shape, (num_samples, FLAGS.sample_duration))


if __name__ == '__main__':
      tf.test.main()