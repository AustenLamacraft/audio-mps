import tensorflow as tf
import numpy as np
from keras_model import CMPSCell, PsiCMPSCell
from data import get_audio

from tensorflow.contrib.training import HParams


tf.flags.DEFINE_integer("sample_duration", 2**8, "Duration of samples (as integer).")
tf.flags.DEFINE_integer("sample_rate", 16000, "Sampling rate.")
FLAGS = tf.flags.FLAGS

hparams = HParams(minibatch_size=8, bond_dim=7, delta_t=1/FLAGS.sample_rate, sigma=0.0001, initial_rank=None, A=100.,
                  h_reg=2/(np.pi * FLAGS.sample_rate)**2, r_reg=2/(np.pi * FLAGS.sample_rate))


class TestCMPSCell(tf.test.TestCase):

    def testRHasNoDiagonalElements(self):

        model = CMPSCell(hparams)
        model.build()

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(tf.matrix_diag_part(model.R.eval()), hparams.bond_dim * [0.])

class TestPsiCMPSCell(tf.test.TestCase):

    def testTrivialUpdatOfAncilla(self):

        test_freqs = np.zeros([hparams.bond_dim], dtype=np.float32)
        test_R = np.zeros(2*[hparams.bond_dim], dtype=np.complex64)
        signal = np.random.rand(hparams.minibatch_size).astype(dtype=np.float32)
        model = PsiCMPSCell(hparams, freqs_in=test_freqs, R_in=test_R)

        model.build()

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            stack_psi_0 = tf.stack(hparams.minibatch_size * [model.psi_0])

            updated_psi = model._update_ancilla_psi(stack_psi_0, signal, 0.)
            self.assertAllClose(stack_psi_0, updated_psi)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(tf.matrix_diag_part(model.R.eval()), hparams.bond_dim * [0.])



if __name__ == '__main__':
      tf.test.main()