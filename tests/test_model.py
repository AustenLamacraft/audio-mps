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
            self.assertAllClose(model.H_diag, tf.conj(model.H_diag))


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

        test_H_diag = tf.constant(np.random.rand(hparams.bond_dim).astype(dtype=np.float32))
        model = RhoCMPS(hparams, H_in=test_H_diag)
        # TODO this test fails because one of the two, contains a trailing coma in the
        # TODO end, but otherwise the test is succesful.
        with self.cached_session() as sess:

            sess.run(tf.global_variables_initializer())
            self.assertAllClose(model.H_diag, test_H_diag)

    def testTrivialUpdateOfAncilla(self):
        """
        Update with H=R=0
        """

        test_H_diag = np.zeros([hparams.bond_dim], dtype=np.float32)
        test_R = np.zeros(2*[hparams.bond_dim], dtype=np.float32)
        # signal = np.random.rand(hparams.minibatch_size).astype(dtype=np.float32)
        signal_and_time = np.random.rand(hparams.minibatch_size+1).astype(dtype=np.float32)
        model = RhoCMPS(hparams, H_in=test_H_diag, Rx_in=test_R, Ry_in=test_R)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            stack_rho_0 = tf.stack(hparams.minibatch_size * [model.rho_0])
            updated_rho = model._update_ancilla_rho(stack_rho_0, signal_and_time)
            self.assertAllClose(stack_rho_0, updated_rho)

    def testPureRhoMatchPsi(self):

        hparams = HParams(minibatch_size=3,
                          bond_dim=2,
                          delta_t=0.01,
                          sigma=1,
                          initial_rank=1,
                          A=1,
                          h_reg=2 / (np.pi * FLAGS.sample_rate) ** 2,
                          r_reg=2 / (np.pi * FLAGS.sample_rate) ** 2, )

        test_H_diag = tf.constant([2., 4.], dtype=tf.float32)
        test_R = tf.constant([[1., 2.], [3., 4.]], dtype=tf.float32)
        test_psi_x = tf.constant([.7, 0.5], dtype=tf.float32)
        test_psi_y = tf.constant([4.1, 9.3], dtype=tf.float32)
        test_psi = tf.cast(test_psi_x, dtype=tf.complex64) + 1j*tf.cast(test_psi_y, dtype=tf.complex64)
        test_Wx = tf.reshape(test_psi_x, [1, 2])
        test_Wy = -tf.reshape(test_psi_y, [1, 2])
        test_W = tf.cast(test_Wx, dtype=tf.complex64) + 1j*tf.cast(test_Wy, dtype=tf.complex64)
        length = 10
        range_stack = tf.stack(hparams.minibatch_size * [tf.range(length, dtype=np.float32)])
        data = tf.cos((range_stack))

        model_rho = RhoCMPS(hparams, data_iterator=data,
                            H_in=test_H_diag,
                            Rx_in=test_R,
                            Ry_in=test_R,
                            Wx_in=test_Wx,
                            Wy_in=test_Wy)

        model_psi = PsiCMPS(hparams, data_iterator=data,
                            H_in=test_H_diag,
                            Rx_in=test_R,
                            Ry_in=test_R,
                            psi_x_in=test_psi_x,
                            psi_y_in=test_psi_y)

        # model_rho = RhoCMPS(hparams, data_iterator=data,
        #                     H_in=test_H_diag,
        #                     Rx_in=test_R,
        #                     Ry_in=test_R,
        #                     W_in=test_W)

        # model_psi = PsiCMPS(hparams, data_iterator=data,
        #                     H_in=test_H_diag,
        #                     Rx_in=test_R,
        #                     Ry_in=test_R,
        #                     psi_in=test_psi)



        with tf.Session() as sess:
        # with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            energy = -25.93724
            loss_psi = model_psi.loss
            loss_rho = model_rho.loss
            self.assertAllClose(loss_psi, energy)
            self.assertAllClose(loss_rho, energy)
            self.assertAllClose(loss_rho, loss_psi)

    def testnoislessSamplingPureRhoMatchPsi(self):

        hparams = HParams(minibatch_size=3,
                          bond_dim=2,
                          delta_t=0.01,
                          sigma=1,
                          initial_rank=1,
                          A=1,
                          h_reg=2 / (np.pi * FLAGS.sample_rate) ** 2,
                          r_reg=2 / (np.pi * FLAGS.sample_rate) ** 2, )

        test_H_diag = tf.constant([2., 4.], dtype=tf.float32)
        test_R = tf.constant([[1., 2.], [3., 4.]], dtype=tf.float32)
        test_psi_x = tf.constant([.7, 0.5], dtype=tf.float32)
        test_psi_y = tf.constant([4.1, 9.3], dtype=tf.float32)
        test_psi = tf.cast(test_psi_x, dtype=tf.complex64) + 1j * tf.cast(test_psi_y, dtype=tf.complex64)
        test_Wx = tf.reshape(test_psi_x, [1, 2])
        test_Wy = -tf.reshape(test_psi_y, [1, 2])
        test_W = tf.cast(test_Wx, dtype=tf.complex64) + 1j * tf.cast(test_Wy, dtype=tf.complex64)

        model_rho = RhoCMPS(hparams,
                            H_in=test_H_diag,
                            Rx_in=test_R,
                            Ry_in=test_R,
                            Wx_in=test_Wx,
                            Wy_in=test_Wy)

        model_psi = PsiCMPS(hparams,
                            H_in=test_H_diag,
                            Rx_in=test_R,
                            Ry_in=test_R,
                            psi_x_in=test_psi_x,
                            psi_y_in=test_psi_y)

        length = 10
        num_samples = 2
        temp = 0.0000000000001
        samples_rho = model_rho.sample_rho(num_samples, length, temp=temp)
        samples_psi = model_psi.sample_psi(num_samples, length, temp=temp)

        with tf.Session() as sess:
        # with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(samples_rho, samples_psi)



if __name__ == '__main__':
      tf.test.main()