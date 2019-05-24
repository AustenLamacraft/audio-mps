import tensorflow as tf
import numpy as np
from utils import symmetrize
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.regularizers import L1L2

class CMPSCell(tf.keras.layers.Layer):
    """
    Cell that will be fed into RNN class
    """

    def __init__(self, hparams, freqs_in=None, R_in=None, **kwargs):
        super(CMPSCell, self).__init__(**kwargs)

        self.bond_d = hparams.bond_dim
        self.batch_size = hparams.minibatch_size
        self.h_reg = hparams.h_reg
        self.r_reg = hparams.r_reg
        self.delta_t = hparams.delta_t
        self.dt = tf.constant(hparams.delta_t, tf.float32)  # Needed for increments

        self.A = hparams.A
        self.A = tf.get_variable("A", dtype=tf.float32, initializer=hparams.A)

        self.sigma = hparams.sigma
        # self.sigma = tf.get_variable("sigma", dtype=tf.float32, initializer=hparams.sigma)
        # self.sigma = tf.cast(self.sigma, dtype=tf.complex64)

        self.freqs_in = freqs_in
        self.R_in = R_in

    def build(self):

        # ======================================================
        # Training variables (cannot be complex)
        # ======================================================

        if self.R_in is not None:
            Rreal_init = Constant(self.R_in.real)
            Rimag_init = Constant(self.R_in.imag)
        else:
            Rreal_init = RandomNormal(stddev=tf.rsqrt(self.r_reg))
            Rimag_init = RandomNormal(stddev=tf.rsqrt(self.r_reg))

        Rx = self.add_variable("Rx", dtype=tf.float32, shape=2*[self.bond_d],
                               initializer=Rreal_init,
                               regularizer=L1L2(l2=self.r_reg))

        Ry = self.add_variable("Ry", dtype=tf.float32, shape=2*[self.bond_d],
                               initializer=Rimag_init,
                               regularizer=L1L2(l2=self.r_reg))

        self.R = tf.complex(Rx, Ry)
        self.R -= tf.matrix_diag_part(self.R)  # Remove diagonal part

        if self.freqs_in is not None:
            freqs_init = Constant(self.freqs_in)
        else:
            freqs_init = RandomNormal(stddev=tf.rsqrt(self.h_reg))

        self.freqs = self.add_variable("freqs", dtype=tf.float32, shape=[self.bond_d],
                                       initializer=freqs_init,
                                       regularizer=L1L2(l2=self.h_reg))

        self.freqsc = tf.cast(self.freqs, dtype=tf.complex64)
        self.built = True


class PsiCMPSCell(CMPSCell):
    def __init__(self, hparams, psi_in=None, *args, **kwargs):
        super(PsiCMPSCell, self).__init__(hparams, *args, **kwargs)
        self.psi_in = psi_in

    def build(self):
        super(PsiCMPSCell, self).build()

        if self.psi_in is not None:
            psi_real_init = Constant(self.psi_in.real)
            psi_imag_init = Constant(self.psi_in.real)
        else:
            psi_real_init = None
            psi_imag_init = None

        psi_x = self.add_variable("psi_x", shape=[self.bond_d],
                                  dtype=tf.float32, initializer=psi_real_init)
        psi_y = self.add_variable("psi_y", shape=[self.bond_d],
                                  dtype=tf.float32, initializer=psi_imag_init)

        self.psi_0 = tf.complex(psi_x, psi_y)
        self.psi_0 = self._normalize_psi(self.psi_0)  # No need of axis=1 because this is not a batch of psis

    def call(self, signal, psi_loss_t):
        psi, loss, t = psi_loss_t
        psi = self._update_ancilla_psi(psi, signal, t)
        loss += self._inc_loss_psi(psi, signal, t)
        psi = self._normalize_psi(psi, axis=1)
        t += self.dt
        return _, [psi, loss, t]

    def _inc_loss_psi(self, psi, signal, t):
        return - tf.log(1. + self._expectation(psi, t) * signal / self.A)

    def _update_ancilla_psi(self, psi, signal, t):
        # Note we do not normalize the state anymore in this method
        with tf.variable_scope("update_ancilla"):
            signal = tf.cast(signal / self.A, dtype=tf.complex64)
            t = tf.cast(t, dtype=tf.complex64)
            phases = tf.exp(1j * self.freqsc * t)
            Upsi = psi * tf.conj(phases)

            Rdag = tf.linalg.adjoint(self.R)
            RUpsi = tf.einsum('bc,ac->ab', self.R, Upsi)
            RdagRUpsi = tf.einsum('bc,ac->ab', Rdag, RUpsi)

            delta_Upsi = - self.delta_t * self.sigma**2 * RdagRUpsi / 2.
            delta_Upsi += tf.expand_dims(signal, axis=1) * RUpsi

            delta_psi = phases * delta_Upsi

            return psi + delta_psi

    def _expectation(self, psi, t):
        with tf.variable_scope("expectation"):
            t = tf.cast(t, dtype=tf.complex64)
            phases = tf.exp(1j * self.freqsc * t)
            Upsi = psi * tf.conj(phases)
            exp = tf.einsum('ab,bc,ac->a', tf.conj(Upsi), self.R, Upsi)
            return 2 * tf.real(exp)  # Conveniently returns a float

    def _normalize_psi(self, x, axis=None, epsilon=1e-12):
        #TODO change the method so that it ise clear that the argument axis changes whether we normalize a single psi
        #TODO or a batch of psis
        with tf.variable_scope("normalize"):
            square_sum = tf.reduce_sum(tf.square(tf.abs(x)), axis, keepdims=True)
            x_inv_norm = tf.rsqrt(tf.maximum(square_sum, epsilon))
            x_inv_norm = tf.cast(x_inv_norm, tf.complex64)
            return tf.multiply(x, x_inv_norm)

