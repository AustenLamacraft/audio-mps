import tensorflow as tf
import numpy as np
from utils import symmetrize, normalize
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.regularizers import L1L2

class CMPSCell(tf.keras.layers.Layer):
    """
    Cell that will be fed into RNN class
    """

    def __init__(self, hparams, freqs_in=None, R_in=None, **kwargs):
        super().__init__(**kwargs)

        self.bond_d = hparams.bond_dim

        self.h_scale = hparams.h_scale
        self.r_scale = hparams.r_scale
        self.h_reg = hparams.h_reg
        self.r_reg = hparams.r_reg
        self.delta_t = hparams.delta_t

        self.A_in = hparams.A
        self.sigma = hparams.sigma

        self.freqs_in = freqs_in
        self.R_in = R_in

    def build(self, _): # Build requires input_shape argument

        # ======================================================
        # Training variables (cannot be complex)
        # ======================================================

        A_init = Constant(self.A_in)
        self.A = self.add_variable("A", dtype=tf.float32, shape=[1], initializer=A_init)

        if self.R_in is not None:
            Rreal_init = Constant(self.R_in.real)
            Rimag_init = Constant(self.R_in.imag)
        else:
            Rreal_init = RandomNormal(stddev=self.r_scale)
            Rimag_init = RandomNormal(stddev=self.r_scale)

        Rx = self.add_variable("Rx", dtype=tf.float32, shape=2*[self.bond_d],
                               initializer=Rreal_init,
                               regularizer=L1L2(l2=self.r_reg))

        Ry = self.add_variable("Ry", dtype=tf.float32, shape=2*[self.bond_d],
                               initializer=Rimag_init,
                               regularizer=L1L2(l2=self.r_reg))

        self.R = tf.complex(Rx, Ry)
        self.R -= tf.matrix_diag(tf.matrix_diag_part(self.R))  # Remove diagonal part

        if self.freqs_in is not None:
            freqs_init = Constant(self.freqs_in)
        else:
            freqs_init = RandomNormal(stddev=self.h_scale)

        self.freqs = self.add_variable("freqs", dtype=tf.float32, shape=[self.bond_d],
                                       initializer=freqs_init,
                                       regularizer=L1L2(l2=self.h_reg))

        self.built = True


class PsiCMPSCell(CMPSCell):
    def __init__(self, hparams, psi_in=None, *args, **kwargs):
        super().__init__(hparams, *args, **kwargs)

        self.psi_in = psi_in
        self.state_size = (hparams.bond_dim, 1)
        self.output_size = 1

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """
        State consists of ancilla state and last signal value.
        """
        if self.psi_in is not None:
            psi_real_init = Constant(self.psi_in.real)
            psi_imag_init = Constant(self.psi_in.real)
        else:
            psi_real_init = None
            psi_imag_init = None

        psi_x = self.add_variable("psi_x", shape=[self.bond_d],
                                  dtype=dtype, initializer=psi_real_init)
        psi_y = self.add_variable("psi_y", shape=[self.bond_d],
                                  dtype=dtype, initializer=psi_imag_init)

        psi_0 = tf.complex(psi_x, psi_y)
        psi_0 = normalize(psi_0)  # No need of axis=1 because this is not a batch of psis
        psi_0 = tf.expand_dims(psi_0, axis=0)
        psi_0 = tf.tile(psi_0, [batch_size, 1])
        # state = tf.stack(batch_size * [psi_0]) # This doesn't work when batch_size is a tensor
        x_0 = tf.zeros((batch_size), dtype=tf.float32)
        return [psi_0, x_0]

    def call(self, x_t, state, training=True):
        """
        At training time, inputs are values and times.
        At sampling, inputs are noise values and times
        """
        psi = state[0]
        lastx = tf.cast(state[1], dtype=tf.float32)
        x = x_t[:, 0]
        t = x_t[:, 1]
        if training:
            psi = self._update_ancilla(psi, x - lastx, t)
            psi = normalize(psi, axis=1)
            # Prediction for next value
            output = x + self.A * self._expectation(psi, t) * self.delta_t
            return output, [psi, x]
        else:
            # For sampling x is noise
            inc = x + self.A * self._expectation(psi, t) * self.delta_t
            nextx = lastx + inc
            psi = self._update_ancilla(psi, inc, t)
            psi = normalize(psi, axis=1)
            return nextx, [psi, nextx]

    def _update_ancilla(self, psi, inc, t):
        with tf.variable_scope("update_ancilla"):
            inc = tf.cast(inc, dtype=tf.complex64)
            t = tf.cast(t, dtype=tf.complex64)
            freqsc = tf.cast(self.freqs, dtype=tf.complex64)
            phases = tf.exp(1j * tf.einsum('a,b->ab', t, freqsc))
            Upsi = psi * tf.conj(phases)

            Rdag = tf.linalg.adjoint(self.R)
            RUpsi = tf.einsum('bc,ac->ab', self.R, Upsi)
            RdagRUpsi = tf.einsum('bc,ac->ab', Rdag, RUpsi)
            # I guess we could do this in one go and einsum would figure out the correct way...

            delta_Upsi = - self.delta_t * self.sigma**2 * RdagRUpsi / 2.
            delta_Upsi += tf.expand_dims(inc, axis=1) * RUpsi

            delta_psi = phases * delta_Upsi

            return psi + delta_psi

    def _expectation(self, psi, t):
        with tf.variable_scope("expectation"):
            t = tf.cast(t, dtype=tf.complex64)
            freqsc = tf.cast(self.freqs, dtype=tf.complex64)
            psi = normalize(tf.ones_like(psi, dtype=tf.complex64), axis=1)
            phases = tf.exp(1j * tf.einsum('a,b->ab', t, freqsc))
            Upsi = psi * tf.conj(phases)
            exp = tf.einsum('ab,bc,ac->a', tf.conj(Upsi), self.R, Upsi)
            return 2 * tf.real(exp)  # Conveniently returns a float


class StochasticSchrodinger(tf.keras.layers.RNN):
    def __init__(self, hparams, **kwargs):
        cell = PsiCMPSCell(hparams)
        self.delta_t = tf.constant(hparams.delta_t, tf.float32)

        super().__init__(cell, return_sequences=True,
                         return_state=False, **kwargs)  # Note that batch major is the default

    def call(self, inputs, training=True):
        """
        Takes signal and returns predictions for next value
        """
        time = tf.cast(tf.range(inputs.shape[1], dtype=tf.int32), dtype=tf.float32) * self.delta_t
        batch_size = inputs.shape[0]
        time = tf.expand_dims(time, axis=0)
        time = tf.tile(time, [batch_size, 1])
        # time = tf.stack(batch_size * [time]) # Doesn't work when batch_size a tensor
        rnn_inputs = tf.stack([inputs, time], axis=2)
        output = super().call(rnn_inputs)
        return output


class SchrodingerRNN(tf.keras.Model):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.sigma = hparams.sigma
        self.delta_t = hparams.delta_t
        self.sse = StochasticSchrodinger(hparams)

    def call(self, signal):
        predictions = self.sse(signal, training=True)
        # Must match predictions with the right input
        predictions = tf.concat([signal[:, :1], predictions[:, :-1]], axis=1)
        return predictions

    def sample(self, num_samples, sample_duration):
        noise = tf.random_normal([num_samples, sample_duration],
                                 stddev=self.sigma * np.sqrt(self.delta_t))

        samples = self.sse(noise, training=False)
        return samples