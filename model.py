import tensorflow as tf
import numpy as np

class AudioMPS:
    """
    Matrix Product State model for audio signal
    """

    def __init__(self, bond_d, delta_t, data_iterator=None,
                 num_samples=0, sample_length=1000):

        self.bond_d = bond_d
        self.delta_t = delta_t

        self.H = tf.get_variable("H", shape=[bond_d, bond_d], dtype=tf.float32,
                                 initializer=None)
        self.H = self._symmetrize(self.H)
        self.R = tf.get_variable("R", shape=[bond_d, bond_d], dtype=tf.float32,
                                 initializer=None)

        if data_iterator is not None:
            self.loss = self._build_loss(data_iterator)

        if num_samples != 0:
            self.sample = self._sample(num_samples, sample_length)

    def _sample(self, num_samples, length):
        batch_zeros = tf.zeros([num_samples])
        psi_0 = tf.one_hot(tf.cast(batch_zeros, dtype=tf.int32), self.bond_d, dtype=tf.complex64)
        noise = tf.random_normal([length, num_samples], stddev=np.sqrt(self.delta_t))
        psi, samples = tf.scan(self._psi_and_sample_update, noise,
                             initializer=(psi_0, batch_zeros), name="sample_scan")
        # TODO The use of tf.scan here must have some inefficiency as we keep all the intermediate psi values
        return psi, tf.transpose(samples, [1,0])

    def _build_loss(self, data):
        data = data[:,1:] - data[:,:-1] # Take the derivative
        batch_zeros = tf.zeros_like(data[:,0])
        psi_0 = tf.one_hot(tf.cast(batch_zeros, dtype=tf.int32), self.bond_d, dtype=tf.complex64)
        loss = batch_zeros
        data = tf.transpose(data, [1,0]) # foldl goes along the first dimension
        _, loss = tf.foldl(self._psi_and_loss_update, data,
                           initializer=(psi_0, loss), name="loss_fold")
        # TODO Should the loss be divided by the length?
        return tf.reduce_mean(loss)

    def _psi_and_loss_update(self, psi_and_loss, signal):
        psi, loss = psi_and_loss
        psi = self._update_ancilla(psi, signal)
        loss += self._inc_loss(psi, signal)
        return psi, loss

    def _psi_and_sample_update(self, psi_and_sample, noise):
        psi, last_sample = psi_and_sample
        psi = self._update_ancilla(psi, noise / self.delta_t)
        new_sample = last_sample + noise + self._expectation(psi)
        return psi, new_sample

    def _inc_loss(self, psi, signal):
        return (signal - self._expectation(psi))**2 / 2

    def _update_ancilla(self, psi, signal):
        with tf.variable_scope("update_ancilla"):
            signal = tf.cast(signal, dtype=tf.complex64)
            H_c = tf.cast(self.H, dtype=tf.complex64)
            R_c = tf.cast(self.R, dtype=tf.complex64)
            Q = self.delta_t * (-1j * H_c - tf.matmul(R_c, R_c, transpose_a=True) / 2)
            new_psi = psi
            new_psi += tf.einsum('ab,cb->ca', Q, psi)
            new_psi += self.delta_t * tf.einsum('a,bc,ac->ab', signal, R_c, psi)
            new_psi = self._normalize(psi, axis=1)
            return new_psi

    def _expectation(self, psi):
        with tf.variable_scope("expectation"):
            R_c = tf.cast(self.R, dtype=tf.complex64)
            exp = tf.einsum('ab,bc,ac->a', tf.conj(psi), R_c, psi)
            return 2 * tf.real(exp) # Conveniently returns a float

    def _symmetrize(self, M):
        with tf.variable_scope("symmetrize"):
            M_lower = tf.matrix_band_part(M, -1, 0)  # takes the lower triangular part of M (including the diagonal)
            return M_lower + tf.matrix_transpose(M_lower)

    def _normalize(self, x, axis=None, epsilon=1e-12):
        with tf.variable_scope("normalize"):
            square_sum = tf.reduce_sum(tf.square(tf.abs(x)), axis, keepdims=True)
            x_inv_norm = tf.rsqrt(tf.maximum(square_sum, epsilon))
            x_inv_norm = tf.cast(x_inv_norm, tf.complex64)
            return tf.multiply(x, x_inv_norm)