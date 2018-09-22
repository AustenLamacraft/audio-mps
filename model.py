import tensorflow as tf
import numpy as np


class AudioMPS:
    """
    Matrix Product State model for audio signal
    """

    def __init__(self, bond_d, delta_t, batch_size, data_iterator=None, h0=None, r0=None):

        self.bond_d = bond_d
        self.delta_t = delta_t
        self.batch_size = batch_size

        if r0 is not None:
            self.R = tf.get_variable("R", dtype=tf.float32,
                                     initializer=r0, trainable=False)
        else:
            self.R = tf.get_variable("R", shape=[bond_d, bond_d], dtype=tf.float32,
                                     initializer=None)

        if h0 is not None:
            self.H = tf.get_variable("H", dtype=tf.float32,
                                     initializer=h0, trainable=False)
        else:
            self.H = tf.get_variable("H", shape=[bond_d, bond_d], dtype=tf.float32,
                                     initializer=None)

        self.H = self._symmetrize(self.H)

        if data_iterator is not None:
            self.loss = self._build_loss(data_iterator)

    def _build_loss(self, data):
        batch_zeros = tf.zeros_like(data[:, 0])  # data[note,time]
        rho_0 = tf.stack(self.batch_size * [(1. / self.bond_d) * tf.eye(self.bond_d, dtype=tf.complex64)])
        loss = batch_zeros
        data = tf.transpose(data, [1, 0])  # foldl goes along the first dimension
        _, loss = tf.foldl(self._rho_and_loss_update, data,
                           initializer=(rho_0, loss), name="loss_fold")
        return tf.reduce_mean(loss)

    def sample(self, num_samples, length, temp=1):
        batch_zeros = tf.zeros([num_samples])
        rho_0 = tf.stack(num_samples * [(1. / self.bond_d) * tf.eye(self.bond_d, dtype=tf.complex64)])
        noise = tf.random_normal([length, num_samples], stddev=np.sqrt(temp / self.delta_t))
        rho, samples = tf.scan(self._rho_and_sample_update, noise,
                               initializer=(rho_0, batch_zeros), name="sample_scan")
        # TODO The use of tf.scan here must have some inefficiency as we keep all the intermediate psi values
        return tf.transpose(samples, [1, 0])

    def _rho_and_sample_update(self, rho_and_sample, noise):
        rho, last_sample = rho_and_sample
        new_sample = self._expectation(rho) + noise
        rho = self._update_ancilla(rho, new_sample)
        return rho, new_sample

    def _rho_and_loss_update(self, rho_and_loss, signal):
        rho, loss = rho_and_loss  # these come from initializer=(rho_0, loss), in _build_loss
        loss += self._inc_loss(rho, signal)
        rho = self._update_ancilla(rho, signal)
        return rho, loss

    def _inc_loss(self, rho, signal):
        return (signal - self._expectation(rho)) ** 2 / 2

    def _update_ancilla(self, rho, signal):  # rho has dimensions (n_samp,D,D)
        with tf.variable_scope("update_ancilla"):
            signal = tf.cast(signal, dtype=tf.complex64)
            H_c = tf.cast(self.H, dtype=tf.complex64)
            R_c = tf.cast(self.R, dtype=tf.complex64)
            Q = self.delta_t * (-1j * H_c - tf.matmul(R_c, R_c, transpose_a=True) / 2)
            oneQdagdt = tf.add(tf.eye(self.bond_d, dtype=tf.complex64), tf.linalg.adjoint(Q))
            IRdagdt = self.delta_t * tf.einsum('a,bc->abc', signal, tf.linalg.adjoint(R_c))
            new_rho = rho
            new_rho += tf.einsum('ab,cbd->cad', Q, rho)
            new_rho += self.delta_t * tf.einsum('a,bc,ace->abe', signal, R_c, rho)
            a1 = tf.einsum('abe,ec->abc', new_rho, oneQdagdt)
            a2 = tf.einsum('abe,aec->abc', new_rho, IRdagdt)
            new_rho = a1 + a2
            new_rho = self._normalize(new_rho)
            return new_rho

    def _expectation(self, rho):
        with tf.variable_scope("expectation"):
            R_c = tf.cast(self.R, dtype=tf.complex64)
            x = tf.add(R_c, tf.linalg.adjoint(R_c))
            exp = tf.trace(tf.einsum('ab,cbd->cad', x, rho))
            return tf.real(exp)  # Conveniently returns a float

    def _symmetrize(self, M):
        with tf.variable_scope("symmetrize"):
            M_lower = tf.matrix_band_part(M, -1, 0)  # takes the lower triangular part of M (including the diagonal)
            M_diag = tf.matrix_band_part(M, 0, 0)
            return M_lower + tf.matrix_transpose(M_lower) - M_diag

    def _normalize(self, x, epsilon=1e-12):
        with tf.variable_scope("normalize"):
            tr = tf.reshape(tf.trace(x), [tf.shape(x)[0], 1, 1])
            x_inv_tr = tf.reciprocal(tf.maximum(tf.real(tr), epsilon))
            x_inv_tr = tf.cast(x_inv_tr, tf.complex64)
            return tf.multiply(x, x_inv_tr)
