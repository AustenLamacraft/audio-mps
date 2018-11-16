import tensorflow as tf
import numpy as np


class AudioMPS:
    """
    Matrix Product State model for audio signal.
    The argument mixed discriminates between two models:
        mixed=False: dynamics of a pure quantum state
        mixed=True: dynamics of a density matrix
    """

    def __init__(self, bond_d, delta_t, batch_size, data_iterator=None, H_in=None, R_in=None, rho_0_in=None, mixed=False):

        self.bond_d = bond_d
        self.delta_t = delta_t
        self.batch_size = batch_size

        #=========================
        # Initial state, if mixed
        #=========================

        if mixed:
            if rho_0_in is not None:
                self.rho_0 = rho_0_in
            else:
                self.rho_0 = (1. / self.bond_d) * tf.eye(self.bond_d, dtype=tf.complex64)

        #======================================================
        # Inital values for parameters to be learned, if given
        #======================================================

        if R_in is not None:
            self.R = tf.get_variable("R", dtype=tf.float32,
                                     initializer=R_in)
        else:
            self.R = tf.get_variable("R", shape=[bond_d, bond_d], dtype=tf.float32,
                                     initializer=None)

        if H_in is not None:
            self.H = tf.get_variable("H", dtype=tf.float32,
                                     initializer=H_in)
        else:
            self.H = tf.get_variable("H", shape=[bond_d, bond_d], dtype=tf.float32,
                                     initializer=None)

        self.H = self._symmetrize(self.H)

        #=====================================
        # Loss function object, if data given
        #=====================================

        if data_iterator is not None:
            if mixed:
                self.loss = self._build_loss_rho(data_iterator)
            else:
                self.loss = self._build_loss_psi(data_iterator)

    #====================
    # Rho methods-PUBLIC
    #====================

    def rho_evolve_with_data(self, num_samples, data):
        batch_zeros = tf.zeros([num_samples])
        rho_0 = tf.stack(num_samples * [self.rho_0])
        data = tf.transpose(data, [1, 0])
        rho, _ = tf.scan(self._rho_update, data,
                               initializer=(rho_0, batch_zeros), name="rho_scan_data_evolved")
        return rho


    def rho_evolve_with_sampling(self, num_samples, length, temp=1):
        batch_zeros = tf.zeros([num_samples])
        rho_0 = tf.stack(num_samples * [self.rho_0])
        noise = tf.random_normal([length, num_samples], stddev=np.sqrt(temp / self.delta_t))
        rho, samples = tf.scan(self._rho_and_sample_update, noise,
                               initializer=(rho_0, batch_zeros), name="rho_scan")
        return rho

    def purity(self, num_samples, length, temp=1):
        batch_zeros = tf.zeros([num_samples])
        rho_0 = tf.stack(num_samples * [self.rho_0])
        noise = tf.random_normal([length, num_samples], stddev=np.sqrt(temp / self.delta_t))
        rho, samples = tf.scan(self._rho_and_sample_update, noise,
                               initializer=(rho_0, batch_zeros), name="purity_scan")
        return tf.real(tf.transpose(tf.trace(tf.einsum('abcd,abde->abce', rho, rho)), [1, 0]))

    def sample_rho(self, num_samples, length, temp=1):
        batch_zeros = tf.zeros([num_samples])
        rho_0 = tf.stack(num_samples * [self.rho_0])
        noise = tf.random_normal([length, num_samples], stddev=np.sqrt(temp / self.delta_t))
        rho, samples = tf.scan(self._rho_and_sample_update, noise,
                               initializer=(rho_0, batch_zeros), name="sample_scan")
        # TODO The use of tf.scan here must have some inefficiency as we keep all the intermediate psi values
        return tf.transpose(samples, [1, 0])

    def sample_time_evolved_rho0(self, num_samples, length, data, temp=1):
        """The data is only used to time evolve one step"""
        batch_zeros = tf.zeros([num_samples])
        rho_0 = tf.stack(num_samples * [self.rho_0])
        data = tf.transpose(data, [1, 0])  # foldl goes along the first dimension
        rho_0 = self._update_ancilla_rho(rho_0, data[0])
        noise = tf.random_normal([length, num_samples], stddev=np.sqrt(temp / self.delta_t))
        rho, samples = tf.scan(self._rho_and_sample_update, noise,
                               initializer=(rho_0, batch_zeros), name="sample_scan_time_evolved")
        # TODO The use of tf.scan here must have some inefficiency as we keep all the intermediate psi values
        return tf.transpose(samples, [1, 0])

    #====================
    # Psi methods-PUBLIC
    #====================

    def psi_evolve_with_data(self, num_samples, data):
        batch_zeros = tf.zeros([num_samples])
        psi_0 = tf.one_hot(tf.cast(batch_zeros, dtype=tf.int32), self.bond_d, dtype=tf.complex64)
        data = tf.transpose(data, [1, 0])
        psi, _ = tf.scan(self._psi_update, data,
                               initializer=(psi_0, batch_zeros), name="psi_scan_data_evolved")
        return psi

    def sample_time_evolved_psi0(self, num_samples, length, data, temp=1):
        batch_zeros = tf.zeros([num_samples])
        psi_0 = tf.one_hot(tf.cast(batch_zeros, dtype=tf.int32), self.bond_d, dtype=tf.complex64)
        data = tf.transpose(data, [1, 0])  # foldl goes along the first dimension
        psi_0 = self._update_ancilla_psi(psi_0, data[0])
        noise = tf.random_normal([length, num_samples], stddev=np.sqrt(temp / self.delta_t))
        psi, samples = tf.scan(self._psi_and_sample_update, noise,
                               initializer=(psi_0, batch_zeros), name="sample_scan")
        # TODO The use of tf.scan here must have some inefficiency as we keep all the intermediate psi values
        return tf.transpose(samples, [1, 0])

    #=====================
    # Psi methods-PRIVATE
    #=====================

    def _build_loss_psi(self, data):
        batch_zeros = tf.zeros_like(data[:, 0])
        psi_0 = tf.one_hot(tf.cast(batch_zeros, dtype=tf.int32), self.bond_d, dtype=tf.complex64)
        loss = batch_zeros
        data = tf.transpose(data, [1, 0])  # foldl goes along the first dimension
        psi_0 = self._update_ancilla_psi(psi_0, data[0])
        data = data[1:]
        _, loss = tf.foldl(self._psi_and_loss_update, data,
                           initializer=(psi_0, loss), name="loss_fold")
        return tf.reduce_mean(loss)

    def _psi_update(self, psi_and_loss, signal):
        psi, loss = psi_and_loss
        psi = self._update_ancilla_psi(psi, signal)
        return psi, loss

    def _psi_and_loss_update(self, psi_and_loss, signal):
        psi, loss = psi_and_loss
        loss += self._inc_loss_psi(psi, signal)
        psi = self._update_ancilla_psi(psi, signal)
        return psi, loss

    def _psi_and_sample_update(self, psi_and_sample, noise):
        psi, last_sample = psi_and_sample
        new_sample = self._expectation_psi(psi) + noise
        psi = self._update_ancilla_psi(psi, new_sample)
        return psi, new_sample

    def _inc_loss_psi(self, psi, signal):
        return (signal - self._expectation_psi(psi)) ** 2 / 2

    def _update_ancilla_psi(self, psi, signal):
        with tf.variable_scope("update_ancilla"):
            signal = tf.cast(signal, dtype=tf.complex64)
            H_c = tf.cast(self.H, dtype=tf.complex64)
            R_c = tf.cast(self.R, dtype=tf.complex64)
            Q = self.delta_t * (-1j * H_c - tf.matmul(R_c, R_c, transpose_a=True) / 2)
            new_psi = psi
            new_psi += tf.einsum('ab,cb->ca', Q, psi)
            new_psi += self.delta_t * tf.einsum('a,bc,ac->ab', signal, R_c, psi)
            new_psi = self._normalize_psi(new_psi, axis=1)
            return new_psi

    def _expectation_psi(self, psi):
        with tf.variable_scope("expectation"):
            R_c = tf.cast(self.R, dtype=tf.complex64)
            exp = tf.einsum('ab,bc,ac->a', tf.conj(psi), R_c, psi)
            return 2 * tf.real(exp)  # Conveniently returns a float

    def _normalize_psi(self, x, axis=None, epsilon=1e-12):
        with tf.variable_scope("normalize"):
            square_sum = tf.reduce_sum(tf.square(tf.abs(x)), axis, keepdims=True)
            x_inv_norm = tf.rsqrt(tf.maximum(square_sum, epsilon))
            x_inv_norm = tf.cast(x_inv_norm, tf.complex64)
            return tf.multiply(x, x_inv_norm)

    #=====================
    # Rho methods-PRIVATE
    #=====================

    def _build_loss_rho(self, data):
        batch_zeros = tf.zeros_like(data[:, 0])
        rho_0 = tf.stack(self.batch_size * [self.rho_0])
        loss = batch_zeros
        data = tf.transpose(data, [1, 0])  # foldl goes along the 1st dimension
        rho_0 = self._update_ancilla_rho(rho_0, data[0])
        data = data[1:]
        _, loss = tf.foldl(self._rho_and_loss_update, data,
                           initializer=(rho_0, loss), name="loss_fold")
        return tf.reduce_mean(loss)

    def _rho_update(self, rho_and_loss, signal):
        rho, loss = rho_and_loss
        rho = self._update_ancilla_rho(rho, signal)
        return rho, loss

    def _rho_and_loss_update(self, rho_and_loss, signal):
        rho, loss = rho_and_loss
        loss += self._inc_loss_rho(rho, signal)
        rho = self._update_ancilla_rho(rho, signal)
        return rho, loss

    def _rho_and_sample_update(self, rho_and_sample, noise):
        rho, last_sample = rho_and_sample
        new_sample = self._expectation_rho(rho) + noise
        rho = self._update_ancilla_rho(rho, new_sample)
        return rho, new_sample

    def _inc_loss_rho(self, rho, signal):
        return (signal - self._expectation_rho(rho)) ** 2 / 2

    def _update_ancilla_rho(self, rho, signal):
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
            new_rho = self._normalize_rho(new_rho)
            return new_rho

    def _expectation_rho(self, rho):
        with tf.variable_scope("expectation"):
            R_c = tf.cast(self.R, dtype=tf.complex64)
            x = tf.add(R_c, tf.linalg.adjoint(R_c))
            exp = tf.trace(tf.einsum('ab,cbd->cad', x, rho))
            return tf.real(exp)  # Conveniently returns a float

    def _normalize_rho(self, x, epsilon=1e-12):
        with tf.variable_scope("normalize"):
            tr = tf.reshape(tf.trace(x), [tf.shape(x)[0], 1, 1])
            x_inv_tr = tf.reciprocal(tf.maximum(tf.real(tr), epsilon))
            x_inv_tr = tf.cast(x_inv_tr, tf.complex64)
            return tf.multiply(x, x_inv_tr)

    #=====================================
    # Methods that don't depend on rho/psi
    #=====================================

    def _symmetrize(self, M):
        with tf.variable_scope("symmetrize"):
            M_lower = tf.matrix_band_part(M, -1, 0)
            M_diag = tf.matrix_band_part(M, 0, 0)
            return M_lower + tf.matrix_transpose(M_lower) - M_diag