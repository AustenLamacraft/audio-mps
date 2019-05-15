import tensorflow as tf
import numpy as np
from utils import symmetrize

class CMPS:
    """
    Continuous Matrix Product State.
    """

    # def __init__(self, bond_d, delta_t, batch_size, data_iterator=None, H=None, R=None):
    #
    #     self.bond_d = bond_d
    #     self.delta_t = delta_t
    #     self.batch_size = batch_size
    #     self.data_iterator = data_iterator
    #
    #     # TODO Switch to complex
    #     # TODO Accommodate diagonal
    #
    #     if R is not None:
    #         self.R = tf.get_variable("R", dtype=tf.float32,
    #                                  initializer=R)
    #     else:
    #         self.R = tf.get_variable("R", shape=[bond_d, bond_d], dtype=tf.float32,
    #                                  initializer=None)
    #
    #     if H is not None:
    #         self.H = tf.get_variable("H", dtype=tf.float32,
    #                                  initializer=H)
    #     else:
    #         self.H = tf.get_variable("H", shape=[bond_d, bond_d], dtype=tf.float32,
    #                                  initializer=None)
    #
    #     self.H = symmetrize(self.H)

    def __init__(self, bond_d, h_reg, r_reg, Asgdt, batch_size, data_iterator=None, H_in=None, Rx_in=None, Ry_in=None):

        self.bond_d = bond_d
        self.batch_size = batch_size
        self.rank_rho_0 = bond_d
        self.h_reg = h_reg
        self.r_reg = r_reg
        self.Asgdt = Asgdt

        #======================================================
        # Inital values for parameters to be learned, if given
        #======================================================

        # Training variables cannot be complex

        if Rx_in is not None and Ry_in is not None:

            self.Rx = tf.get_variable("Rx", dtype=tf.float32, initializer=Rx_in)
            self.Ry = tf.get_variable("Ry", dtype=tf.float32, initializer=Ry_in)
        else:

            self.Rx = tf.get_variable("Rx", shape=[bond_d, bond_d], dtype=tf.float32, initializer=None)
            self.Ry = tf.get_variable("Ry", shape=[bond_d, bond_d], dtype=tf.float32, initializer=None)

        if H_in is not None:

            self.H_diag = tf.get_variable("H_diag", dtype=tf.float32, initializer=H_in)
        else:

            self.H_diag = tf.get_variable("H_diag", shape=[bond_d], dtype=tf.float32, initializer=None)


        self.Rx = tf.cast(self.Rx, dtype=tf.complex64)
        self.Ry = tf.cast(self.Ry, dtype=tf.complex64)
        self.R = self.Rx + 1j * self.Ry
        self.H = tf.cast(tf.diag(self.H_diag), dtype=tf.complex64)

        #=====================================
        # Loss function object, if data given
        #=====================================

        if data_iterator is not None:
            if mixed:
                self.loss = self._build_loss_rho(data_iterator)
            else:
                self.loss = self._build_loss_psi(data_iterator)


class RhoCMPS(CMPS):
    """
        Evolves the density matrix
    """
    def __init__(self, Wx_in=None, Wy_in=None, *args, **kwargs):
        super(RhoCMPS, self).__init__(*args, **kwargs)


        if Wx_in is not None and Wy_in is not None:
            self.Wx = tf.get_variable("Wx", dtype=tf.float32, initializer=Wx_in)
            self.Wy = tf.get_variable("Wy", dtype=tf.float32, initializer=Wy_in)
        else:
            self.Wx = tf.get_variable("Wx", shape=[self.rank_rho_0, bond_d], dtype=tf.float32, initializer=None)
            self.Wy = tf.get_variable("Wy", shape=[self.rank_rho_0, bond_d], dtype=tf.float32, initializer=None)

        self.Wx = tf.cast(self.Wx, dtype=tf.complex64)
        self.Wy = tf.cast(self.Wy, dtype=tf.complex64)
        self.W = self.Wx + 1j * self.Wy
        self.rho_0 = tf.matmul(self.W, self.W, adjoint_a=True)
        self.rho_0 = self.rho_0 / tf.trace(self.rho_0)

        if self.data_iterator is not None:
            self.loss = self._build_loss_rho(self.data_iterator)

    # ====================
    # Rho methods-PUBLIC
    # ====================

    # TODO all the public methods are from the old version

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
        noise = tf.random_normal([length, num_samples], stddev=np.sqrt(temp))
        rho, samples = tf.scan(self._rho_and_sample_update, noise,
                               initializer=(rho_0, batch_zeros), name="rho_scan")
        return rho

    def purity(self, num_samples, length, temp=1):
        batch_zeros = tf.zeros([num_samples])
        rho_0 = tf.stack(num_samples * [self.rho_0])
        noise = tf.random_normal([length, num_samples], stddev=np.sqrt(temp))
        rho, samples = tf.scan(self._rho_and_sample_update, noise,
                               initializer=(rho_0, batch_zeros), name="purity_scan")
        return tf.real(tf.transpose(tf.trace(tf.einsum('abcd,abde->abce', rho, rho)), [1, 0]))

    def sample_rho(self, num_samples, length, temp=1):
        batch_zeros = tf.zeros([num_samples])
        rho_0 = tf.stack(num_samples * [self.rho_0])
        noise = tf.random_normal([length, num_samples], stddev=np.sqrt(temp))
        rho, samples = tf.scan(self._rho_and_sample_update, noise,
                               initializer=(rho_0, batch_zeros), name="sample_scan")
        # TODO The use of tf.scan here must have some inefficiency as we keep all the intermediate psi values
        return tf.transpose(samples, [1, 0])

    # =====================
    # Rho methods-PRIVATE
    # =====================

    def _build_loss_rho(self, data):
        batch_zeros = tf.zeros_like(data[:, 0])
        rho_0 = tf.stack(self.batch_size * [self.rho_0])
        loss = batch_zeros
        # We switch to increments
        data = data[:, 1:] - data[:, :-1]
        data = tf.transpose(data, [1, 0])  # foldl goes along the 1st dimension
        _, loss = tf.foldl(self._rho_and_loss_update, data,
                           initializer=(rho_0, loss), name="loss_fold")
        return tf.reduce_mean(loss) + self.h_reg * tf.reduce_sum(tf.square(self.H_diag)) + \
               self.r_reg * tf.real(tf.reduce_sum(tf.conj(self.R)*self.R))

    def _rho_update(self, rho_and_loss, signal):
        # TODO change the name of the first argument
        rho, loss = rho_and_loss
        rho = self._update_ancilla_rho(rho, signal)
        rho = self._normalize_rho(rho)
        return rho, loss

    def _rho_and_loss_update(self, rho_and_loss, signal):
        rho, loss = rho_and_loss
        rho = self._update_ancilla_rho(rho, signal)
        loss += self._inc_loss_rho(rho, signal)
        rho = self._normalize_rho(rho)
        return rho, loss

    def _rho_and_sample_update(self, rho_and_sample, noise):
        #TODO think how to do the sampling
        rho, last_sample = rho_and_sample
        new_sample = self._expectation_rho(rho) + noise
        rho = self._update_ancilla_rho(rho, new_sample)
        return rho, new_sample

    def _inc_loss_rho(self, rho, signal):
        return - tf.log(self._expectation_rho(rho))

    def _update_ancilla_rho(self, rho, signal):
        # Note we do not normalize the state anymore in this method
        with tf.variable_scope("update_ancilla"):
            signal = tf.cast(signal, dtype=tf.complex64)
            num_samples = tf.size(signal)
            H_tile = tf.reshape(tf.tile(self.H, [num_samples, 1]), [num_samples, self.bond_d, self.bond_d])
            RR_dag = tf.matmul(self.R, self.R, adjoint_a=True)
            RR_dag_tile = tf.reshape(tf.tile(RR_dag, [num_samples, 1]), [num_samples, self.bond_d, self.bond_d])
            IR = tf.einsum('a,bc->abc', signal, self.R)
            one_tile = tf.reshape(tf.tile(tf.eye(self.bond_d, dtype=tf.complex64), [num_samples, 1]),
                                  [num_samples, self.bond_d, self.bond_d])
            U = one_tile + (-1j * H_tile - 0.5 * RR_dag_tile + IR / self.Asgdt)
            U_dag = tf.linalg.adjoint(U)
            new_rho = tf.einsum('abc,acd,ade->abe', U, rho, U_dag)
            return new_rho

    def _expectation_rho(self, rho):
        with tf.variable_scope("expectation"):
            x = tf.add(self.R, tf.linalg.adjoint(self.R))
            exp = tf.trace(tf.einsum('ab,cbd->cad', x, rho))
            return tf.real(exp)

    def _normalize_rho(self, x, epsilon=1e-12):
        with tf.variable_scope("normalize"):
            tr = tf.reshape(tf.trace(x), [tf.shape(x)[0], 1, 1])
            x_inv_tr = tf.reciprocal(tf.maximum(tf.real(tr), epsilon))
            x_inv_tr = tf.cast(x_inv_tr, tf.complex64)
            return tf.multiply(x, x_inv_tr)


class PsiCMPS(CMPS):
    """
        Evolves the state
    """
    # TODO everything I have done for rho, I have not touched PsiCMPS class

    def __init__(self, *args, **kwargs):
        super(PsiCMPS, self).__init__(*args, **kwargs)
        if self.data_iterator is not None:
            self.loss = self._build_loss_psi(data_iterator)

    # ====================
    # Psi methods-PUBLIC
    # ====================

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

    # =====================
    # Psi methods-PRIVATE
    # =====================

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
            # Q = self.delta_t * (-1j * H_c - tf.matmul(R_c, R_c, transpose_a=True) / 2)
            # new_psi = psi
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
