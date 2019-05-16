import tensorflow as tf
import numpy as np
from utils import symmetrize

class CMPS:
    """
    Continuous Matrix Product State.
    """
    def __init__(self, hparams, data_iterator=None, H_in=None, Rx_in=None, Ry_in=None):

        self.bond_d = hparams.bond_dim
        self.batch_size = hparams.minibatch_size
        self.h_reg = hparams.h_reg
        self.r_reg = hparams.r_reg
        self.sigma = hparams.sigma
        self.A = hparams.A
        self.delta_t = hparams.delta_t

        self.data_iterator = data_iterator

        #======================================================
        # Training variables (cannot be complex)
        #======================================================

        if Rx_in is not None and Ry_in is not None:

            self.Rx = tf.get_variable("Rx", dtype=tf.float32, initializer=Rx_in)
            self.Ry = tf.get_variable("Ry", dtype=tf.float32, initializer=Ry_in)
        else:

            self.Rx = tf.get_variable("Rx", shape=2*[self.bond_d], dtype=tf.float32, initializer=None)
            self.Ry = tf.get_variable("Ry", shape=2*[self.bond_d], dtype=tf.float32, initializer=None)

        if H_in is not None:

            self.H_diag = tf.get_variable("H_diag", dtype=tf.float32, initializer=H_in)
        else:

            self.H_diag = tf.get_variable("H_diag", shape=[self.bond_d], dtype=tf.float32, initializer=None)

        self.R = tf.cast(self.Rx, dtype=tf.complex64) + 1j * tf.cast(self.Ry, dtype=tf.complex64)
        self.H = tf.cast(tf.diag(self.H_diag), dtype=tf.complex64)

class RhoCMPS(CMPS):
    """
        Evolves the density matrix
    """
    def __init__(self, hparams, Wx_in=None, Wy_in=None, *args, **kwargs):
        super(RhoCMPS, self).__init__(hparams, *args, **kwargs)

        if hparams.initial_rank is not None:
            self.rank_rho_0 = hparams.initial_rank
        else:
            self.rank_rho_0 = hparams.bond_dim

        if Wx_in is not None and Wy_in is not None:
            self.Wx = tf.get_variable("Wx", dtype=tf.float32, initializer=Wx_in)
            self.Wy = tf.get_variable("Wy", dtype=tf.float32, initializer=Wy_in)
        else:
            self.Wx = tf.get_variable("Wx", shape=[self.rank_rho_0, self.bond_d], dtype=tf.float32, initializer=None)
            self.Wy = tf.get_variable("Wy", shape=[self.rank_rho_0, self.bond_d], dtype=tf.float32, initializer=None)

        self.rho_0 = self._rho_init()

        if self.data_iterator is not None:
            self.loss = self._build_loss_rho(self.data_iterator)

    # ====================
    # Rho methods-PUBLIC
    # ====================

    def rho_evolve_with_data(self, num_samples, data):
        batch_zeros = tf.zeros([num_samples])
        rho_0 = tf.stack(num_samples * [self.rho_0])
        # We switch to increments to evolve rho
        data = data[:, 1:] - data[:, :-1]
        data = tf.transpose(data, [1, 0])
        rho, _ = tf.scan(self._rho_update, data,
                         initializer=(rho_0, batch_zeros), name="rho_scan_data_evolved")
        return rho

    def rho_evolve_with_sampling(self, num_samples, length, temp=1):
        batch_zeros = tf.zeros([num_samples])
        rho_0 = tf.stack(num_samples * [self.rho_0])
        noise = tf.random_normal([length, num_samples], stddev=self.A * self.sigma * np.sqrt(temp * self.delta_t))
        rho, samples = tf.scan(self._rho_and_sample_update, noise,
                               initializer=(rho_0, batch_zeros), name="rho_scan")
        return rho

    def purity(self, num_samples, length, temp=1):
        batch_zeros = tf.zeros([num_samples])
        rho_0 = tf.stack(num_samples * [self.rho_0])
        noise = tf.random_normal([length, num_samples], stddev=self.A * self.sigma * np.sqrt(temp * self.delta_t))
        rho, samples = tf.scan(self._rho_and_sample_update, noise,
                               initializer=(rho_0, batch_zeros), name="purity_scan")
        return tf.real(tf.transpose(tf.trace(tf.einsum('abcd,abde->abce', rho, rho)), [1, 0]))

    def sample_rho(self, num_samples, length, temp=1):
        batch_zeros = tf.zeros([num_samples])
        rho_0 = tf.stack(num_samples * [self.rho_0])
        noise = tf.random_normal([length, num_samples], stddev=self.A * self.sigma * np.sqrt(temp * self.delta_t))
        rho, samples = tf.scan(self._rho_and_sample_update, noise,
                               initializer=(rho_0, batch_zeros), name="sample_scan")
        # TODO The use of tf.scan here must have some inefficiency as we keep all the intermediate psi values
        # TODO check batch_zeros is the right initializer. I think it is if I define X_0 = 0.
        return tf.transpose(samples, [1, 0])

    # =====================
    # Rho methods-PRIVATE
    # =====================

    def _rho_init(self):
        Wx = tf.cast(self.Wx, dtype=tf.complex64)
        Wy = tf.cast(self.Wy, dtype=tf.complex64)
        W = Wx + 1j * Wy
        rho_0 = tf.matmul(W, W, adjoint_a=True)
        rho_0 = rho_0 / tf.trace(rho_0)
        return rho_0

    def _build_loss_rho(self, data):
        batch_zeros = tf.zeros_like(data[:, 0])
        rho_0 = tf.stack(self.batch_size * [self.rho_0])
        loss = batch_zeros
        # We switch to increments
        data = data[:, 1:] - data[:, :-1]
        data = tf.transpose(data, [1, 0])  # foldl goes along the 1st dimension
        _, loss = tf.foldl(self._rho_and_loss_update, data,
                           initializer=(rho_0, loss), name="loss_fold")
        L2_regularization = self.h_reg * tf.reduce_sum(tf.square(self.H_diag)) + \
               self.r_reg * tf.real(tf.reduce_sum(tf.conj(self.R)*self.R))
        return tf.reduce_sum(loss) + L2_regularization

    def _rho_update(self, rho_and_loss, signal):
        # TODO change the name of the first argument
        rho, loss = rho_and_loss
        rho = self._update_ancilla_rho(rho, signal) # signal is the increment
        rho = self._normalize_rho(rho)
        return rho, loss

    def _rho_and_loss_update(self, rho_and_loss, signal):
        rho, loss = rho_and_loss
        rho = self._update_ancilla_rho(rho, signal)
        loss += self._inc_loss_rho(rho)
        rho = self._normalize_rho(rho)
        return rho, loss

    def _rho_and_sample_update(self, rho_and_sample, noise):
        rho, last_sample = rho_and_sample
        new_sample = last_sample + self._expectation_rho(rho) * self.A * self.delta_t + noise
        increment = new_sample - last_sample
        rho = self._update_ancilla_rho(rho, increment) # Note update with increment
        rho = self._normalize_rho(rho)
        return rho, new_sample

    def _inc_loss_rho(self, rho):
        return - tf.log(tf.real(tf.trace(rho)))

    def _update_ancilla_rho(self, rho, signal):
        # Note we do not normalize the state anymore in this method
        with tf.variable_scope("update_ancilla"):
            signal = tf.cast(signal, dtype=tf.complex64)
            H = tf.stack(self.batch_size * [self.H])
            RR_dag = tf.matmul(self.R, self.R, adjoint_a=True)
            RR_dag = tf.stack(self.batch_size * [RR_dag])
            IR = tf.einsum('a,bc->abc', signal, self.R)
            one = tf.stack(self.batch_size * [tf.eye(self.bond_d, dtype=tf.complex64)])
            U = one + (-1j * H * self.delta_t - 0.5 * RR_dag * self.delta_t * self.sigma**2 + IR / self.A)
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

    def __init__(self, hparams, psi_x_in=None, psi_y_in=None, *args, **kwargs):
        super(PsiCMPS, self).__init__(hparams, *args, **kwargs)

        if psi_x_in is not None and psi_y_in is not None:
            self.psi_x = tf.get_variable("psi_x", dtype=tf.float32, initializer=psi_x_in)
            self.psi_y = tf.get_variable("psi_y", dtype=tf.float32, initializer=psi_y_in)
        else:
            self.psi_x = tf.get_variable("psi_x", shape=[self.bond_d], dtype=tf.float32, initializer=None)
            self.psi_y = tf.get_variable("psi_y", shape=[self.bond_d], dtype=tf.float32, initializer=None)

        self.psi_0 = tf.cast(self.psi_x, dtype=tf.complex64) + 1j * tf.cast(self.psi_y, dtype=tf.complex64)
        self.psi_0 = self._normalize_psi(self.psi_0)

        if self.data_iterator is not None:
            self.loss = self._build_loss_psi(self.data_iterator)

    # ====================
    # Psi methods-PUBLIC
    # ====================

    def psi_evolve_with_data(self, num_samples, data):
        batch_zeros = tf.zeros([num_samples])

        # TODO change to learned psi_0
        psi_0 = tf.one_hot(tf.cast(batch_zeros, dtype=tf.int32), self.bond_d, dtype=tf.complex64)

        #TODO introduce increments
        data = tf.transpose(data, [1, 0])
        psi, _ = tf.scan(self._psi_update, data,
                         initializer=(psi_0, batch_zeros), name="psi_scan_data_evolved")
        return psi

    def sample_time_evolved_psi0(self, num_samples, length, temp=1):
        batch_zeros = tf.zeros([num_samples])
        # TODO change to learned psi_0
        psi_0 = tf.one_hot(tf.cast(batch_zeros, dtype=tf.int32), self.bond_d, dtype=tf.complex64)
        # TODO change to learned psi_0
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
