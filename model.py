import tensorflow as tf
import numpy as np
from utils import symmetrize

class CMPS:
    """
    Continuous Matrix Product State.
    """
    def __init__(self, hparams, data_iterator=None, H_in=None, Rx_in=None, Ry_in=None):
        #TODO change name of H_in to something more informative
        self.bond_d = hparams.bond_dim
        self.batch_size = hparams.minibatch_size
        self.h_reg = hparams.h_reg
        self.r_reg = hparams.r_reg
        self.delta_t = hparams.delta_t

        self.A = hparams.A
        # self.A = tf.get_variable("A", dtype=tf.float32, initializer=hparams.A)
        # self.A = tf.cast(self.A, dtype=tf.complex64)

        self.sigma = hparams.sigma
        # self.sigma = tf.get_variable("sigma", dtype=tf.float32, initializer=hparams.sigma)
        # self.sigma = tf.cast(self.sigma, dtype=tf.complex64)

        self.data_iterator = data_iterator

        #======================================================
        # Training variables (cannot be complex)
        #======================================================

        if Rx_in is not None and Ry_in is not None:

            self.Rx = tf.get_variable("Rx", dtype=tf.float32, initializer=Rx_in)
            self.Ry = tf.get_variable("Ry", dtype=tf.float32, initializer=Ry_in)
        else:

            self.Rx = tf.rsqrt(self.r_reg) * tf.get_variable("rx", shape=2*[self.bond_d], dtype=tf.float32,
                                                             initializer=tf.random_normal_initializer)
            self.Ry = tf.rsqrt(self.r_reg) * tf.get_variable("ry", shape=2*[self.bond_d], dtype=tf.float32,
                                                             initializer=tf.random_normal_initializer)

        if H_in is not None:

            self.H_diag = tf.get_variable("H_diag", dtype=tf.float32, initializer=H_in)
        else:

            self.H_diag = tf.rsqrt(self.h_reg) * tf.get_variable("h_diag", shape=[self.bond_d], dtype=tf.float32,
                                                                 initializer=tf.random_normal_initializer)

        self.R = tf.cast(self.Rx, dtype=tf.complex64) + 1j * tf.cast(self.Ry, dtype=tf.complex64)


class RhoCMPS(CMPS):
    """
        Evolves the density matrix
    """
    def __init__(self, hparams, Wx_in=None, Wy_in=None, *args, **kwargs):
        with tf.variable_scope("RhoCMPS"):
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
        # TODO this method has not been tested
        batch_zeros = tf.zeros([num_samples])
        rho_0 = tf.stack(num_samples * [self.rho_0])
        # We switch to increments
        data = data[:, 1:] - data[:, :-1]
        # We now introduce a time vector and combine it with the data
        length = tf.shape(data)[1]
        time = tf.cast([tf.range(length) + 1], dtype=tf.float32) * self.delta_t
        # Time will appear at the end of each batch vector, at each time step.
        # So we extract it by splitting up each batch vector inside _update_ancilla_rho
        data_and_time = tf.concat([data, time], axis=0)
        data_and_time = tf.transpose(data_and_time, [1, 0])  # foldl goes along the 1st dimension
        rho, _ = tf.scan(self._rho_update, data_and_time,
                         initializer=(rho_0, batch_zeros), name="rho_scan_data_evolved")
        return rho


    def rho_evolve_with_sampling(self, num_samples, length, temp=1):
        # TODO this method has not been tested
        batch_zeros = tf.zeros([num_samples])
        rho_0 = tf.stack(num_samples * [self.rho_0])

        time = tf.cast([tf.range(length) + 1], dtype=tf.float32) * self.delta_t
        # noise needs to be [num_samples, length] for concatenation with time
        noise = tf.random_normal([num_samples, length], stddev=self.A * self.sigma * np.sqrt(temp * self.delta_t))
        noise_and_time = tf.concat([noise, time], axis=0)
        noise_and_time = tf.transpose(noise_and_time) # scan goes along the 1st dimension

        rho, samples = tf.scan(self._rho_and_sample_update, noise_and_time,
                               initializer=(rho_0, batch_zeros), name="rho_scan")
        return rho

    def purity(self, num_samples, length, temp=1):
        # TODO this method has not been tested
        batch_zeros = tf.zeros([num_samples])
        rho_0 = tf.stack(num_samples * [self.rho_0])
        time = tf.cast([tf.range(length) + 1], dtype=tf.float32) * self.delta_t
        # noise needs to be [num_samples, length] for concatenation with time
        noise = tf.random_normal([num_samples, length], stddev=self.A * self.sigma * np.sqrt(temp * self.delta_t))
        noise_and_time = tf.concat([noise, time], axis=0)
        noise_and_time = tf.transpose(noise_and_time) # scan goes along the 1st dimension
        rho, samples = tf.scan(self._rho_and_sample_update, noise_and_time,
                               initializer=(rho_0, batch_zeros), name="purity_scan")
        return tf.real(tf.transpose(tf.trace(tf.einsum('abcd,abde->abce', rho, rho)), [1, 0]))

    def sample_rho(self, num_samples, length, temp=1):
        # Note we sample X_t and not increments (X_(t+1) - X_t)
        batch_zeros = tf.zeros([num_samples])
        rho_0 = tf.stack(num_samples * [self.rho_0])

        time = tf.cast([tf.range(length) + 1], dtype=tf.float32) * self.delta_t
        # noise needs to be [num_samples, length] for concatenation with time
        noise = tf.random_normal([num_samples, length], stddev=self.A * self.sigma * np.sqrt(temp * self.delta_t))
        noise_and_time = tf.concat([noise, time], axis=0)
        noise_and_time = tf.transpose(noise_and_time) # scan goes along the 1st dimension

        rho, samples = tf.scan(self._rho_and_sample_update, noise_and_time,
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
        # We now introduce a time vector and combine it with the data
        length = tf.shape(data)[1]
        time = tf.cast([tf.range(length) + 1], dtype=tf.float32) * self.delta_t
        # Time will appear at the end of each batch vector, at each time step.
        # So we extract it by splitting up each batch vector inside _update_ancilla_rho
        data_and_time = tf.concat([data, time], axis=0)
        data_and_time = tf.transpose(data_and_time, [1, 0])  # foldl goes along the 1st dimension
        _, loss = tf.foldl(self._rho_and_loss_update, data_and_time,
                           initializer=(rho_0, loss), name="loss_fold")
        return tf.reduce_mean(loss)

    def _rho_update(self, rho_and_loss, signal_and_time):
        # TODO this method has not been tested
        # TODO change the name of the first argument
        rho, loss = rho_and_loss
        rho = self._update_ancilla_rho(rho, signal_and_time) # signal is the increment
        rho = self._normalize_rho(rho)
        return rho, loss

    def _rho_and_loss_update(self, rho_and_loss, signal_and_time):
        rho, loss = rho_and_loss
        rho = self._update_ancilla_rho(rho, signal_and_time)
        loss += self._inc_loss_rho(rho)
        rho = self._normalize_rho(rho)
        return rho, loss

    def _rho_and_sample_update(self, rho_and_sample, noise_and_time):
        #TODO introduce _and_time notation
        noise = noise_and_time[:-1]  # The last element is time
        time = noise_and_time[-1]
        rho, last_sample = rho_and_sample
        # Construct R(t) = exp[iHt] R exp[-iHt], where H is diagonal. Then, R_ab(t)=R_ab exp[i(wa-wb)t].
        # We define the matrix delta_w with elements (delta_w)_ab = wa-wb.
        delta_w = tf.transpose(tf.stack(self.bond_d * [self.H_diag])) - tf.stack(self.bond_d * [self.H_diag])
        delta_w, time = tf.cast(delta_w, dtype=tf.complex64), tf.cast(time, dtype=tf.complex64)
        Rt = tf.exp(1j * delta_w * time) * self.R  # This is elementwise multiplication, as it should be.
        # NEW SAMPLE, note expectation depends on time now
        new_sample = last_sample + self._expectation_RplusRdag_rho(rho, Rt) * self.A * self.delta_t + noise
        increment = new_sample - last_sample
        # Concatenate incremement with time
        increment_and_time = tf.concat([increment, [time]], axis=0)
        rho = self._update_ancilla_rho(rho, increment_and_time) # Note update with increment
        rho = self._normalize_rho(rho)
        return rho, new_sample

    def _inc_loss_rho(self, rho):
        return - tf.log(tf.real(tf.trace(rho)))

    def _update_ancilla_rho(self, rho, signal_and_time):
        # Note we do not normalize the state anymore in this method
        with tf.variable_scope("update_ancilla"):
            # split signal and time
            signal = signal_and_time[:-1] # The last element is time
            time = signal_and_time[-1]
            # Construct R(t) = exp[iHt] R exp[-iHt], where H is diagonal. Then, R_ab(t)=R_ab exp[i(wa-wb)t].
            # We define the matrix delta_w with elements (delta_w)_ab = wa-wb.
            delta_w = tf.transpose(tf.stack(self.bond_d * [self.H_diag])) - tf.stack(self.bond_d * [self.H_diag])
            delta_w, time = tf.cast(delta_w, dtype=tf.complex64), tf.cast(time, dtype=tf.complex64)
            Rt = tf.exp(1j * delta_w * time) * self.R # This is elementwise multiplication, as it should be.
            #####
            signal = tf.cast(signal, dtype=tf.complex64)
            batch_size = rho.shape[0]
            RR_dag = tf.matmul(Rt, Rt, adjoint_a=True)
            RR_dag = tf.stack(batch_size * [RR_dag])
            IR = tf.einsum('a,bc->abc', signal, Rt)
            one = tf.stack(batch_size * [tf.eye(self.bond_d, dtype=tf.complex64)])
            U = one - 0.5 * RR_dag * self.delta_t * self.sigma ** 2 + IR / self.A
            U_dag = tf.linalg.adjoint(U)
            new_rho = tf.einsum('abc,acd,ade->abe', U, rho, U_dag)
            return new_rho

    def _expectation_RplusRdag_rho(self, rho, Rt):
        with tf.variable_scope("expectation"):
            x = tf.add(Rt, tf.linalg.adjoint(Rt))
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

    def __init__(self, hparams, psi_x_in=None, psi_y_in=None, *args, **kwargs):
        with tf.variable_scope("PsiCMPS"):
            super(PsiCMPS, self).__init__(hparams, *args, **kwargs)

            if psi_x_in is not None and psi_y_in is not None:
                self.psi_x = tf.get_variable("psi_x", dtype=tf.float32, initializer=psi_x_in)
                self.psi_y = tf.get_variable("psi_y", dtype=tf.float32, initializer=psi_y_in)
            else:
                self.psi_x = tf.get_variable("psi_x", shape=[self.bond_d], dtype=tf.float32, initializer=None)
                self.psi_y = tf.get_variable("psi_y", shape=[self.bond_d], dtype=tf.float32, initializer=None)

            self.psi_0 = tf.cast(self.psi_x, dtype=tf.complex64) + 1j * tf.cast(self.psi_y, dtype=tf.complex64)
            self.psi_0 = self._normalize_psi(self.psi_0) # No need of axis=1 because this is not a batch of psis

        if self.data_iterator is not None:
            self.loss = self._build_loss_psi(self.data_iterator)

    # ====================
    # Psi methods-PUBLIC
    # ====================

    def psi_evolve_with_data(self, num_samples, data):
        # TODO this method has not been tested
        batch_zeros = tf.zeros([num_samples])
        psi_0 = tf.stack(num_samples * [self.psi_0])

        # We switch to increments
        data = data[:, 1:] - data[:, :-1]
        # We now introduce a time vector and combine it with the data
        length = tf.shape(data)[1]
        time = tf.cast([tf.range(length) + 1], dtype=tf.float32) * self.delta_t
        # Time will appear at the end of each batch vector, at each time step.
        # So we extract it by splitting up each batch vector inside _update_ancilla_rho
        data_and_time = tf.concat([data, time], axis=0)
        data_and_time = tf.transpose(data_and_time, [1, 0])  # foldl goes along the 1st dimension

        psi, _ = tf.scan(self._psi_update, data_and_time,
                         initializer=(psi_0, batch_zeros), name="psi_scan_data_evolved")
        return psi

    def sample_psi(self, num_samples, length, temp=1):
        # Note we sample X_t and not increments (X_(t+1) - X_t)
        batch_zeros = tf.zeros([num_samples])
        psi_0 = tf.stack(num_samples * [self.psi_0])
        time = tf.cast([tf.range(length) + 1], dtype=tf.float32) * self.delta_t
        # noise needs to be [num_samples, length] for concatenation with time
        noise = tf.random_normal([num_samples, length], stddev=self.A * self.sigma * np.sqrt(temp * self.delta_t))
        noise_and_time = tf.concat([noise, time], axis=0)
        noise_and_time = tf.transpose(noise_and_time)  # scan goes along the 1st dimension
        psi, samples = tf.scan(self._psi_and_sample_update, noise_and_time,
                               initializer=(psi_0, batch_zeros), name="sample_scan")
        # TODO The use of tf.scan here must have some inefficiency as we keep all the intermediate psi values
        # TODO check batch_zeros is the right initializer. I think it is if I define X_0 = 0.
        return tf.transpose(samples, [1, 0])

    # =====================
    # Psi methods-PRIVATE
    # =====================

    def _build_loss_psi(self, data):
        batch_zeros = tf.zeros_like(data[:, 0])
        psi_0 = tf.stack(self.batch_size * [self.psi_0])
        loss = batch_zeros
        # We switch to increments
        data = data[:, 1:] - data[:, :-1]
        # We now introduce a time vector and combine it with the data
        length = tf.shape(data)[1]
        time = tf.cast([tf.range(length) + 1], dtype=tf.float32) * self.delta_t
        # Time will appear at the end of each batch vector, at each time step.
        # So we extract it by splitting up each batch vector inside _update_ancilla_rho
        data_and_time = tf.concat([data, time], axis=0)
        data_and_time = tf.transpose(data_and_time, [1, 0])  # foldl goes along the 1st dimension
        _, loss = tf.foldl(self._psi_and_loss_update, data_and_time,
                           initializer=(psi_0, loss), name="loss_fold")
        return tf.reduce_mean(loss)


    def _psi_update(self, psi_and_loss, signal_and_time):
        # TODO this method has not been tested
        # TODO change name of first argument
        psi, loss = psi_and_loss
        psi = self._update_ancilla_psi(psi, signal_and_time) # signal is the increment
        psi = self._normalize_psi(psi, axis=1)
        return psi, loss

    def _psi_and_loss_update(self, psi_and_loss, signal_and_time):
        psi, loss = psi_and_loss
        psi = self._update_ancilla_psi(psi, signal_and_time)
        loss += self._inc_loss_psi(psi)
        psi = self._normalize_psi(psi, axis=1)
        return psi, loss

    def _psi_and_sample_update(self, psi_and_sample, noise_and_time):

        noise = noise_and_time[:-1]  # The last element is time
        time = noise_and_time[-1]
        psi, last_sample = psi_and_sample
        # Construct R(t) = exp[iHt] R exp[-iHt], where H is diagonal. Then, R_ab(t)=R_ab exp[i(wa-wb)t].
        # We define the matrix delta_w with elements (delta_w)_ab = wa-wb.
        delta_w = tf.transpose(tf.stack(self.bond_d * [self.H_diag])) - tf.stack(self.bond_d * [self.H_diag])
        delta_w, time = tf.cast(delta_w, dtype=tf.complex64), tf.cast(time, dtype=tf.complex64)
        Rt = tf.exp(1j * delta_w * time) * self.R  # This is elementwise multiplication, as it should be.
        # NEW SAMPLE, note expectation depends on time now
        new_sample = last_sample + self._expectation_RplusRdag_psi(psi, Rt) * self.A * self.delta_t + noise
        increment = new_sample - last_sample
        # Concatenate incremement with time
        increment_and_time = tf.concat([increment, [time]], axis=0)
        psi = self._update_ancilla_psi(psi, increment_and_time)  # Note update with increment
        psi = self._normalize_psi(psi, axis=1)
        return psi, new_sample

    def _inc_loss_psi(self, psi):
        return - tf.log(self._norm_square_psi(psi))

    def _norm_square_psi(self, psi):
        exp = tf.einsum('ab,ab->a', tf.conj(psi), psi)
        return tf.real(exp)

    def _update_ancilla_psi(self, psi, signal_and_time):
        # Note we do not normalize the state anymore in this method
        with tf.variable_scope("update_ancilla"):
            # split signal and time
            signal = signal_and_time[:-1]  # The last element is time
            time = signal_and_time[-1]
            # this all the same
            signal, time = tf.cast(signal, dtype=tf.complex64), tf.cast(time, dtype=tf.complex64)
            batch_size = psi.shape[0]
            R = tf.stack(batch_size * [self.R])
            one = tf.stack(batch_size * [tf.eye(self.bond_d, dtype=tf.complex64)])
            IR = tf.einsum('a,bc->abc', signal, self.R)
            R_dag = tf.linalg.adjoint(R)
            # Construct exp[iHt], where H is diagonal.
            expiHt = tf.stack(batch_size * [tf.exp(1j * tf.cast(self.H_diag, dtype=tf.complex64) * time)])
            #
            psi = tf.conj(expiHt) * psi
            Rpsi = tf.einsum('abc,ac->ab', R, psi)
            RRdagpsi = - 0.5 * self.delta_t * self.sigma ** 2 * tf.einsum('abc,ac->ab', R_dag, Rpsi)
            U_partial = one + IR / self.A
            Upartialpsi = tf.einsum('abc,ac->ab', U_partial, psi)
            new_psi = Upartialpsi + RRdagpsi
            new_psi = expiHt * new_psi
            return new_psi

    def _expectation_RplusRdag_psi(self, psi, Rt):
        with tf.variable_scope("expectation"):
            exp = tf.einsum('ab,bc,ac->a', tf.conj(psi), Rt, psi)
            return 2 * tf.real(exp)

    def _normalize_psi(self, x, axis=None, epsilon=1e-12):
        #TODO change the method so that it ise clear that the argument axis changes whether we normalize a single psi
        #TODO or a batch of psis
        with tf.variable_scope("normalize"):
            square_sum = tf.reduce_sum(tf.square(tf.abs(x)), axis, keepdims=True)
            x_inv_norm = tf.rsqrt(tf.maximum(square_sum, epsilon))
            x_inv_norm = tf.cast(x_inv_norm, tf.complex64)
            return tf.multiply(x, x_inv_norm)