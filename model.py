import tensorflow as tf

class AudioMPS:
    """
    Matrix Product State model for audio signal
    """

    def __init__(self, data_iterator, bond_d, delta_t):

        self.bond_d = bond_d
        self.delta_t = delta_t

        self.H = tf.get_variable("H", shape=[bond_d, bond_d], dtype=tf.complex64)
        self.H = self._hermitian_matrix(H)
        self.R = tf.get_variable("R", shape=[bond_d, bond_d], dtype=tf.complex64)

        self.loss = self._build_loss(data_iterator)


    def _build_loss(self, data):
        batch_size = data.shape.as_list()[0]
        psi_0 = tf.one_hot([0]*batch_size, self.bond_d)
        loss = tf.zeros(batch_size)
        data = data.transpose([0,1]) # Scan along the first dimension
        _, loss = tf.foldl(self._psi_and_loss_update, data, initializer=(psi_0, loss))
        # TODO Should the loss be divided by the length?
        return loss

    def _psi_and_loss_update(self, psi_and_loss, signal):
        psi, loss = psi_and_loss
        psi = self._update_ancilla(psi, signal)
        loss += self._inc_loss(psi, signal)
        return psi, loss

    def _inc_loss(self, psi, signal):
        return (signal - self._expectation(psi))**2 / 2

    def _update_ancilla(self, psi, signal):

        Q = self.delta_t * (-1j*self.H - tf.matmul(self.R, self.R, adjoint_a=True) / 2)
        new_psi = psi
        new_psi += tf.einsum('ab,cb->ca', Q, psi)
        new_psi += self.delta_t * tf.einsum('a,bc,ac->ab', signal, R, psi)
        new_psi = new_psi / tf.sqrt(tf.norm(new_psi, axis=1))
        return new_psi

    def _expectation(self, psi):
        exp =  tf.einsum('ab,bc,ac->a', tf.conj(psi), self.R, psi)
        return 2 * tf.real(exp)

    def _hermitian_matrix(self, M):
        M_lower = tf.matrix_band_part(M, -1, 0)  # takes the lower triangular part of M (including the diagonal)
        return M_lower + tf.matrix_transpose(M_lower)