import tensorflow as tf

class AudioMPS:
    """
    Matrix Product State model for audio signal
    """

    def __init__(self, data_iterator, bond_d, delta_t):

        self.bond_d = bond_d
        self.delta_t = delta_t
        self.psi = tf.one_hot(1, bond_d, dtype=tf.complex64)

        self.H = tf.get_variable("H", shape=[bond_d, bond_d], dtype=tf.complex64)
        self.H = self._hermitian_matrix(H)
        self.R = tf.get_variable("R", shape=[bond_d, bond_d], dtype=tf.complex64)

        self.loss = self._build_loss(data_iterator)


    def _build_loss(self, data):

        return loss


    def _update_ancilla(self, current_psi, signal):

        update_matrix = self.delta_t * (-1j*self.H - tf.matmul(self.R, self.R, adjoint_a=True) / 2)
        update_matrix += self.delta_t * signal * self.R
        new_psi = current_psi + tf.einsum('ab,cb->ca', update_matrix, current_psi)
        new_psi = new_psi / tf.sqrt(tf.norm(new_psi, axis=1))
        return new_psi

    def _expectation(self, psi):
        exp =  tf.einsum('ab,bc,ac->a', tf.conj(psi), self.R, psi)
        return 2 * tf.real(exp)

    def _hermitian_matrix(self, M):
        M_lower = tf.matrix_band_part(M, -1, 0)  # takes the lower triangular part of M (including the diagonal)
        return M_lower + tf.matrix_transpose(M_lower)