import tensorflow as tf
import numpy as np
import tfplot

def symmetrize(M):
    with tf.variable_scope("symmetrize"):
        M_lower = tf.matrix_band_part(M, -1, 0)
        return M_lower + tf.matrix_transpose(M_lower) - tf.diag_part(M)

def normalize(x, axis=None, epsilon=1e-12):
    #TODO change the method so that it ise clear that the argument axis changes whether we normalize a single psi
    #TODO or a batch of psis
    with tf.variable_scope("normalize"):
        square_sum = tf.reduce_sum(tf.square(tf.abs(x)), axis, keepdims=True)
        x_inv_norm = tf.rsqrt(tf.maximum(square_sum, epsilon))
        x_inv_norm = tf.cast(x_inv_norm, tf.complex64)
    return tf.multiply(x, x_inv_norm)

def waveform_plot(waveform, delta_t):
    fig, ax = tfplot.subplots(figsize=(3, 3))
    times = np.arange(len(waveform)) * delta_t
    ax.plot(times, waveform)
    ax.set_ylabel('signal')
    ax.set_xlabel('time')

    return fig

