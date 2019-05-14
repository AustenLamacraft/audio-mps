import tensorflow as tf
import tfplot

def symmetrize(M):
    with tf.variable_scope("symmetrize"):
        M_lower = tf.matrix_band_part(M, -1, 0)
        M_diag = tf.matrix_band_part(M, 0, 0)
        return M_lower + tf.matrix_transpose(M_lower) - M_diag

def waveform_plot(waveform):
    fig, ax = tfplot.subplots(figsize=(3, 3))
    ax.plot(waveform)
    return fig
