import tensorflow as tf
import numpy as np
import tfplot

from tensorflow.contrib.training import HParams
from model import RhoCMPS, PsiCMPS
from data import get_audio
from utils import waveform_plot

DTYPE=tf.float32
NP_DTYPE=np.float32

tf.set_random_seed(0)

FLAGS = tf.flags.FLAGS

# Model flags
tf.flags.DEFINE_enum('mps_model', 'rho_mps',
                     ['rho_mps', 'psi_mps'],
                     'MPS mdoel. Must be one of "rho_mps" or "psi_mps".')

# Training flags
tf.flags.DEFINE_enum('dataset', 'damped_sine',
                     ['damped_sine', 'guitar', 'organ', 'nsynth'],
                     'Dataset. Must be one of "damped_sine", "guitar", "organ", or "nsynth".')

tf.flags.DEFINE_integer("sample_duration", 2**16, "Duration of samples (as integer).")
tf.flags.DEFINE_integer("sample_rate", 16000, "Sampling rate.")
tf.flags.DEFINE_boolean('visualize', True, 'Produce visualization. Probably Slow!')
tf.flags.DEFINE_string("hparams", "", 'Comma separated list of "name=value" pairs e.g. "--hparams=learning_rate=0.3"')
tf.flags.DEFINE_string("datadir", "./data", "Data directory.")
tf.flags.DEFINE_string("logdir", f"../logging/audio_mps/{FLAGS.dataset}", "Directory to write logs.")


def main(argv):
    # I introduced the hyperparameter A, which is set to 1. by default. The meaning of sigma is not Asgsq{dt} anymore.
    # sigma now is the strength of the noise.
    hparams = HParams(minibatch_size=8, bond_dim=8, delta_t=1/FLAGS.sample_rate, sigma=0.001,
                      h_reg=2/(np.pi * FLAGS.sample_rate)**2, r_reg=2/(np.pi * FLAGS.sample_rate)**2,
                      initial_rank=None, A=1)
    hparams.parse(FLAGS.hparams)

    with tf.variable_scope("data"):
        data = get_audio(datadir=FLAGS.datadir, dataset=FLAGS.dataset, hps=hparams)

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        if FLAGS.mps_model == 'rho_mps':
            model = RhoCMPS(hparams=hparams, data_iterator=data)
        else:
            model = PsiCMPS(hparams=hparams, data_iterator=data)

        h_l2sqnorm = tf.reduce_sum(tf.square(model.H_diag))
        r_l2sqnorm = tf.real(tf.reduce_sum(tf.conj(model.R) * model.R))

    with tf.variable_scope("total_loss"):
        total_loss = model.loss + hparams.h_reg * h_l2sqnorm \
                                    + hparams.r_reg * r_l2sqnorm

    with tf.variable_scope("summaries"):
        tf.summary.scalar("loss_function", tf.reshape(model.loss, []))
        tf.summary.scalar("h_l2norm", tf.reshape(h_l2sqnorm, []))
        tf.summary.scalar("r_l2norm", tf.reshape(r_l2sqnorm, []))
        tf.summary.scalar("total_loss", tf.reshape(total_loss, []))

        if FLAGS.visualize:
            # Doesn't work for Datasets where batch size can't be inferred
            waveform_op = tfplot.autowrap(waveform_plot, batch=True)(data)
            tf.summary.image("waveform", waveform_op)

        tf.summary.audio("samples", data, sample_rate=FLAGS.sample_rate, max_outputs=5)
        tf.summary.histogram("frequencies", model.H_diag / (2 * np.pi))

    step = tf.get_variable("global_step", [], tf.int64, tf.zeros_initializer(), trainable=False)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(total_loss, global_step=step)

    # TODO Unrolling in time?

    tf.contrib.training.train(train_op, save_checkpoint_secs=60,
                              logdir=f"{FLAGS.logdir}/{hparams.bond_dim}_{hparams.delta_t}_{hparams.minibatch_size}")

if __name__ == '__main__':
    tf.app.run(main)