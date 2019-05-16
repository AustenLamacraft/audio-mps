import tensorflow as tf
import numpy as np
import tfplot

from tensorflow.contrib.training import HParams
from model import RhoCMPS
from data import get_audio
from utils import waveform_plot

DTYPE=tf.float32
NP_DTYPE=np.float32

tf.set_random_seed(0)

FLAGS = tf.flags.FLAGS

# Training flags
tf.flags.DEFINE_enum('dataset', 'damped_sine',
                     ['damped_sine', 'guitar', 'organ', 'nsynth'],
                     'Dataset. Must be one of "damped_sine", "guitar", "organ", or "nsynth".')

tf.flags.DEFINE_integer("sample_duration", 2**16, "Duration of samples (as integer).")
tf.flags.DEFINE_boolean('visualize', True, 'Produce visualization. Probably Slow!')
tf.flags.DEFINE_string("hparams", "", 'Comma separated list of "name=value" pairs e.g. "--hparams=learning_rate=0.3"')
tf.flags.DEFINE_string("datadir", "./data", "Data directory.")
tf.flags.DEFINE_string("logdir", f"../logging/audio_mps/{FLAGS.dataset}", "Directory to write logs.")


def main(argv):
    # I introduced the hyperparameter A, which is set to 1. by default. The meaning of sigma is not Asgsq{dt} anymore.
    # sigma now is the strength of the noise.
    hparams = HParams(minibatch_size=8, bond_dim=8, delta_t=0.001,
                      sigma=1, h_reg=0, r_reg=0, initial_rank=None, A=1)
    hparams.parse(FLAGS.hparams)

    with tf.variable_scope("data"):
        data = get_audio(datadir=FLAGS.datadir, dataset=FLAGS.dataset, hps=hparams)
        #TODO put it back

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        model = RhoCMPS(hparams=hparams, data_iterator=data)

    with tf.variable_scope("summaries"):
        tf.summary.scalar("loss_function", tf.reshape(model.loss, []))
        # TODO Add other loss terms to summaries

        if FLAGS.visualize:
            # Doesn't work for Datasets where batch size can't be inferred
            waveform_op = tfplot.autowrap(waveform_plot, batch=True)(data)
            tf.summary.image("waveform", waveform_op)

        tf.summary.audio("samples", data, sample_rate=FLAGS.sample_rate, max_outputs=5)

    step = tf.get_variable("global_step", [], tf.int64, tf.zeros_initializer(), trainable=False)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(model.loss, global_step=step)

    # TODO Unrolling in time?

    tf.contrib.training.train(train_op, save_checkpoint_secs=60,
                              logdir=f"{FLAGS.logdir}/{hparams.bond_dim}_{hparams.delta_t}_{hparams.minibatch_size}")

if __name__ == '__main__':
    tf.app.run(main)