import tensorflow as tf
import numpy as np
import tfplot

from tensorflow.contrib.training import HParams
from model import RhoCMPS
from data import get_audio

DTYPE=tf.float32
NP_DTYPE=np.float32

tf.set_random_seed(0)

FLAGS = tf.flags.FLAGS

# Training flags
tf.flags.DEFINE_enum('dataset', 'guitar',
                     ['damped_sine', 'guitar', 'organ', 'nsynth'],
                     'Dataset. Must be one of "damped_sine", "guitar", "organ", or "nsynth".')
tf.flags.DEFINE_boolean('visualize', True, 'Produce visualization.')
tf.flags.DEFINE_string("hparams", "", 'Comma separated list of "name=value" pairs e.g. "--hparams=learning_rate=0.3"')
tf.flags.DEFINE_string("datadir", "./data", "Data directory.")
tf.flags.DEFINE_string("logdir", f"../logging/audio_mps/{FLAGS.dataset}", "Directory to write logs.")


def main(argv):

    hparams = HParams(minibatch_size=8, bond_dim=8, delta_t=0.001, h_reg=0, r_reg=0)
    hparams.parse(FLAGS.hparams)

    bond_dim = hparams.bond_dim
    dt = hparams.delta_t
    minibatch_size = hparams.minibatch_size

    # CHOOSE BETWEEN PURE AND MIXED MODEL. True = mixed & False = pure
    discr = False

    # CHOOSE INITIAL STATE OF THE ANCILLA
    path_is = '_pure'
    #path_is = '_maximally_mixed'

    data = get_audio(datadir=FLAGS.datadir, dataset=FLAGS.dataset, hps=hparams)

    # INITIAL STATE OF ANCILLA IF MIXED
    if discr:
            if path_is == '_maximally_mixed':

                    rho_0 = (1. / bond_dim) * tf.eye(bond_dim, dtype=tf.complex64)

            elif path_is == '_pure':

                    pure = np.zeros((bond_dim, bond_dim))
                    pure[0][0] = 1.
                    rho_0 = tf.constant(pure, dtype=tf.complex64)



    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        model = RhoCMPS(bond_dim, dt, minibatch_size, data_iterator=data, rho_0=rho_0)


    tf.summary.scalar("loss_function", tf.reshape(model.loss, []))
    tf.summary.scalar("H_00", tf.reshape(model.H[0][0], []))
    tf.summary.scalar("R_00", tf.reshape(model.R[0][0], []))

    if FLAGS.visualize:
        # TODO Create waveform summaries using tfplot. Add purity
        pass

    step = tf.get_variable("global_step", [], tf.int64, tf.zeros_initializer(), trainable=False)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(model.loss, global_step=step)

    # TODO Unrolling in time?

    tf.contrib.training.train(train_op, save_checkpoint_secs=60,
                              logdir=f"{FLAGS.logdir}/{hparams.bond_dim}_{hparams.delta_t}_{hparams.minibatch_size}")

if __name__ == '__main__':
    tf.app.run(main)