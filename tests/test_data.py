import tensorflow as tf
import numpy as np
from data import get_audio
from tensorflow.contrib.training import HParams


tf.flags.DEFINE_integer("sample_duration", 2**16, "Duration of samples (as integer).")
hps = hparams = HParams(minibatch_size=8, bond_dim=8, delta_t=0.001)
FLAGS = tf.flags.FLAGS

class TestGetAudio(tf.test.TestCase):

    def testCorrectShape(self):

        data = get_audio('../data', 'guitar', hps)
        with self.cached_session() as sess:
            self.assertEqual(data.eval().shape, (hps.minibatch_size, FLAGS.sample_duration))


if __name__ == '__main__':
      tf.test.main()