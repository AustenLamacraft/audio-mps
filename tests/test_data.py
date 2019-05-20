import tensorflow as tf
import numpy as np
from data import get_audio
from tensorflow.contrib.training import HParams


hparams = HParams(minibatch_size=8, bond_dim=8, delta_t=0.001)
FLAGS = tf.flags.FLAGS

class TestGetAudio(tf.test.TestCase):

    def testCorrectShape(self):

        data = get_audio(None, 'damped_sine', hparams)
        with self.cached_session() as sess:
            self.assertEqual(data.eval().shape, (hparams.minibatch_size, FLAGS.sample_duration))


if __name__ == '__main__':
      tf.test.main()