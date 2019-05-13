import tensorflow as tf
import numpy as np
from data import get_audio
from tensorflow.contrib.training import HParams

FLAGS = tf.flags.FLAGS
hps = hparams = HParams(minibatch_size=8, bond_dim=8, delta_t=0.001)


class TestGetAudio(tf.test.TestCase):

    def testHIsHermitian(self):

        data = get_audio('./data', 'guitar', hps)
        print(data)
        with self.cached_session() as sess:
            self.assertEquals(data.eval().shape[0], hps.minibatch_size)


if __name__ == '__main__':
      tf.test.main()