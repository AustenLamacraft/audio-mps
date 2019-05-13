import tensorflow as tf
import numpy as np
from model import AudioMPS


class TestAudioMPS(tf.test.TestCase):

    def testHIsSymmetric(self):

        model = AudioMPS(bond_d=8, delta_t=0.01, batch_size=8)

        with self.session() as sess:
            sess.run(tf.global_variables_initializer())
            H = model.H.eval()
            self.assertAllClose(H, tf.transpose(H))


if __name__ == '__main__':
      tf.test.main()