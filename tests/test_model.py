import tensorflow as tf
import numpy as np
from model import CMPS


class TestAudioMPS(tf.test.TestCase):

    def testHIsHermitian(self):

        model = CMPS(bond_d=8, delta_t=0.01, batch_size=8)

        with self.session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(model.H, tf.transpose(model.H, conjugate=True))


if __name__ == '__main__':
      tf.test.main()