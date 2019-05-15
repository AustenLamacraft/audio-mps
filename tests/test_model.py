import tensorflow as tf
import numpy as np
from model import CMPS, RhoCMPS, PsiCMPS

from tensorflow.contrib.training import HParams
hparams = HParams(minibatch_size=8, bond_dim=8, delta_t=0.001,
                  sigma=1, h_reg=0, r_reg=0, initial_rank=3)


class TestCMPS(tf.test.TestCase):

    def testHIsHermitian(self):

        model = CMPS(hparams)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(model.H, tf.linalg.adjoint(model.H))


class TestRhoCMPS(tf.test.TestCase):

    def testRhoIsADensityMatrix(self):

        model = RhoCMPS(hparams)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(model.rho_0, model.rho_0 / tf.trace(model.rho_0))
            self.assertAllClose(model.rho_0, tf.transpose(model.rho_0, conjugate=True))


if __name__ == '__main__':
      tf.test.main()