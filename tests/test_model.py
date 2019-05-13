import tensorflow as tf
import numpy as np
from model import CMPS, RhoCMPS, PsiCMPS


class TestCMPS(tf.test.TestCase):

    def testHIsHermitian(self):

        model = CMPS(bond_d=8, delta_t=0.01, batch_size=8)

        with self.session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(model.H, tf.linalg.adjoint(model.H))


class TestRhoCMPS(tf.test.TestCase):

    def testRhoIsNormalized(self):

        model = RhoCMPS(bond_d=8, delta_t=0.01, batch_size=8)

        with self.session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(model.rho_0, model.rho_0 / tf.trace(model.rho_0))


if __name__ == '__main__':
      tf.test.main()