import numpy as np
import tensorflow as tf


tf.set_random_seed(0)

FLAGS = tf.flags.FLAGS

# Sampling flags
tf.flags.DEFINE_integer("sample_duration", 2**16, "Duration of samples (as integer).")
tf.flags.DEFINE_integer("sample_rate", 16000, "Sampling rate.")
tf.flags.DEFINE_boolean('visualize', True, 'Produce visualization. Probably Slow!')

tf.flags.DEFINE_string("modeldir", "./data", "Model directory.")



def main(argv):
    pass

if __name__ == '__main__':
    tf.app.run(main)