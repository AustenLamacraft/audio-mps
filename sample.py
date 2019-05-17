import numpy as np
import tensorflow as tf


tf.set_random_seed(0)

FLAGS = tf.flags.FLAGS

# Sampling flags

tf.flags.DEFINE_enum('dataset', 'damped_sine',
                     ['damped_sine', 'guitar', 'organ', 'nsynth'],
                     'Dataset. Must be one of "damped_sine", "guitar", "organ", or "nsynth".')

tf.flags.DEFINE_integer("sample_duration", 2**16, "Duration of samples (as integer).")
tf.flags.DEFINE_boolean('visualize', True, 'Produce visualization. Probably Slow!')
tf.flags.DEFINE_string("modeldir", f"../logging/audio_mps/{FLAGS.dataset}", "Model directory.")


def main(argv):
    pass

if __name__ == '__main__':
    tf.app.run(main)