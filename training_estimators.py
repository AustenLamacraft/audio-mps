from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

# Dependency imports
from absl import flags
import numpy as np
from six.moves import urllib
import tensorflow as tf

from model import AudioMPS

flags.DEFINE_integer(
    "viz_steps", default=1, help="Frequency at which to save visualizations.")
flags.DEFINE_integer(
    "max_steps", default=5001, help="Number of training steps to run.")
flags.DEFINE_integer(
    "bond_d", default=10, help="Bond dimension.")
flags.DEFINE_float(
    "dt", default=0.001, help="Time discretization.")
flags.DEFINE_bool(
    "discr",
    default=False,
    help="If false, we are using a pure state.")
flags.DEFINE_integer(
    "batch_size",
    default=32,
    help="Batch size.")
flags.DEFINE_string(
    "model_dir",
    default="../logging/loggingrrrrrrrrtt",
    help="Directory to put the model's fit.")
flags.DEFINE_string(
    "data_dir",
    default="/rscratch/bm485/Code/audio-mps/audio-mps-github/data/pitch_30.tfrecords",
    help="Directory where data is stored (if using real data).")

FLAGS = flags.FLAGS

def audiomps(bond_d, dt, batch_size, data, discr):
    our_model = AudioMPS(bond_d, dt, batch_size, data_iterator=data, mixed=discr)
    return our_model


def model_fn(features, labels, mode, params, config):
  """Builds the model function for use in an estimator.

  Arguments:
    features: The input features for the estimator.
    labels: The labels, unused here.
    mode: Signifies whether it is train or test or predict.
    params: Some hyperparameters as a dictionary.
    config: The RunConfig, unused here.

  Returns:
    EstimatorSpec: A tf.estimator.EstimatorSpec instance.
  """
  del labels, config

  data = features
  loss = audiomps(params["bond_d"], params["dt"], params["batch_size"], data, params["discr"]).loss
  tf.summary.scalar("loss_function", tf.reshape(loss, []))

  global_step = tf.train.get_or_create_global_step()
  train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops={"loss": tf.metrics.mean(loss)})

def static_nsynth_dataset(directory):
  """Returns binary static NSynth tf.data.Dataset."""
  dataset = tf.data.TFRecordDataset(directory)

  def _parser(example_proto):
      features = {"audio": tf.FixedLenFeature([2 ** 16], dtype=tf.float32)}
      parsed_features = tf.parse_single_example(example_proto, features)
      return parsed_features["audio"]

  return dataset.map(_parser)

def build_input_fns(data_dir, batch_size):
  """Builds an Iterator switching between train and heldout data."""

  # Build an iterator over training batches.
  training_dataset = static_nsynth_dataset(data_dir)
  training_dataset = training_dataset.shuffle(buffer_size=24).repeat().batch(batch_size)
  train_input_fn = lambda: training_dataset.make_one_shot_iterator().get_next()

  return train_input_fn

def main(argv):
  del argv  # unused

  params = FLAGS.flag_values_dict()
  tf.gfile.MakeDirs(FLAGS.model_dir)

  train_input_fn = build_input_fns(FLAGS.data_dir, FLAGS.batch_size)

  estimator = tf.estimator.Estimator(
      model_fn,
      params=params,
      config=tf.estimator.RunConfig(
          model_dir=FLAGS.model_dir,
          save_checkpoints_steps=FLAGS.viz_steps,
      ),
  )

  for _ in range(FLAGS.max_steps // FLAGS.viz_steps):
    estimator.train(train_input_fn, steps=FLAGS.viz_steps)

tf.app.run()