import tensorflow as tf
import numpy as np

from model import AudioMPS

# PARAMETERS
BOND_D = 1005
dt  = 0.001
BATCH_SIZE = 32

# CHOOSE BETWEEN PURE AND MIXED MODEL. True = mixed & False = pure
discr = False

# CHOOSE INITIAL STATE OF THE ANCILLA
path_is = '_pure'
#path_is = '_maximally_mixed'

# # LOAD DATA
# dataset = tf.data.TFRecordDataset('/rscratch/bm485/Code/audio-mps/audio-mps-github/data/pitch_30.tfrecords')
#
# # PARSE THE RECORD INTO TENSORS
# parse_function = lambda example_proto: tf.parse_single_example(example_proto,
#                                                                {"audio": tf.FixedLenFeature([2**16], dtype=tf.float32)})
# #TODO change to 64000 when I drop the padding in future datasets
# dataset = dataset.map(parse_function)
#
# # CONSUMING TFRecord DATA
# dataset = dataset.batch(batch_size=BATCH_SIZE)
# dataset = dataset.shuffle(buffer_size=100)
# dataset = dataset.repeat()
# iterator = dataset.make_one_shot_iterator()
# batch = iterator.get_next()
# data = batch['audio']

INPUT_LENGTH = 200
with tf.variable_scope("model_data", reuse=tf.AUTO_REUSE):
	range_stack = tf.stack(BATCH_SIZE * [tf.range(INPUT_LENGTH,dtype=np.float32)])
	data = tf.sin(range_stack/4 + tf.random_uniform([BATCH_SIZE,1],minval=0,maxval=2*np.pi))* tf.exp(-0.03*tf.range(INPUT_LENGTH,dtype=np.float32))

# INITIAL STATE OF ANCILLA IF MIXED
if discr:
        if path_is == '_maximally_mixed':

                rho_0 = (1. / BOND_D) * tf.eye(BOND_D, dtype=tf.complex64)

        elif path_is == '_pure':

                pure = np.zeros((BOND_D, BOND_D))
                pure[0][0] = 1.
                rho_0 = tf.constant(pure, dtype=tf.complex64)


# CREATE THE OBJECT our_model
with tf.variable_scope("our_model", reuse=tf.AUTO_REUSE):
    our_model = AudioMPS(BOND_D, dt, BATCH_SIZE, data_iterator=data, mixed=discr)

# CREATE SUMMARIES OF THE STUFF WE WANT TO KEEP TRACK OF
tf.summary.scalar("loss_function", tf.reshape(our_model.loss, []))
tf.summary.scalar("H_00", tf.reshape(our_model.H[0][0], []))
tf.summary.scalar("R_00", tf.reshape(our_model.R[0][0], []))

# global_step: Optional Variable to increment by one after the variables have been updated.
step = tf.get_variable("global_step", [], tf.int64, tf.zeros_initializer(), trainable=False)
train_op = tf.train.AdamOptimizer(1e-3).minimize(our_model.loss, global_step=step)

# RUN THE TRAINING LOOP
tf.contrib.training.train(train_op, logdir="../logging/logging_D"+str(BOND_D)+"_dt"+str(dt)+"_batchsize"+
                                           str(BATCH_SIZE)+"_discr"+str(discr)+"_sinthetic",save_checkpoint_secs=60)