import tensorflow as tf
import numpy as np

from model import AudioMPS

# PARAMETERS	

BOND_D = 20
dt  = 0.001
BATCH_SIZE = 20

# CHOOSE DATA

#path = '_linear'
#path = '_linear_and_minuslinear'
#path = '_linear_and_alphalinear'
#path = '_quadratic'
#path = '_gaussian'
#path = '_sine'
#path = '_damped_sine_1note'
#path = '_damped_sine_2note'
path = '_damped_sine_multirandomphase'
#path = '_two_quadratics'


# CHOOSE INITIAL STATE OF THE ANCILLA

# path_is = '_pure'
path_is = '_maximally_mixed'


# CREATE DATA

if path == '_quadratic':

	INPUT_LENGTH = 10
	with tf.variable_scope("model_data", reuse=tf.AUTO_REUSE):
		range_stack = tf.stack(BATCH_SIZE * [tf.range(INPUT_LENGTH,dtype=np.float32)])
		data = tf.square(range_stack)

elif path == '_linear':

        INPUT_LENGTH = 10
        with tf.variable_scope("model_data", reuse=tf.AUTO_REUSE):
                range_stack = tf.stack(BATCH_SIZE * [tf.range(INPUT_LENGTH,dtype=np.float32)])
                data = range_stack

elif path == '_linear_and_minuslinear':

        INPUT_LENGTH = 10
        with tf.variable_scope("model_data", reuse=tf.AUTO_REUSE):
                range_stack = tf.stack([tf.range(1,INPUT_LENGTH,dtype=np.float32),-tf.range(1,INPUT_LENGTH,dtype=np.float32)])
                data = range_stack

elif path == '_linear_and_alphalinear':

        INPUT_LENGTH = 10
        with tf.variable_scope("model_data", reuse=tf.AUTO_REUSE):
                range_stack = tf.stack([tf.range(1,INPUT_LENGTH,dtype=np.float32),-3.*tf.range(1,INPUT_LENGTH,dtype=np.float32)])
                data = range_stack


elif path == '_gaussian':

        INPUT_LENGTH = 20
        dist = tf.distributions.Normal(loc=10.,scale=3.)
        with tf.variable_scope("model_data", reuse=tf.AUTO_REUSE):
                range_stack = tf.stack(BATCH_SIZE * [tf.range(INPUT_LENGTH,dtype=np.float32)])
                data = dist.prob(range_stack)

elif path == '_sine':

        INPUT_LENGTH = 40
        with tf.variable_scope("model_data", reuse=tf.AUTO_REUSE):
                range_stack = tf.stack(BATCH_SIZE * [tf.range(INPUT_LENGTH,dtype=np.float32)])
                data = tf.sin(range_stack/2)

elif path == '_damped_sine_1note':

        INPUT_LENGTH = 200
        with tf.variable_scope("model_data", reuse=tf.AUTO_REUSE):
                range_stack = tf.stack(BATCH_SIZE * [tf.range(INPUT_LENGTH,dtype=np.float32)])
                data = tf.sin(range_stack / 4)* tf.exp(-0.03*tf.range(INPUT_LENGTH,dtype=np.float32))

elif path == '_damped_sine_2note':

        INPUT_LENGTH = 10
        with tf.variable_scope("model_data", reuse=tf.AUTO_REUSE):
                range_stack = tf.stack(BATCH_SIZE * [tf.range(1,INPUT_LENGTH,dtype=np.float32)])
                #data = tf.sin((range_stack / 2)+[[1.88],[np.pi+2.12]])* tf.exp(-0.1*tf.range(1,INPUT_LENGTH,dtype=np.float32))
                data = tf.sin((range_stack / 2)+[[],[6.63],[],[]])* tf.exp(-0.1*tf.range(1,INPUT_LENGTH,dtype=np.float32))

elif path == '_damped_sine_multirandomphase':

        qq = np.reshape(np.random.uniform(np.zeros(BATCH_SIZE),2*np.pi*np.ones(BATCH_SIZE)),(BATCH_SIZE,1)) 
        INPUT_LENGTH = 50
        with tf.variable_scope("model_data", reuse=tf.AUTO_REUSE):
                range_stack = tf.stack(BATCH_SIZE * [tf.range(1,INPUT_LENGTH,dtype=np.float32)])
                #data = tf.sin(range_stack / 2 + tf.random_uniform([BATCH_SIZE,1],minval=0,maxval=2*np.pi))* tf.exp(-0.1*tf.range(1,INPUT_LENGTH,dtype=np.float32))
                data = tf.sin((range_stack / 2)+qq)* tf.exp(-0.1*tf.range(1,INPUT_LENGTH,dtype=np.float32))

elif path == '_two_quadratics':

        INPUT_LENGTH = 10
        with tf.variable_scope("model_data", reuse=tf.AUTO_REUSE):
                data = 0.005*tf.stack([-100+3*tf.square(tf.range(1,INPUT_LENGTH,dtype=np.float32)),100-3*tf.square(tf.range(1,INPUT_LENGTH,dtype=np.float32))])


#INITIAL ANCILLA STATE

if path_is == '_maximally_mixed':

	rho_0 = (1. / BOND_D) * tf.eye(BOND_D, dtype=tf.complex64)

elif path_is == '_pure':

	rho_0 = tf.constant([[1,0],[0,0]],dtype=tf.complex64)	 

# CREATE THE OBJECT sine_model

with tf.variable_scope("our_model", reuse=tf.AUTO_REUSE):
    our_model = AudioMPS(BOND_D, dt, BATCH_SIZE, data_iterator=data, rho_0_in=rho_0)

# CREATE SUMMARIES OF THE STUFF WE WANT TO KEEP TRACK OF

tf.summary.scalar("loss_function", tf.reshape(our_model.loss, []))
tf.summary.scalar("H_00", tf.reshape(our_model.H[0][0], []))
tf.summary.scalar("R_00", tf.reshape(our_model.R[0][0], []))

# global_step: Optional Variable to increment by one after the variables have been updated. 

step = tf.get_variable("global_step", [], tf.int64, tf.zeros_initializer(), trainable=False)
train_op = tf.train.AdamOptimizer(1e-3).minimize(our_model.loss, global_step=step)

# RUN THE TRAINING LOOP

tf.contrib.training.train(train_op, logdir="../logging/logging_D"+str(BOND_D)+"_dt"+str(dt)+"_batchsize"+str(BATCH_SIZE)+path+path_is+"_neglect_first_term", save_checkpoint_secs=60)
