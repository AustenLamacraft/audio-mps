import sys
import tensorflow as tf
sys.path.append("../")
sess = tf.InteractiveSession()
from reader import NSynthDataset


# MAKE A PITCH FIXED DATASET 

dataset = NSynthDataset('../data/nsynth-test.tfrecord', instrument='all', pitch=50, count=-1)
iterator = dataset.iterator(batch_size=32, length=2**16, repeat=False)


train_filename = 'pitch_50.tfrecords'  # address to save the TFRecords file

count = 0

with tf.python_io.TFRecordWriter(train_filename) as writer:
    while count < 100:
        try:
            batch = sess.run(iterator.get_next())
        except tf.errors.OutOfRangeError:
            break
        for datum in batch['audio']:
            # Create a feature
            audio_feature = tf.train.Feature(float_list=tf.train.FloatList(value=datum))
            feature = {'audio': audio_feature}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        
        count += len(batch['audio'])


    sys.stdout.flush()

