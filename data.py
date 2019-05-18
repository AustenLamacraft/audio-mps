import tensorflow as tf
import numpy as np

FLAGS = tf.flags.FLAGS

def get_audio(datadir, dataset, hps):

    if dataset == 'damped_sine':

        input_length = FLAGS.sample_duration
        freq = 261.6 # Middle C
        decay_time = 0.1
        delay_time = input_length / 100

        delays = tf.stack(input_length * [tf.random_gamma([hps.minibatch_size], alpha=2, beta=2/delay_time)], axis=-1)

        input_range = tf.expand_dims(tf.range(input_length, dtype=np.float32), axis=0)
        times = (input_range - delays) * hps.delta_t
        sine_wave_random_delay = 0.5 * (tf.sign(times) + 1) \
                                 * tf.sin(2 * np.pi * freq * times) * tf.exp(- times / decay_time)

        data = sine_wave_random_delay


    else:

        # LOAD DATA
        audio_dataset = tf.data.TFRecordDataset(f'{datadir}/{dataset}.tfrecords')

        # PARSE THE RECORD INTO TENSORS
        parse_function = lambda example_proto: \
            tf.parse_single_example(example_proto, {"audio": tf.FixedLenFeature([FLAGS.sample_duration], dtype=tf.float32)})
        #TODO change to 64000 when I drop the padding in future datasets
        audio_dataset = audio_dataset.map(parse_function)

        # CONSUMING TFRecord DATA
        audio_dataset = audio_dataset.batch(batch_size=hps.minibatch_size)
        audio_dataset = audio_dataset.shuffle(buffer_size=24)
        audio_dataset = audio_dataset.repeat()
        iterator = audio_dataset.make_one_shot_iterator()
        batch = iterator.get_next()

        data = batch['audio']

    return data

