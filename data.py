import tensorflow as tf

FLAGS = tf.flags.FLAGS

def get_audio(datadir, dataset, hps):

    if dataset == 'damped_sine':
        # TODO
        pass
    else:

        # LOAD DATA
        audio_dataset = tf.data.TFRecordDataset(f'{datadir}/{dataset}.tfrecords')

        # PARSE THE RECORD INTO TENSORS
        parse_function = lambda example_proto: tf.parse_single_example(example_proto,
                                                                       {"audio": tf.FixedLenFeature([2**16], dtype=tf.float32)})
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

