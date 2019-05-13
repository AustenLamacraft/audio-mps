import tensorflow as tf

FLAGS = tf.flags.FLAGS

def get_audio(hparams):

    # LOAD DATA
    dataset = tf.data.TFRecordDataset('/Users/mencia/PhD_local/audioMPS/data/pitch_30.tfrecords')

    # PARSE THE RECORD INTO TENSORS
    parse_function = lambda example_proto: tf.parse_single_example(example_proto,
                                                                   {"audio": tf.FixedLenFeature([2**16], dtype=tf.float32)})
    #TODO change to 64000 when I drop the padding in future datasets
    dataset = dataset.map(parse_function)

    # CONSUMING TFRecord DATA
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    dataset = dataset.shuffle(buffer_size=24)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()

    return batch['audio']

