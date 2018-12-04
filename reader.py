import tensorflow as tf

class NSynthDataset(object):
    """Dataset object to help manage the TFRecord loading."""

    instruments = ["all", "bass", "brass", "flute", "guitar", "keyboard", "mallet", "organ", "reed", "string",
                   "synth_lead", "vocal"]

    def __init__(self, tfrecord_path, instrument="all", pitch=-1, count=-1):

        self.record_path = tfrecord_path
        self.count = count
        if instrument.lower() not in self.instruments:
            raise ValueError("instrument must be one of: "+ ", ".join(instruments))
        else:
            self.instrument = instrument.lower()
        if pitch not in range(21,109) and pitch != -1:
            raise ValueError("pitch must be in range 21-108, or -1 for all pitches")
        else:
            self.pitch = pitch

    def iterator(self, batch_size=32, length=2**16, repeat=False):
        """Get an initializable iterator over the data."""

        dataset = tf.data.TFRecordDataset(self.record_path)
        dataset = dataset.take(self.count)  # This serves only for testing, cutting the dataset down to size...
        dataset = dataset.map(self._parse_function)  # Parse the record into tensors.
        dataset = dataset.filter(self._filter_function)  #  Apply filter
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000))
        dataset = dataset.padded_batch(batch_size, padded_shapes={"note": [None],
                                                                  "pitch": [None],
                                                                  "sample_rate": [None],
                                                                  "instrument_source_str": [None],
                                                                  "instrument_family_str": [None],
                                                                  'audio': [length]})  # Pad audio data out to `length`

        if repeat:
            dataset = dataset.repeat()  # Repeat the input indefinitely

        return dataset.make_one_shot_iterator()
        #return dataset.make_initializable_iterator()
    
    def _parse_function(self, example_proto):
      features = {
            "note": tf.FixedLenFeature([1], dtype=tf.int64),
            "pitch": tf.FixedLenFeature([1], dtype=tf.int64),
            "sample_rate": tf.FixedLenFeature([1], dtype=tf.int64),
            "instrument_source_str": tf.FixedLenFeature([1], dtype=tf.string),
            "instrument_family_str": tf.FixedLenFeature([1], dtype=tf.string),
            "audio": tf.FixedLenFeature([64000], dtype=tf.float32),
        }
      return tf.parse_single_example(example_proto, features)

    def _filter_function(self, features):
        record_instrument = features['instrument_family_str'][0]
        record_pitch = features['pitch'][0]
        if self.instrument == 'all':
            instrument_check = True
        else:
            instrument_check = tf.equal(record_instrument, self.instrument)
        if self.pitch == -1:
            pitch_check = True
        else:
            pitch_check = tf.equal(record_pitch, self.pitch)

        return tf.logical_and(pitch_check, instrument_check)
