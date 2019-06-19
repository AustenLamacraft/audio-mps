import tensorflow as tf
import numpy as np
from scipy.linalg import block_diag
import tensorflow_probability as tfp
tfd = tfp.distributions

FLAGS = tf.flags.FLAGS

def get_audio(datadir, dataset, hps):

    if dataset == 'damped_sine':

        input_length = FLAGS.sample_duration

        freq = 261.6 # Middle C
        decay_time = 0.1

        # freq = 600.
        # decay_time = 0.003

        delay_time = input_length / 100

        delays = tf.stack(input_length * [tf.random_gamma([hps.minibatch_size], alpha=2, beta=2/delay_time)], axis=-1)

        input_range = tf.expand_dims(tf.range(input_length, dtype=np.float32), axis=0)
        times = (input_range - delays) * hps.delta_t
        sine_wave_random_delay = 0.5 * (tf.sign(times) + 1) \
                                 * tf.sin(2 * np.pi * freq * times) * tf.exp(- times / decay_time)

        data = sine_wave_random_delay
        datalog = f"_freq{freq}_dect{decay_time}_delt{delay_time}"

    elif dataset == 'fixed_damped_sine':

        input_length = FLAGS.sample_duration
        freq = 600.
        decay_time = 0.003

        input_range = tf.expand_dims(tf.range(input_length, dtype=np.float32), axis=0)
        times = input_range * hps.delta_t
        sine_wave_fixed = tf.sin(2 * np.pi * freq * times) * tf.exp(- times / decay_time)

        data = sine_wave_fixed
        datalog = f"_freq{freq}_dect{decay_time}"

    elif dataset == 'fixed_damped_sine_2_freq':

        input_length = FLAGS.sample_duration
        freqa = 600.
        freqb = 800.
        decay_time = 0.003

        input_range = tf.expand_dims(tf.range(input_length, dtype=np.float32), axis=0)
        times = input_range * hps.delta_t
        sine_wave_fixed_a = tf.sin(2 * np.pi * freqa * times) * tf.exp(- times / decay_time)
        sine_wave_fixed_b = tf.sin(2 * np.pi * freqb * times) * tf.exp(- times / decay_time)

        data = tf.concat([sine_wave_fixed_a, sine_wave_fixed_b], 0)
        datalog = f"_freqa{freqa}_freqb{freqb}_dect{decay_time}_fix"

    elif dataset == 'damped_sine_2_freq':

        input_length = FLAGS.sample_duration

        freq1 = 261.6 # Middle C
        freq2 = 0.5 * freq1
        decay_time = 0.1

        # freq = 600.
        # decay_time = 0.003

        delay_time = input_length / 100
        # TODO probably a better idea to have all delays random (note hps.minibatch_size/2 below)
        delays = tf.stack(input_length * [tf.random_gamma([np.int(hps.minibatch_size/2)], alpha=2, beta=2/delay_time)], axis=-1)

        input_range = tf.expand_dims(tf.range(input_length, dtype=np.float32), axis=0)
        times = (input_range - delays) * hps.delta_t
        sine_wave_random_delay_1 = 0.5 * (tf.sign(times) + 1) \
                                 * tf.sin(2 * np.pi * freq1 * times) * tf.exp(- times / decay_time)
        sine_wave_random_delay_2 = 0.5 * (tf.sign(times) + 1) \
                                 * tf.sin(2 * np.pi * freq2 * times) * tf.exp(- times / decay_time)

        data = tf.concat([sine_wave_random_delay_1,sine_wave_random_delay_2],0)
        datalog = f"_freqa{freq1}_freqb{freq2}_dect{decay_time}_delt{delay_time}"

    elif dataset == 'gaussian_process':

        # D=1
        λ = [800]
        ω = [4800]
        σ = [2.]
        D_mix = len(λ)

        # D=2
        # λ = [50.*16,50.*16]
        # ω = [300.*16,500.*16]
        # σ = [1., 1.]
        # D_mix = len(λ)

        # D=3
        # λ = [50. * 16, 50. * 16, 50. * 16]
        # ω = [300. * 16, 500. * 16, 700. * 16]
        # σ = [1., 1., 1.]
        # D_mix = len(λ)

        Δt = hps.delta_t

        A_list = []
        for i in range(D_mix):
            A_list.append(np.exp(-λ[i] * Δt) * np.array([[np.cos(ω[i] * Δt), -np.sin(ω[i] * Δt)],
                                                         [np.sin(ω[i] * Δt), np.cos(ω[i] * Δt)]]))
        A_tf = tf.cast(block_diag(*A_list), dtype=tf.float32)

        def spectral_mixture(T, BATCH_SIZE, D_mix):
            """T: length of the notes
               BATCH_SIZE: number of notes"""
            # Concatenate all the q_matrix matrices in the second index, i.e. axis = 1

            α_list = []
            for i in range(D_mix):
                α_list.append(
                    tf.constant(σ[i]) * tf.sqrt(1 - tf.exp(-2 * tf.constant(λ[i] * Δt))))

            q_matrix_list = []
            for i in range(D_mix):
                q_matrix_list.append(tfd.Normal(loc=tf.constant(0.), scale=α_list[i]).sample([T, 2, BATCH_SIZE]))

            q_matrix = tf.concat(q_matrix_list, 1)  # axis = 1 concatenates in the second index

            f_initial_list = []
            for i in range(D_mix):
                f_initial_list.append(tfd.Normal(loc=tf.constant(0.), scale=σ[i]).sample([2, BATCH_SIZE]))

            f_initial = tf.concat(f_initial_list, 0)  # axis = 0 concatenates in the first index

            # Pick the odd elements of f=(f_1, f_2, ...) and sum (measurement model) them, both in the second index
            f_1 = tf.reduce_sum(tf.scan(lambda f, q: tf.matmul(A_tf, f) + q, elems=q_matrix,
                                        initializer=f_initial)[:, ::2], 1)

            return tf.transpose(f_1)


        data = spectral_mixture(FLAGS.sample_duration, hps.minibatch_size, D_mix)
        datalog = f"_Dmix{D_mix}"

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

    return data, datalog

