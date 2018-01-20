import numpy
import tensorflow as tf

from tensorflow.contrib import layers
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops

from tensorflow.contrib.seq2seq.python.ops.helper import MonteCarloEmbeddingHelper as MonteCarloHelper

class Generator(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
                 sequence_length,rep_sequence_length,start_token ,end_token,hist_end_token,
                 learning_rate=0.0004, reward_gamma=1.00):


        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.rep_sequence_length = rep_sequence_length
        self.end_token = end_token
        self.hist_end_token = end_token
        self.start_tokens = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        # self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.reward_gamma = reward_gamma
        self.g_params = []
        self.temperature = 1.0
        self.grad_clip = 5.0
        self.expected_reward = tf.Variable(tf.zeros([self.sequence_length]))

        self.prev_mem = tf.zeros((batch_size, self.hidden_dim))
        self.enc_inp = tf.placeholder(tf.int32, shape=[batch_size, self.sequence_length])
        self.labels = tf.placeholder(tf.int32, shape=[batch_size, self.rep_sequence_length])

        # self.start_tokens = tf.zeros([batch_size],tf.int32)
        self.dec_inp = tf.concat([tf.expand_dims(self.start_tokens, 1), self.labels], 1)

        self.input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(self.enc_inp, self.hist_end_token)), 1)
        self.output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(self.dec_inp, self.end_token)), 1)
        # self.dec_inp = [tf.placeholder(tf.int32, shape=(None,),
        #                           name="labels%i" % t)
        #            for t in range(self.rep_seq_length)]
        # self.labels = [tf.placeholder(tf.int32, shape=(None,),
        #                          name="labels%i" % t)
        #           for t in range(self.rep_seq_length)]
        #
        # self.weights = [tf.ones_like(labels_t, dtype=tf.float32)
        #            for labels_t in self.labels]

        self.g_embeddings = tf.Variable(tf.constant(0.0, shape=[self.num_emb, self.emb_dim]),
                                   trainable=False, name="W")

        self.embedding_placeholder = tf.placeholder(tf.float32, [self.num_emb, self.emb_dim])
        self.embedding_init = self.g_embeddings.assign(self.embedding_placeholder)

        self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.rep_sequence_length]) # get from rollout policy and discriminator
        with tf.device("/cpu:0"):
            # processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.enc_inp), perm=[1, 0, 2])
            # processed_y = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.dec_inp), perm=[1, 0, 2])
            processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.enc_inp), perm=[0, 1, 2])
            processed_y = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.dec_inp), perm=[0, 1, 2])
        self.enc_cell = tf.contrib.rnn.GRUCell(self.hidden_dim)
        self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(self.enc_cell, processed_x, self.input_lengths, self.prev_mem)

        train_helper = tf.contrib.seq2seq.TrainingHelper(processed_y,  self.output_lengths)
        sampling_prob = tf.Variable(0.0, dtype=tf.float32)
        # pred_helper = tf.contrib.seq2seq.SampleEmbeddingHelper(sample,numpy.full((self.batch_size),self.rep_sequence_length),tf.int32, processed_y,end_fn,next_inputs)
        #
        # pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        #     self.g_embeddings, start_tokens=tf.to_int32(self.start_tokens), end_token=1)
        pred_helper = MonteCarloHelper(processed_y,self.output_lengths, self.g_embeddings, start_tokens=tf.to_int32(self.start_tokens), end_token=self.end_token,softmax_temperature=self.temperature,seed = 1881)

        # pred_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
        #     processed_y, self.output_lengths,
        #     self.g_embeddings,
        #     sampling_probability=sampling_prob)
        # pred_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
        #     self.g_embeddings, start_tokens=tf.to_int32(self.start_tokens), end_token=1,sampling_probability=sampling_prob)

        def decode(helper, scope, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=self.hidden_dim, memory=self.encoder_outputs,
                    memory_sequence_length=self.input_lengths)
                cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_dim)
                attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell, attention_mechanism, attention_layer_size=self.hidden_dim / 2)
                out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                    attn_cell, self.num_emb, reuse=reuse
                )

                projection_layer = layers_core.Dense(self.num_emb,use_bias = False)

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=out_cell, helper=helper,
                    initial_state=out_cell.zero_state(
                        dtype=tf.float32, batch_size=batch_size),
                    # initial_state=self.encoder_state,
                    output_layer = projection_layer
                )

                outputs = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder, output_time_major=False,
                    impute_finished=True, maximum_iterations=self.rep_sequence_length
                )
                return outputs[0]

        self.train_outputs = decode(train_helper, 'decode')
        self.gen_x = decode(pred_helper, 'decode', reuse=True)

        tf.identity(self.train_outputs.sample_id[0], name='train_pred')

        weights = tf.to_float(tf.not_equal(self.dec_inp[:, :-1], self.end_token))
        print(self.labels.get_shape())
        print(weights.get_shape())
        print(self.train_outputs.rnn_output.get_shape())

        self.pretrain_loss = tf.contrib.seq2seq.sequence_loss(
            logits = self.train_outputs.rnn_output, targets = self.labels, weights=weights)

        self.params = tf.trainable_variables()
        self.gradients = tf.gradients(self.pretrain_loss, self.params)
        self.clipped_gradients, _ = tf.clip_by_global_norm( self.gradients, self.grad_clip)

        optimizer = self.g_optimizer(self.learning_rate)
        self.pretrain_updates = optimizer.apply_gradients(
            zip(self.clipped_gradients, self.params))



        # self.g_loss = -tf.reduce_sum(
        #     tf.reduce_sum(
        #         tf.one_hot(tf.to_int32(tf.reshape(self.labels, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
        #             tf.clip_by_value(tf.reshape(self.train_outputs.rnn_output, [-1, self.num_emb]), 1e-20, 1.0)
        #         ), 1) * tf.reshape(self.rewards, [-1])
        # )
        #
        # g_opt = self.g_optimizer(self.learning_rate)
        #
        # self.g_grad, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, self.params), self.grad_clip)
        # self.g_updates = g_opt.apply_gradients(zip(self.g_grad, self.params))




    def generate(self, sess):
        outputs = sess.run(self.gen_x)
        return outputs

    def assign_emb(self, sess,x):
        sess.run(self.embedding_init,feed_dict={self.embedding_placeholder: x})
        return

    def pretrain_step(self, sess, x,y):
        outputs = sess.run([self.pretrain_updates, self.pretrain_loss], feed_dict={self.enc_inp: x,self.labels:y})
        return outputs

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        return tf.zeros(shape)

    # def create_recurrent_unit(self, params):
    #     # Weights and Bias for input and hidden tensor
    #     self.Wi = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
    #     self.Ui = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
    #     self.bi = tf.Variable(self.init_matrix([self.hidden_dim]))
    #
    #     self.Wf = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
    #     self.Uf = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
    #     self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))
    #
    #     self.Wog = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
    #     self.Uog = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
    #     self.bog = tf.Variable(self.init_matrix([self.hidden_dim]))
    #
    #     self.Wc = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
    #     self.Uc = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
    #     self.bc = tf.Variable(self.init_matrix([self.hidden_dim]))
    #     params.extend([
    #         self.Wi, self.Ui, self.bi,
    #         self.Wf, self.Uf, self.bf,
    #         self.Wog, self.Uog, self.bog,
    #         self.Wc, self.Uc, self.bc])
    #
    #     def unit(x, hidden_memory_tm1):
    #         previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
    #
    #         # Input Gate
    #         i = tf.sigmoid(
    #             tf.matmul(x, self.Wi) +
    #             tf.matmul(previous_hidden_state, self.Ui) + self.bi
    #         )
    #
    #         # Forget Gate
    #         f = tf.sigmoid(
    #             tf.matmul(x, self.Wf) +
    #             tf.matmul(previous_hidden_state, self.Uf) + self.bf
    #         )
    #
    #         # Output Gate
    #         o = tf.sigmoid(
    #             tf.matmul(x, self.Wog) +
    #             tf.matmul(previous_hidden_state, self.Uog) + self.bog
    #         )
    #
    #         # New Memory Cell
    #         c_ = tf.nn.tanh(
    #             tf.matmul(x, self.Wc) +
    #             tf.matmul(previous_hidden_state, self.Uc) + self.bc
    #         )
    #
    #         # Final Memory cell
    #         c = f * c_prev + i * c_
    #
    #         # Current Hidden state
    #         current_hidden_state = o * tf.nn.tanh(c)
    #
    #         return tf.stack([current_hidden_state, c])
    #
    #     return unit
    #
    # def create_output_unit(self, params):
    #     self.Wo = tf.Variable(self.init_matrix([self.hidden_dim, self.num_emb]))
    #     self.bo = tf.Variable(self.init_matrix([self.num_emb]))
    #     params.extend([self.Wo, self.bo])
    #
    #     def unit(hidden_memory_tuple):
    #         hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
    #         # hidden_state : batch x hidden_dim
    #         logits = tf.matmul(hidden_state, self.Wo) + self.bo
    #         # output = tf.nn.softmax(logits)
    #         return logits
    #
    #     return unit

    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer(*args, **kwargs)