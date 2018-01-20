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
        self.x = tf.placeholder(tf.int32, shape=[batch_size, self.rep_sequence_length])

        # self.start_tokens = tf.zeros([batch_size],tf.int32)
        self.dec_inp = tf.concat([tf.expand_dims(self.start_tokens, 1), self.labels], 1)

        self.input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(self.enc_inp, self.hist_end_token)), 1)
        self.output_lengths_full = tf.constant([self.rep_sequence_length]*self.batch_size, dtype=tf.int32)
        self.output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(self.dec_inp, self.end_token)), 1)

        self.g_embeddings = tf.Variable(tf.constant(0.0, shape=[self.num_emb, self.emb_dim]),
                                   trainable=False, name="W")

        self.embedding_placeholder = tf.placeholder(tf.float32, [self.num_emb, self.emb_dim])
        self.embedding_init = self.g_embeddings.assign(self.embedding_placeholder)
        
        # word_proba[t] = p(y_t | X, y_{0:t-1}
        self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.rep_sequence_length])
        self.baseline = tf.placeholder(tf.float32, shape=[self.batch_size, self.rep_sequence_length])
        self.word_probas = tf.placeholder(tf.float32, shape=[self.batch_size, self.rep_sequence_length])

        with tf.device("/cpu:0"):
            # processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.enc_inp), perm=[1, 0, 2])
            # processed_y = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.dec_inp), perm=[1, 0, 2])
            processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.enc_inp), perm=[0, 1, 2])
            processed_y = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.dec_inp), perm=[0, 1, 2])
        self.enc_cell = tf.contrib.rnn.GRUCell(self.hidden_dim)
        self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(self.enc_cell, processed_x, self.input_lengths, self.prev_mem)

        train_helper = tf.contrib.seq2seq.TrainingHelper(processed_y,  self.output_lengths_full)
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
                print("outputs: ", outputs)
                return outputs[0]

        self.train_outputs = decode(train_helper, 'decode')
        print("\nself.train_outputs: ",self.train_outputs)
        print("\nself.train_outputs.rnn_output: ",self.train_outputs.rnn_output)
        print("\nself.train_outputs.sample_id: ",self.train_outputs.sample_id)
        #print("train_outputs.shape()", self.train_outputs.get_shape())
        self.gen_x = decode(pred_helper, 'decode', reuse=True)

        #tf.identity(self.train_outputs.sample_id[0], name='train_pred')
        #tf.identity(self.train_outputs.rnn_output[0], name='train_pred')

        weights = tf.to_float(tf.not_equal(self.dec_inp[:, :-1], self.end_token))
        print(self.labels.get_shape())
        print(weights.get_shape())
        print(self.train_outputs.rnn_output.get_shape())
        logits = self.train_outputs.rnn_output
       
        # Pre training optimization
        self.pretrain_loss = tf.contrib.seq2seq.sequence_loss(
            logits = self.train_outputs.rnn_output, targets = self.labels, weights=weights)
        self.params = tf.trainable_variables()
        self.gradients = tf.gradients(self.pretrain_loss, self.params)
        self.clipped_gradients, _ = tf.clip_by_global_norm( self.gradients, self.grad_clip)

        optimizer = self.g_optimizer(self.learning_rate)
        self.pretrain_updates = optimizer.apply_gradients(
            zip(self.clipped_gradients, self.params))

        # Adversarial optimization
        print("\ntrain variables: \n",tf.trainable_variables())
        print("self.x.get_shpae(): ", self.x.get_shape())
        print("self.rewards.get_shape()): ", self.rewards.get_shape())
        print("self.baseline.get_shape()): ", self.baseline.get_shape())
        
        self.g_loss = -tf.reduce_sum(
                tf.reduce_sum(tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 0.0, 1.0) *
                    tf.log(tf.clip_by_value(tf.reshape(self.train_outputs.rnn_output, [-1, self.num_emb]),1e-20, 1.0)) )
                * tf.reshape(self.rewards, [-1]))

        self.params = tf.trainable_variables()
        g_opt = self.g_optimizer(self.learning_rate)
        self.g_grad, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, self.params), self.grad_clip)
        self.g_updates = g_opt.apply_gradients(zip(self.g_grad, self.params))


    def generate(self, sess, X,Y):
        """
        Generates a sentence and the proba of each of word of it
        Args: 
            sess: tensorflow session
            X: dialogue history (encoder input)
        """
        feed_dict = {self.enc_inp:X, self.labels:Y} 
        _, sentence = sess.run(self.gen_x, feed_dict=feed_dict)
        return sentence

    def assign_emb(self, sess,x):
        sess.run(self.embedding_init,feed_dict={self.embedding_placeholder: x})
        return

    def pretrain_step(self, sess, x,y):
        outputs = sess.run([self.pretrain_updates, self.pretrain_loss], feed_dict={self.enc_inp: x,self.labels:y})
        return outputs

    #def advtrain_step(self, sess, word_probas, rewards, baseline):
    def advtrain_step(self, sess, history, labels, sentence, rewards, baseline):
        """

        Args:
            sess: tf session
            sentence: sentence output by generator
            rewards: 
            baseline
        """
        #feed_dict = {self.word_probas: word_probas, self.rewards:rewards, self.baseline:baseline}
        feed_dict = {self.enc_inp:history, self.labels: labels, self.x: sentence, self.rewards:rewards, self.baseline:baseline}
        sess.run([self.g_updates], feed_dict)

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        return tf.zeros(shape)
    
    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer(*args, **kwargs)


