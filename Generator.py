import numpy
import tensorflow as tf

from tensorflow.contrib import layers
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops

from tensorflow.contrib.seq2seq.python.ops.helper import MonteCarloEmbeddingHelper as MonteCarloHelper

class Generator(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
                 sequence_length,rep_sequence_length,
                 learning_rate=0.0004, reward_gamma=1.00):
        """
        Args:
            num_emb: vocab size
            batch_size: batch size
            emb_dim: word ebedding dimension
            hidden_dim: hiddent state dimension
            sequence_length: history max length
            rep_sequence_length: sentence max length
        """

        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.rep_sequence_length = rep_sequence_length

        # self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.reward_gamma = reward_gamma
        self.g_params = []
        self.temperature = 1.0 # ?
        self.grad_clip = 5.0
        self.expected_reward = tf.Variable(tf.zeros([self.sequence_length]))

        self.prev_mem = tf.zeros((batch_size, self.hidden_dim)) #h_{t-1}
        self.enc_inp = tf.placeholder(tf.int32, shape=[batch_size, self.sequence_length])#history
        self.labels = tf.placeholder(tf.int32, shape=[batch_size, self.rep_sequence_length])#expected sentence
        self.start_tokens = tf.zeros([batch_size],tf.int32)# start token decoder input
        self.dec_inp = tf.concat([tf.expand_dims(self.start_tokens, 1), self.labels], 1) #decoder input 

        self.input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(self.enc_inp, 1)), 1)
        self.output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(self.dec_inp, 1)), 1)

        self.g_embeddings = tf.Variable(tf.constant(0.0, shape=[self.num_emb, self.emb_dim]),
                                   trainable=False, name="W")
        self.embedding_placeholder = tf.placeholder(tf.float32, [self.num_emb, self.emb_dim])
        self.embedding_init = self.g_embeddings.assign(self.embedding_placeholder)

        self.seq_len = tf.placeholder(tf.int32, shape=(None,))
            
        # rewards[t] = rewards for the first t tokens of the sentence
        # baseline[t] = baseline for the first t tokens of the sentence
        #word_probas[t] = p(y_t | X, y_{0:t-1})
        self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.rep_sequence_length])
        self.baseline = tf.placeholder(tf.float32, shape=[self.batch_size, self.rep_sequence_length])
        self.word_probas = tf.placeholder(tf.float32, shape=[self.batch_size, self.rep_sequence_length])

        with tf.device("/cpu:0"):
            # processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.enc_inp), perm=[1, 0, 2])
            # processed_y = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.dec_inp), perm=[1, 0, 2])
            processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.enc_inp), perm=[0, 1, 2])
            processed_y = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.dec_inp), perm=[0, 1, 2])
        
        # Build encoder
        self.enc_cell = tf.contrib.rnn.GRUCell(self.hidden_dim)
        self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(self.enc_cell, processed_x, self.seq_len, self.prev_mem)

        train_helper = tf.contrib.seq2seq.TrainingHelper(processed_y,  self.output_lengths)
        sampling_prob = tf.Variable(0.0, dtype=tf.float32)
        # sampling_prob =tf.zeros([rep_sequence_length], tf.float32)

        def end_fn(sample_ids):

            return tf.equal(sample_ids, self._end_token)

        def sample( outputs):
            """sample for GreedyEmbeddingHelper."""

            # Outputs are logits, use argmax to get the most probable id
            if not isinstance(outputs, tf.Tensor):
                raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                                type(outputs))
            sample_ids = tf.argmax(outputs, axis=-1, output_type=tf.int32)
            return sample_ids

        def next_inputs( sample_ids):
            """next_inputs_fn for GreedyEmbeddingHelper."""

            finished = tf.equal(sample_ids, self._end_token)
            all_finished = tf.reduce_all(finished)
            next_inputs = tf.cond(
                all_finished,
                # If we're finished, the next_inputs value doesn't matter
                lambda: self._start_inputs,
                lambda: self._embedding_fn(sample_ids))
            return ( next_inputs)

        # pred_helper = tf.contrib.seq2seq.SampleEmbeddingHelper(sample,numpy.full((self.batch_size),self.rep_sequence_length),tf.int32, processed_y,end_fn,next_inputs)
        #
        # pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        #     self.g_embeddings, start_tokens=tf.to_int32(self.start_tokens), end_token=1)
        pred_helper = MonteCarloHelper(processed_y,self.output_lengths, self.g_embeddings, start_tokens=tf.to_int32(self.start_tokens), end_token=1,softmax_temperature=self.temperature,seed = 1881)

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
        
        # Can you specify which one to use when ?
        self.train_outputs = decode(train_helper, 'decode')
        self.gen_x = decode(pred_helper, 'decode', reuse=True)

        tf.identity(self.train_outputs.sample_id[0], name='train_pred')
        
        # Pre training optimization
        weights = tf.to_float(tf.not_equal(self.dec_inp[:, :-1], 1))
        self.pretrain_loss = tf.contrib.seq2seq.sequence_loss(
            self.train_outputs.rnn_output, self.labels, weights=weights)

        self.params = tf.trainable_variables()
        self.gradients = tf.gradients(self.pretrain_loss, self.params)
        self.clipped_gradients, _ = tf.clip_by_global_norm(
            self.gradients, self.grad_clip)

        optimizer = self.g_optimizer(self.learning_rate)
        self.pretrain_updates = optimizer.apply_gradients(
            zip(self.clipped_gradients, self.params))

        
        # Adversarial optimiwzation
        self.g_loss = -tf.reduce_sum(
            tf.reduce_sum(
                tf.one_hot(tf.to_int32(tf.reshape(self.labels, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                    tf.clip_by_value(tf.reshape(self.word_probas, [-1, self.num_emb]), 1e-20, 1.0)
                ), 1) * tf.reshape(self.rewards, [-1])
        )
        g_opt = self.g_optimizer(self.learning_rate)
        self.g_grad, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, self.params), self.grad_clip)
        self.g_updates = g_opt.apply_gradients(zip(self.g_grad, self.params))


    def generate(self, sess, X):
        """
        Returns a sentence output by generator given history
        Args: 
            X: history
        """
        outputs = sess.run(self.gen_x, feed_dict=feed_dict)
        return outputs

    def assign_emb(self, sess):
        sess.run(self.embedding_init)
        return

    def pretrain_step(self, sess, X, Y):
        """
        Run seq2seq training step
        Args:
            X: dialogue history
            Y: expected sentence
        """
        feed_dict = {enc_input[t]: X[t] for t in rqnge(seq_length)}
        feed_dict.update({labels[t]: Y[t] for t in range(rep_seq_length)})
        outputs = sess.run([self.pretrain_updates, self.pretrain_loss], feed_dict=feed_dict)
        return outputs

    def adv_update(sess, sentence, word_probas, rewards, baseline):
        """
        Run adversarial training step
        Args:
            sentence: sentence output by generator
            word_probas: proba of each word in the generate sentence
            baseline: baseline[t] = baseline(sentence[0:t])
            rewards: rewards[t] = rewards(sentence[0:t])
        """
        feed_dict = {self.labels= sentence, self.word_probas: word_probas, self.baseline: baseline, self.rewards = rewards}
        sess.run(self.g_updates, feed_dict)

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        return tf.zeros(shape)

    def g_optimizer(self, learning_rate):
        return tf.train.AdamOptimizer(learning_rate)

def test_advtrain():
    """
    Adversarial training for G1 and D1
    """
    # Load G1 and D1
    g_steps = 1
    d_steps = 5
    train_set_size = 5000
    batch√size 50
    max_seq_length = 200
    mc_num = 5

    embedding_matrix, hist_train, hist_val, reply_train, reply_val,
    word_index = readFBTask1Seq2Seq.crate_con(True, max_seq_length)

    embedding_dim = len(word_index+1)

    G1 = Generator()

    for i in range(g_steps):
        for j in range(train_set_size/batch_size):
            X = hist_train[idxTrain[j*batch_size:(j+1)*batch_size],:]
            Y = reply_train[idxTrain[j*batch_size:(j+1)*batch_size],:]
            
            # Shuffle
            X = np.array(X).T
            Y = np.array(Y).T

            feed_dict = {enc_input[t]: X[t] for t in rqnge(seq_length)}
            pretrain_step(feed_dict)
 

def test_advtrain():
    """
    Adversarial training for G1 and D1
    """
    # Load G1 and D1
    g_steps = 1
    d_steps = 5
    train_set_size = 5000
    batch√size 50
    max_seq_length = 200
    mc_num = 5

    embedding_matrix, hist_train, hist_val, reply_train, reply_val,
    word_index = readFBTask1Seq2Seq.crate_con(True, max_seq_length)

    embedding_dim = len(word_index+1)

    G1 = Generator()
    D1 = Discriminator(max_seq_length, word_index, embedding_train)
    rollout = WordRollout(G1)
    baseline = Baseline

    for i in range(g_steps):
        for j in range(train_set_size/batch_size):
            X = hist_train[idxTrain[j*batch_size:(j+1)*batch_size],:]
            Y = reply_train[idxTrain[j*batch_size:(j+1)*batch_size],:]
            
            # Shuffle
            X = np.array(X).T
            Y = np.array(Y).T

            feed_dict = {enc_input[t]: X[t] for t in rqnge(seq_length)}
            word_probas,sentence = generator.generate(feed_dict)

            rewards = rollout.get_rewards(sentence, mc_num,
                    discriminator)
            # TODO
            #baseline = baseline.get_values(sentence)

            generator.adv_update(sess, sentence, word_probas, rewards, baseline)

    for i in range(d_steps):
        X = hist_train[idxTrain[j*batch_size:(j+1)*batch_size],:]
        Y+ = reply_train[idxTrain[j*batch_size:(j+1)*batch_size],:]
        
        # Shuffle
        X = np.array(X).T
        Y = np.array(Y).T
 
        feed_dict = {enc_input[t]: X[t] for t in rqnge(seq_length)}
        
        sentence_distro = generator.get_sentence_distro(feed_dict)
        Y_ = distro2sentence(sentence_distro)

        # Make 2 batches out of Y_ and Y+
        X1, X1, Y1, Y2 = make_batches(X,Y+,Y_)
        disc.train(X1, Y1, batch_size)
        disc.train(X2, Y2, batch_size)
            

def adv_train_2():


def interactive_train():


