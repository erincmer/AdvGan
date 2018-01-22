import numpy as np
import tensorflow as tf

from tensorflow.python.layers import core as layers_core

from tensorflow.contrib.seq2seq.python.ops.helper import MonteCarloEmbeddingHelper as MonteCarloHelper


class Generator(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
                 sequence_length, rep_sequence_length, start_token, end_token, hist_end_token,
                 learning_rate=0.0004, reward_gamma=1.00):

        self.num_emb = num_emb  # vocab size
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.rep_sequence_length = rep_sequence_length
        self.end_token = end_token
        self.hist_end_token = end_token
        self.start_tokens = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.start_tokens_check = tf.constant([start_token + 5] * self.batch_size, dtype=tf.int32)
        # self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.reward_gamma = reward_gamma
        self.g_params = []
        self.temperature = 1.0
        self.grad_clip = 5.0
        self.expected_reward = tf.Variable(tf.zeros([self.sequence_length]))
        self.saver = tf.train.Saver()
        self.prev_mem = tf.zeros((batch_size, self.hidden_dim))  # h_{t-1}
        self.enc_inp = tf.placeholder(tf.int32, shape=[batch_size, self.sequence_length],
                                      name="encoderInputs")  # history
        self.labels = tf.placeholder(tf.int32, shape=[batch_size, self.rep_sequence_length],
                                     name="labels")  # expected sentence
        self.sentence = tf.placeholder(tf.int32, shape=[batch_size, self.rep_sequence_length],
                                       name="sentence")  # generated sentence

        # self.start_tokens = tf.zeros([batch_size],tf.int32)
        self.dec_inp = tf.concat([tf.expand_dims(self.start_tokens, 1), self.labels], 1)

        self.input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(self.enc_inp, self.hist_end_token)), 1)
        self.output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(self.dec_inp, self.end_token)), 1)
        self.output_lengths_full = tf.constant([self.rep_sequence_length] * self.batch_size, dtype=tf.int32)
        self.g_embeddings = tf.Variable(tf.constant(0.0, shape=[self.num_emb, self.emb_dim]),
                                        trainable=False, name="W")
        self.embedding_placeholder = tf.placeholder(tf.float32, [self.num_emb, self.emb_dim])
        self.embedding_init = self.g_embeddings.assign(self.embedding_placeholder)

        # rewards[t] : rewards of the first t words of the generated sentence
        # baseline[t] : baseline of the first t words
        # word_proba[t] = p(y_t | X, y_{0:t-1}
        self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.rep_sequence_length], name="rewards")
        self.baseline = tf.placeholder(tf.float32, shape=[self.batch_size, self.rep_sequence_length], name="baseline")
        self.word_probas = tf.placeholder(tf.float32, shape=[self.batch_size, self.rep_sequence_length],
                                          name="wordprobas")

        self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size,
                                                         self.rep_sequence_length])  # get from rollout policy and discriminator
        with tf.device("/cpu:0"):
            # processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.enc_inp), perm=[1, 0, 2])
            # processed_y = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.dec_inp), perm=[1, 0, 2])
            processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.enc_inp), perm=[0, 1, 2])
            processed_y = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.dec_inp), perm=[0, 1, 2])

        # Encoder definition
        self.enc_cell = tf.contrib.rnn.GRUCell(self.hidden_dim)
        self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(self.enc_cell, processed_x, self.input_lengths,
                                                                     self.prev_mem)

        # Decoder definition
        train_helper = tf.contrib.seq2seq.TrainingHelper(processed_y, self.output_lengths_full)
        sampling_prob = tf.Variable(0.0, dtype=tf.float32)
        # pred_helper = tf.contrib.seq2seq.SampleEmbeddingHelper(sample,numpy.full((self.batch_size),self.rep_sequence_length),tf.int32, processed_y,end_fn,next_inputs)
        #
        # pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        #     self.g_embeddings, start_tokens=tf.to_int32(self.start_tokens), end_token=200)
        pred_helper = MonteCarloHelper(processed_y, self.output_lengths, self.g_embeddings,
                                       start_tokens=tf.to_int32(self.start_tokens), end_token=self.end_token,
                                       softmax_temperature=self.temperature, seed=1881)

        # pred_helper = tf.contrib.seq2seq.TrainingHelper(processed_y, self.output_lengths_full)
        # train_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
        #     processed_y, self.output_lengths_full,
        #     self.g_embeddings,
        #     sampling_probability=0.0)
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

                projection_layer = layers_core.Dense(self.num_emb, use_bias=False)

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=out_cell, helper=helper,
                    initial_state=out_cell.zero_state(
                        dtype=tf.float32, batch_size=batch_size),
                    # initial_state=self.encoder_state,
                    output_layer=projection_layer
                )

                outputs = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder, output_time_major=False,
                    impute_finished=True, maximum_iterations=self.rep_sequence_length
                )
                return outputs

        self.train_outputs = decode(train_helper, 'decode')
        self.gen_x = decode(pred_helper, 'decode', reuse=True)

        # tf.identity(self.train_outputs.sample_id[0], name='train_pred')
        # tf.identity(self.train_outputs.rnn_output[0], name='train_pred')

        weights = tf.to_float(tf.not_equal(self.dec_inp[:, :-1], self.end_token))
        # print(self.labels.get_shape())
        # print(weights.get_shape())
        # print(self.train_outputs[0].rnn_output.get_shape())
        # input("wait")
        self.pretrain_loss = tf.contrib.seq2seq.sequence_loss(
            logits=self.train_outputs[0].rnn_output, targets=self.labels, weights=weights)

        # crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=self.labels, logits=self.train_outputs.rnn_output)
        #
        # self.pretrain_loss = (tf.reduce_sum(crossent * weights) /
        #               batch_size)
        self.pred_output = self.gen_x[0].rnn_output
        self.pred_train_output = self.train_outputs[0].rnn_output

        self.pred_output_ids = self.gen_x[0].sample_id
        self.pred_train_output_ids = self.train_outputs[0].sample_id

        self.params = tf.trainable_variables()
        self.gradients = tf.gradients(self.pretrain_loss, self.params)
        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, self.grad_clip)

        optimizer = self.g_optimizer(self.learning_rate)
        self.pretrain_updates = optimizer.apply_gradients(
            zip(self.clipped_gradients, self.params))

        # Adversarial optimization
        # DEBUG
        # print("\ntrain variables: \n",tf.trainable_variables())
        # print("self.sentence.get_shpae(): ", self.sentence.get_shape())
        # print("self.rewards.get_shape()): ", self.rewards.get_shape())
        # print("self.baseline.get_shape()): ", self.baseline.get_shape())

        self.g_loss = -tf.reduce_mean(
            tf.reduce_mean(tf.one_hot(tf.to_int32(tf.reshape(self.sentence, [-1])), self.num_emb, 0.0, 1.0) *
                          tf.log(
                              tf.clip_by_value(tf.reshape(tf.nn.softmax(self.train_outputs[0].rnn_output), [-1, self.num_emb]), 1e-20,
                                               1.0)))
            * (tf.reshape(self.rewards, [-1]) - tf.reshape(self.baseline, [-1])))

        g_opt = self.g_optimizer(self.learning_rate)
        self.g_grad, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, self.params), self.grad_clip)
        self.g_updates = g_opt.apply_gradients(zip(self.g_grad, self.params))

    def pad_dim(self,input_tensor):
        padding = tf.tile([[0]], tf.stack( [self.batch_size, self.rep_sequence_length - self.output_lengths], 0))
           # [tf.shape(input_tensor)[0], self.rep_sequence_length - tf.shape(input_tensor)[1]], 0))


        return tf.concat([input_tensor, padding], 1)

    def generate(self, sess, x, y):
        outputs, ids = sess.run([self.pred_output, self.pred_output_ids], feed_dict={self.enc_inp: x, self.labels: y})


        return outputs, ids

    def generate_train(self, sess, x, y):
        outputs, ids = sess.run([self.pred_train_output, self.pred_train_output_ids],
                                feed_dict={self.enc_inp: x, self.labels: y})
        return outputs, ids

    def restore_model(self,sess,savepath):
        self.saver.restore(sess, tf.train.latest_checkpoint(savepath))
        return
    def save_model(self,sess,savepath):
        self.saver.save(sess, savepath + 'my-model-sentence-sen-1024')

        return
    def assign_emb(self, sess, x):
        sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: x})
        return

    def pretrain_step(self, sess, x, y):
        outputs = sess.run([self.pretrain_updates, self.pretrain_loss, self.pred_train_output],
                           feed_dict={self.enc_inp: x, self.labels: y})
        return outputs


    def advtrain_step(self, sess, history, labels, sentence, rewards, baseline):
        """
        Computes sentence from given history and compute loss given reward and
        baseline
        Args:
            sess: tf session
            sentence: sentence output by generator
            rewards:
            baseline
        """
        feed_dict = {self.enc_inp: history, self.labels: labels, self.sentence: sentence, self.rewards: rewards,
                     self.baseline: baseline}
        outputs = sess.run([self.g_updates, self.g_loss], feed_dict)
        return outputs

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        return tf.zeros(shape)

    def concat_hist_reply(self, histories, replies, word_index):

        disc_inp = np.full((self.batch_size, self.sequence_length), word_index['eos'])
        rep_inp = np.full((self.batch_size, self.rep_sequence_length), word_index['eos'])
        rep_inp[:, :replies.shape[1]] = replies
        counter = 0
        for h, r in zip(histories, rep_inp):

            i = 0
            while i != word_index['eoh']:
                disc_inp[counter, i] = h[i]
                i = i + 1

            disc_inp[counter, i] = word_index['eoh']

            disc_inp[counter, i + 1:i + 21] = r
            counter = counter + 1

        return disc_inp

    def MC_reward(self, sess, history, sentence, mc_steps, discriminator,
                  word_index):
        """
        Compute rewards for every sub sentence of input_x input_x_{0:t}
        Args:
            sess: tf session
            history: dialogue history
            sentence: complete sentence from G
            mc_steps: How many times you do MC rollout
            discriminator: D
            word_index: dictionnary of {token:index}
        """
        rewards = np.zeros([self.batch_size, self.rep_sequence_length])
        for i in range(mc_steps):
            for t in range(1, self.rep_sequence_length + 1):
                history_update = np.copy(history)

                # Matrix [batch_size, rep_seq_length]
                # Line l: the first t elements are sentence[l,0:t]
                gen_input_t = np.zeros([self.batch_size, self.rep_sequence_length])

                # DEBUG
                # print("t: ", t)
                # print("sentence.shape: ", sentence.shape)
                # print("gen_input_t.shape: ", gen_input_t.shape)
                # print("gen_input_t[:,0:t].shape: ", gen_input_t[:,0:t].shape)
                # print("sentence[:,0:t].shape: ", sentence[:,0:t].shape)

                gen_input_t[:, 0:t] = sentence[:, 0:t]

                # Ask gen to output a sentence using the first t tokens
                # of the complete sentence $sentence
                _, complete_sentence = self.generate(sess, history, gen_input_t)
                rep_inp = np.full((self.batch_size, self.rep_sequence_length), word_index['eos'])
                rep_inp[:, :complete_sentence.shape[1]] = complete_sentence
                complete_sentence = rep_inp

                complete_sentence[:, 0:t] = sentence[:, 0:t]

                # print("word_index['eoh']: ", word_index['eoh'])
                # print("word_index['eos']: ", word_index['eos'])
                # print("self.hist_end_token: ", self.hist_end_token)
                # print("history: ", history)
                # print("start_insert: ", start_insert)
                # print("start_insert.shape: ", start_insert.shape)
                # print("history.shape: ", history.shape)
                # print("history_update.shape: ", history_update.shape)
                # print("complete_sentence.shape: ", complete_sentence.shape)

                # Ask disc to reward these sentences
                history_update = self.concat_hist_reply(history_update, complete_sentence, word_index)
                disc_proba = discriminator.get_rewards(history_update)
                disc_reward = np.array([item[0] for item in disc_proba])
                rewards[:, (t - 1)] += disc_reward  # disc_reward.reshape(self.batch_size, 1)

                # print("disc_proba.shape: ", disc_proba.shape)
                # print("disc_rewards.shape: ", disc_reward.shape)
                # print("rewards[:,t:(t+1)]: ", rewards[:,t:(t+1)].shape)

        # At this point, for the i-th sentence in the batch,
        # for t in [0,T-1], rewards[i,t] = \sum_{rollout} R_D(y_{0:t})
        # reward[T] = R_D(y_{0:T})

        # Average
        rewards = np.array(rewards) / (1.0 * mc_steps)  # batch_size x seq_length
        return rewards

    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer(*args, **kwargs)