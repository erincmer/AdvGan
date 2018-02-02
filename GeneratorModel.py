import numpy as np
import tensorflow as tf

from tensorflow.python.layers import core as layers_core

from tensorflow.contrib.seq2seq.python.ops.helper import MonteCarloEmbeddingHelper as MonteCarloHelper


def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)

class Generator(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
                 sequence_length, rep_sequence_length, start_token, end_token,gen_name,
                 learning_rate=0.00001, reward_gamma=1.00):
        self.name = gen_name
        with tf.variable_scope(gen_name) as scope:
            self.num_emb = num_emb  # vocab size
            self.batch_size = batch_size
            self.emb_dim = emb_dim
            self.hidden_dim = hidden_dim
            self.sequence_length = sequence_length
            self.rep_sequence_length = rep_sequence_length
            self.end_token = end_token

            self.start_tokens = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
            self.end_tokens = tf.constant([end_token] * self.batch_size, dtype=tf.int32)
            self.start_tokens_check = tf.constant([start_token + 5] * self.batch_size, dtype=tf.int32)
            # self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
            self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
            self.learning_rate_placeholer =  tf.placeholder(tf.float32)
            self.reward_gamma = reward_gamma
            self.g_params = []
            self.temperature = 1.0
            self.grad_clip = 5.0
            self.expected_reward = tf.Variable(tf.zeros([self.sequence_length]))

            self.prev_mem = tf.zeros((batch_size, self.hidden_dim))  # h_{t-1}
            self.enc_inp = tf.placeholder(tf.int32, shape=[batch_size, self.sequence_length],
                                          name="encoderInputs")  # history
            self.labels = tf.placeholder(tf.int32, shape=[batch_size, self.rep_sequence_length],
                                         name="labels")  # expected sentence
            self.sentence = tf.placeholder(tf.int32, shape=[batch_size, self.rep_sequence_length],
                                           name="sentence")  # generated sentence

            # self.start_tokens = tf.zeros([batch_size],tf.int32)
            self.dec_inp = tf.concat([tf.expand_dims(self.start_tokens, 1), self.labels], 1)

            self.input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(self.enc_inp, self.end_token)), 1)
            self.output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(self.dec_inp, self.end_token)), 1)
            self.output_lengths_full = tf.constant([self.rep_sequence_length] * self.batch_size, dtype=tf.int32)
            self.g_embeddings = tf.Variable(tf.constant(0.0, shape=[self.num_emb, self.emb_dim]),
                                            trainable=False, name="W")
            self.embedding_placeholder = tf.placeholder(tf.float32, [self.num_emb, self.emb_dim])
            self.embedding_init = self.g_embeddings.assign(self.embedding_placeholder)
            self.lr_init = self.learning_rate.assign(self.learning_rate_placeholer)
            # rewards[t] : rewards of the first t words of the generated sentence
            # baseline[t] : baseline of the first t words
            # word_proba[t] = p(y_t | X, y_{0:t-1}
            self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.rep_sequence_length], name="rewards")
            self.baseline = tf.placeholder(tf.float32, shape=[self.batch_size, self.rep_sequence_length], name="baseline")

            # Interactive training feed dict
            self.sentence_proba = tf.placeholder(
                    tf.float32,
                    shape=[64,None],
                    name="sentence_proba")
            self.sentence_rewards = tf.placeholder(
                    tf.float32,
                    shape=[64,None],
                    name="sentence_rewards")
            self.sentence_baseline = tf.placeholder(
                    tf.float32,
                    shape=[64,None],
                    name="sentence_baseline")


            with tf.device("/cpu:0"):
                # processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.enc_inp), perm=[1, 0, 2])
                # processed_y = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.dec_inp), perm=[1, 0, 2])
                processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.enc_inp), perm=[0, 1, 2])
                processed_y = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.dec_inp), perm=[0, 1, 2])


            # Encoder definition
            self.enc_cell = tf.contrib.rnn.GRUCell(self.hidden_dim)
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(self.enc_cell, processed_x, self.input_lengths ,self.prev_mem)

            # Decoder definition
            train_helper = tf.contrib.seq2seq.TrainingHelper(processed_y, self.output_lengths_full)
            sampling_prob = tf.Variable(0.0, dtype=tf.float32)
            # pred_helper = tf.contrib.seq2seq.SampleEmbeddingHelper(sample,numpy.full((self.batch_size),self.rep_sequence_length),tf.int32, processed_y,end_fn,next_inputs)
            #
            test_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.g_embeddings, start_tokens=tf.to_int32(self.start_tokens), end_token=200000)
            pred_helper = MonteCarloHelper(processed_y, self.output_lengths,self.output_lengths_full, self.g_embeddings,
                                           start_tokens=tf.to_int32(self.start_tokens),end_token =self.end_token,  end_tokens=tf.to_int32(self.end_tokens),
                                           softmax_temperature=self.temperature, seed=1881)




            # # pred_helper = tf.contrib.seq2seq.TrainingHelper(processed_y, self.output_lengths_full)
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
                        impute_finished=False, maximum_iterations=self.rep_sequence_length
                    )
                    return outputs

            self.train_outputs = decode(train_helper, 'decode'+gen_name)
            self.gen_x = decode(pred_helper, 'decode'+gen_name, reuse=True)
            self.test_x = decode(test_helper, 'decode'+gen_name, reuse=True)
            # tf.identity(self.train_outputs.sample_id[0], name='train_pred')
            # tf.identity(self.train_outputs.rnn_output[0], name='train_pred')

            # weights = tf.to_float(tf.not_equal(self.dec_inp[:, :-1], self.end_token))

            weights = tf.to_float(tf.not_equal(self.dec_inp[:, :-1], 400))

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
            self.pred_output = tf.nn.softmax(self.gen_x[0].rnn_output)
            self.test_output = tf.nn.softmax(self.test_x[0].rnn_output)
            self.pred_train_output = self.train_outputs[0].rnn_output

            self.pred_output_ids = self.gen_x[0].sample_id
            self.test_output_ids = self.test_x[0].sample_id
            self.pred_train_output_ids = self.train_outputs[0].sample_id

            self.gen_sentence_proba = tf.reduce_prod(
                    tf.reduce_sum(
                        tf.one_hot(
                            tf.to_int32( self.pred_output_ids),
                            self.num_emb, 1.0, 0.0) *
                        tf.clip_by_value( 
                            tf.nn.softmax(self.gen_x[0].rnn_output),
                            1e-20, 1.0),
                        2),
                    1) # proba of the generated sentence
            
            self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                    scope=self.name)
            self.saver = tf.train.Saver(var_list=self.params)
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

            self.g_loss = -tf.reduce_mean(tf.reduce_sum(tf.one_hot(tf.to_int32(tf.reshape(self.sentence, [-1])), self.num_emb, 1.0, 0.0) *
                                        tf.log(
                                  tf.clip_by_value(tf.reshape(tf.nn.softmax(self.gen_x[0].rnn_output), [-1, self.num_emb]), 1e-20,
                                                   1.0)),1)* (tf.reshape(self.rewards, [-1]) - tf.reshape(self.baseline, [-1])))

            self.g_part0 =  tf.one_hot(tf.to_int32(tf.reshape(self.sentence, [-1])), self.num_emb, 1.0, 0.0)

            self.g_part1 =  tf.log(tf.clip_by_value(tf.nn.softmax(tf.reshape(self.gen_x[0].rnn_output, [-1, self.num_emb])), 1e-20,1.0))

            self.g_part2 = tf.one_hot(tf.to_int32(tf.reshape(self.sentence, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                                  tf.clip_by_value(tf.reshape(tf.nn.softmax(self.gen_x[0].rnn_output), [-1, self.num_emb]), 1e-20,
                                                   1.0))
            self.g_part3 = -tf.reduce_sum(tf.one_hot(tf.to_int32(tf.reshape(self.sentence, [-1])), self.num_emb, 1.0, 0.0) *tf.log(
                                  tf.clip_by_value(tf.reshape(tf.nn.softmax(self.gen_x[0].rnn_output), [-1, self.num_emb]), 1e-20,
                                                   1.0)),1)
            self.g_part4 = (tf.reshape(self.rewards, [-1]) - tf.reshape(self.baseline, [-1]))

            # self.g_loss = -tf.reduce_sum(
            #     tf.reduce_mean(tf.one_hot(tf.to_int32(tf.reshape(self.sentence, [-1])), self.num_emb, 0.0, 1.0) *tf.log(tf.clip_by_value(tf.reshape(tf.nn.softmax(self.pred_output[0]),
            #                                                                                                                                         [-1, self.num_emb]), 1e-20,1.0)))
            #     * (tf.reshape(self.rewards, [-1]) - tf.reshape(self.baseline, [-1])))
            g_opt = self.g_optimizer(self.learning_rate)
            self.g_grad, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, self.params), self.grad_clip)
            self.g_updates = g_opt.apply_gradients(zip(self.g_grad, self.params))

            # PPO optimization V1 standalone
            # I use only the proba of the words inside the generated sentence to
            # compute the ratio. I am not sure if it is the correct way.
            #self.old_distro = tf.placeholder(
            #        tf.float32, 
            #        shape=[self.batch_size, self.rep_sequence_length, self.emb_dim],
            #        name="old_distro")
            #self.clip_param = 0.2

            ## Normalize rewards (maybe unnecessary) and flatten it
            #mean = tf.reduce_mean(self.rewards, axis=[1], keep_dims=True)
            #var = tf.reduce_mean(tf.square(self.rewards-mean), axis=[1], keep_dims=False)
            #std = tf.expand_dims(tf.sqrt(var), 1)
            #atarg = tf.reshape((self.rewards - mean)/std, [-1])  # A estimator
            #print("atarg.get_shape(): ", atarg.get_shape())

            ## Flatten probas
            #new_proba = tf.reduce_sum( # proba of words in the generated sentence only
            #        tf.one_hot(
            #            tf.to_int32(tf.reshape(self.sentence, [-1])),
            #            self.num_emb, 1.0, 0.0) *  tf.clip_by_value(
            #                tf.reshape(tf.nn.softmax(self.gen_x[0].rnn_output), 
            #                    [-1, self.num_emb]), 
            #                1e-20, 1.0),1)
            #old_proba = tf.reduce_sum( # old proba of words in the generated sentence only
            #        tf.one_hot(
            #            tf.to_int32(tf.reshape(self.sentence, [-1])),
            #            self.num_emb, 1.0, 0.0) *  tf.clip_by_value(
            #                tf.reshape(self.old_distro, 
            #                    [-1, self.num_emb]), 
            #                1e-20, 1.0),1)

            #ratio = new_proba / old_proba # r
            #print("ratio.get_shape(): ", ratio.get_shape())
            #surr1 = ratio * atarg # r * A
            ## clipped version of r*A
            #surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * atarg
            #pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # -L^CLIP
            #
            ## Optimization
            #self.ppo_loss = pol_surr
            #ppo_opt = tf.train.AdamOptimizer(self.learning_rate)
            #self.ppo_grad, _ = tf.clip_by_global_norm(tf.gradients(self.ppo_loss, self.params), self.grad_clip)
            #self.ppo_updates = ppo_opt.apply_gradients(zip(self.ppo_grad, self.params))

            # End of PPO V1 standalone
            
            # PPO optimization V2 standalone
            # Use a copy of the previous policy
            self.clip_param = 0.2
            self.old_distro = tf.placeholder(
                    tf.float32, 
                    shape=[self.batch_size, self.rep_sequence_length, self.emb_dim],
                    name="old_distro")
            self.ppo_mask = tf.placeholder( # To get rid of eos
                    tf.float32,
                    shape=[self.batch_size, self.rep_sequence_length])

            # Normalize rewards (maybe unnecessary) and flatten it
            mean = tf.reduce_mean(self.rewards, axis=[1], keep_dims=True)
            var = tf.reduce_mean(tf.square(self.rewards-mean), axis=[1], keep_dims=False)
            std = tf.expand_dims(tf.sqrt(var), 1)
            atarg = tf.reshape((self.rewards - mean)/std, [-1])  # A estimator
            print("atarg.get_shape(): ", atarg.get_shape())

            # Flatten probas
            new_proba = tf.reduce_sum( # proba of words in the generated sentence only
                    tf.one_hot(
                        tf.to_int32(tf.reshape(self.sentence, [-1])),
                        self.num_emb, 1.0, 0.0) *  tf.clip_by_value(
                            tf.reshape(tf.nn.softmax(self.gen_x[0].rnn_output), 
                                [-1, self.num_emb]), 
                            1e-20, 1.0),1)
            #old_proba = tf.reshape(self.p_old, [-1]) 
            old_proba = tf.reduce_sum( # old proba of words in the generated sentence only
                    tf.one_hot(
                        tf.to_int32(tf.reshape(self.sentence, [-1])),
                        self.num_emb, 1.0, 0.0) *  tf.clip_by_value(
                            tf.reshape(self.old_distro, 
                                [-1, self.num_emb]), 
                            1e-20, 1.0),1)

            ratio = (new_proba / old_proba) * tf.reshape(self.ppo_mask,[-1]) # r
            print("ratio.get_shape(): ", ratio.get_shape())
            surr1 = ratio * atarg # r * A
            # clipped version of r*A
            surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * atarg
            pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # -L^CLIP
            
            # Optimization
            self.ppo_loss = pol_surr
            ppo_opt = tf.train.AdamOptimizer(self.learning_rate)
            self.ppo_grad, _ = tf.clip_by_global_norm(tf.gradients(self.ppo_loss, self.params), self.grad_clip)
            self.ppo_updates = ppo_opt.apply_gradients(zip(self.ppo_grad, self.params))

            # End of PPO V1 standalone

            # Reinforce optimization interact V1 
            # Use other agent reward as baseline
            self.last_sentence_proba = tf.reduce_prod(
                    tf.reduce_sum(
                        tf.one_hot(
                            tf.to_int32(
                                self.sentence), 
                            self.num_emb, 1.0, 0.0) * tf.clip_by_value(
                                tf.nn.softmax(self.gen_x[0].rnn_output), 
                                1e-20, 1.0),
                            2),
                    1)

            self.inter_loss = -tf.reduce_sum(
                    tf.reduce_sum( 
                        self.sentence_proba[:,:-1] * (
                            self.sentence_rewards[:,:-1] 
                            - self.sentence_baseline[:,:-1])
                        , 1) - self.last_sentence_proba *(
                            self.sentence_rewards[:,-1] 
                            - self.sentence_baseline[:,-1]) )

            inter_opt = tf.train.AdamOptimizer(self.learning_rate)
            self.inter_grad, _ = tf.clip_by_global_norm(tf.gradients(self.inter_loss, self.params), self.grad_clip)
            self.inter_updates = inter_opt.apply_gradients(zip(self.inter_grad, self.params))
            # End of Reinforce optimization standalone
    
    def test_eq(self, sess, g_old):
        oldpi_var = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, 
                scope = g_old.name)
        pi_var = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, 
                scope = self.name)
        #print("\noldpi_var: ", oldpi_var)
        #print("\npi_var: ", pi_var)

        test_eq_before = [tf.reduce_all(tf.equal(oldv, newv)) for (oldv,newv) in zipsame(oldpi_var, pi_var)]
        v = sess.run(test_eq_before)
        v = np.array(v)

        if np.sum(v)==np.prod(v.shape):
            print("They are equal")
            return True
        else:
            print("Not equal")
            return False
    
    def copy(self, sess, g_old):
        """ 
        Copy g_old into self
        Args: 
            g_old: current policy to save
        """
        oldpi_var = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, 
                scope = g_old.name)
        pi_var = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, 
                scope = self.name)

        assign_old_eq_new = [tf.assign(oldv, newv) for (oldv,newv) in zipsame(oldpi_var, pi_var)]
        sess.run(assign_old_eq_new) 

    def test_copy(self, sess, g_old):
        """ 
        Test copy g_old into self
        Args: 
            g_old: current policy to save
        """
        
        oldpi_var = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, 
                scope = g_old.name)
        pi_var = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, 
                scope = self.name)
        #print("\noldpi_var: ", oldpi_var)
        #print("\npi_var: ", pi_var)

        test_eq_same = [tf.reduce_all(tf.equal(oldv, newv)) for (oldv,newv) in zipsame(pi_var, pi_var)]
        
        test_eq_before = [tf.reduce_all(tf.equal(oldv, newv)) for (oldv,newv) in zipsame(oldpi_var, pi_var)]
        
        assign_old_eq_new = [tf.assign(oldv, newv) for (oldv,newv) in zipsame(oldpi_var, pi_var)]
        old_name = [oldv.name for (oldv,newv) in zipsame(oldpi_var, pi_var)]
        new_name = [newv.name for (oldv,newv) in zipsame(oldpi_var, pi_var)]

        #print("\nzipsame: ")
        #test_print = [print("oldv-newv: ", oldv, " ", newv) for (oldv,newv) in zipsame(oldpi_var, pi_var)]
        with tf.control_dependencies(assign_old_eq_new):
            test_eq = [tf.reduce_all(tf.equal(oldv, newv)) for (oldv,newv) in zipsame(oldpi_var, pi_var)]

        _,v_eq, v_eq_before, v_name, v_name_before, v_same = sess.run([assign_old_eq_new, 
            test_eq, 
            test_eq_before, 
            old_name, 
            new_name,
            test_eq_same])
        # DEBUG 
        #i=0
        #for (oldv,newv) in zipsame(oldpi_var, pi_var):
        #    if ~v_eq_before[i]:
        #        print("Before var diff: ", oldv.name)
        #    i+=1
        #
        #i=0
        #for (oldv,newv) in zipsame(oldpi_var, pi_var):
        #    if ~v_eq[i]:
        #        print("After var diff: ", oldv.name)
        #    i+=1

        #i=0
        #for (oldv,newv) in zipsame(oldpi_var, pi_var):
        #    if ~v_same[i]:
        #        print("Test same: ", oldv.name)
        #    i+=1

    def ppo_step(self, sess, history, labels, sentence, rewards, old_distro,
            ppo_mask):
        """
        Computes sentence from given history and compute loss given reward and
        baseline
        Args:
            sess: tf session
            sentence: sentence output by generator
            history: Dialogue history used to generate sentence
            labels: Expected reply following history
            setence: Reply generated by the generator following history
            rewards: MC-Discriminator rewards for the generated sentence
            old_distro: Distributions over actions at the previous step 
            ppo_mask: 'eos' mask over sentence
        """
        feed_dict = {self.enc_inp: history, 
                self.labels: labels, 
                self.sentence: sentence, 
                self.rewards: rewards,
                self.old_distro: old_distro, 
                self.ppo_mask: ppo_mask}
        outputs = sess.run([self.ppo_updates, self.ppo_loss], feed_dict)
        return outputs
 

    def pad_dim(self,input_tensor):
        padding = tf.tile([[0]], tf.stack( [self.batch_size, self.rep_sequence_length - self.output_lengths], 0))
           # [tf.shape(input_tensor)[0], self.rep_sequence_length - tf.shape(input_tensor)[1]], 0))


        return tf.concat([input_tensor, padding], 1)

    def generate(self, sess, x, y):
        outputs, ids = sess.run([self.pred_output, self.pred_output_ids], feed_dict={self.enc_inp: x, self.labels: y})
        return outputs, ids


    def gen_proba_sentence(self, sess, x, y):
        """
        Generates sentence and returns the proba of the word in the sentence
        Args:
            sess: tf session
            x: history
            y: ground truth sentence
        """
        proba, ids = sess.run([self.gen_sentence_proba, self.pred_output_ids], feed_dict={self.enc_inp: x, self.labels: y})
        return proba, ids

    
    def test_generate(self, sess, x, y):
        outputs, ids = sess.run([self.test_output, self.test_output_ids], feed_dict={self.enc_inp: x, self.labels: y})


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
    def assign_lr(self, sess, x):
        sess.run(self.lr_init, feed_dict={self.learning_rate_placeholer: x})
        return
    def pretrain_step(self, sess, x, y):
        outputs = sess.run([self.pretrain_updates, self.pretrain_loss, self.pred_train_output_ids],
                           feed_dict={self.enc_inp: x, self.labels: y})
        return outputs
    def get_pretrain_loss(self, sess, x, y):
        outputs = sess.run( self.pretrain_loss,
                           feed_dict={self.enc_inp: x, self.labels: y})
        return outputs
    def get_adv_loss(self, sess, history, labels, sentence, rewards, baseline):
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
        outputs = sess.run([self.g_part0,self.g_part1,self.g_part2,self.g_part3,self.g_part4, self.g_loss], feed_dict)
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

    def inter_train_step(self, sess, history, labels, sentence, proba, rewards, baseline):
        """
        Computes sentence from given history and compute loss given reward and
        baseline
        Args:
            sess: tf session
            sentence: sentence output by generator
            rewards: reward for each sentence in history
            baseline: baseline for each sentence in history
        """
        feed_dict = {self.enc_inp: history, 
                self.labels: labels, 
                self.sentence: sentence,
                self.sentence_proba: proba,
                self.sentence_rewards: rewards,
                self.sentence_baseline: baseline}
        outputs = sess.run([self.inter_updates, self.inter_loss], feed_dict)
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
            while h[i] != word_index['eoh']:
                disc_inp[counter, i] = h[i]
                i = i + 1

            disc_inp[counter, i] = word_index['eoh']

            disc_inp[counter, i + 1:i + self.rep_sequence_length + 1] = r
            counter = counter + 1

        return disc_inp

    def convert_id_to_text(self,ids, word_index):


            sen = ""
            sen_list = []
            for id in ids:
                for i in id:
                    if i != 0 and i != word_index["eos"]:
                        sen = sen + " " + list(word_index.keys())[list(word_index.values()).index(i)]
                sen_list.append(sen)
                sen = ""
            return sen_list

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
                # if sentence[:, (t - 1)].any() != word_index['eos']:
                    history_update = np.copy(history)

                    # Matrix [batch_size, rep_seq_length]
                    # Line l: the first t elements are sentence[l,0:t]
                    gen_input_t = np.ones(([self.batch_size, self.rep_sequence_length])) * word_index['eos']

                    # DEBUG
                    # print("t: ", t)
                    # print("sentence.shape: ", sentence.shape)
                    # print("gen_input_t.shape: ", gen_input_t.shape)
                    # print("gen_input_t[:,0:t].shape: ", gen_input_t[:,0:t].shape)
                    # print("sentence[:,0:t].shape: ", sentence[:,0:t].shape)

                    gen_input_t[:, 0:t] = sentence[:, 0:t]

                    # Ask gen to output a sentence using the first t tokens
                    # of the complete sentence $sentence
                    _, complete_sentence = self.generate(sess, history_update, gen_input_t)
                    rep_inp = np.full((self.batch_size, self.rep_sequence_length), word_index['eos'])
                    rep_inp[:, :complete_sentence.shape[1]] = complete_sentence
                    complete_sentence = rep_inp
                    complete_sentence[complete_sentence == 0] = word_index['eos']
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
                    # print("Sampled Sentence for ----",self.convert_id_to_text(gen_input_t[0],word_index)," ----- is = ",self.convert_id_to_text(complete_sentence[0],word_index))
                    history_update = self.concat_hist_reply(history_update, complete_sentence, word_index)
                    disc_proba = discriminator.get_rewards(sess,history_update)
                    disc_reward = np.array([item[1] for item in disc_proba])
                # print("Reward for Complete Sentence ----- ", self.convert_id_to_text(complete_sentence[0], word_index), " ----- is = ",disc_reward[0])
                #     rewards[:, (t - 1)] += disc_reward  # disc_reward.reshape(self.batch_size, 1)

                    if t ==1:
                        rewards[:, (t - 1)] += disc_reward * (sentence[:, (t - 1)] != word_index['eos'])
                    else:
                        rewards[:, (t - 1)] += disc_reward * (sentence[:, (t - 2)] != word_index['eos'])
                # print("History for Complete Sentence ----- "," is -------\n",self.convert_id_to_text(history_update[0:1], word_index))
                #
                # print("Sampled Sentence for ----", self.convert_id_to_text(gen_input_t[0:3], word_index),
                #           " ----- is = \n ", self.convert_id_to_text(complete_sentence[0:3], word_index))
                # print("Reward for Complete Sentence ----- ",
                #           self.convert_id_to_text(complete_sentence[0:3], word_index), " ----- is = \n", disc_reward[0:3])

                # baseline[:, t - 1] = np.squeeze(baseline_val) * (sentence[:, (t - 1)] != word_index['eos'])
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

def main():
    # Test pre train
    pass

if __name__ == '__main__':
    main()
