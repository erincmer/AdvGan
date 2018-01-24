import numpy as np
import os
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
import tensorflow as tf

MAX_SEQ_LENGTH = 200


class DiscSentence(object):
    def __init__(self,num_emb, batch_size, emb_dim, hidden_dim,
                 max_seq_length,word_index, end_token,
                 learning_rate=0.0004):
        self.max_seq_length = max_seq_length
        self.word_index = word_index
        self.embedding_dim = len(word_index) + 1
        self.num_emb = num_emb  # vocab size
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.end_token = end_token
        self.d_embeddings = tf.Variable(tf.constant(0.0, shape=[self.num_emb, self.emb_dim]),
                                        trainable=False, name="W")
        self.embedding_placeholder = tf.placeholder(tf.float32, [self.num_emb, self.emb_dim])
        self.embedding_init = self.d_embeddings.assign(self.embedding_placeholder)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.grad_clip = 5.0
        self.labels = tf.placeholder(tf.int32, shape=[batch_size],
                                     name="labels")
        self.enc_inp = tf.placeholder(tf.int32, shape=[batch_size, self.max_seq_length],
                                      name="encoderInputs")
        self.input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(self.enc_inp, self.end_token)), 1)
        self.prev_mem = tf.zeros((batch_size, self.hidden_dim))

        with tf.device("/cpu:0"):

            processed_x = tf.transpose(tf.nn.embedding_lookup(self.d_embeddings, self.enc_inp), perm=[0, 1, 2])


            # Encoder definition
        with tf.variable_scope("disc", reuse=None) as scope:
            self.enc_cell = tf.contrib.rnn.GRUCell(self.hidden_dim)
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(self.enc_cell, processed_x, self.input_lengths,
                                                                     self.prev_mem)

            out = tf.layers.dense(self.encoder_state,100,activation=tf.nn.relu)
            logits = tf.layers.dense(out, 2)
        self.pred_train_output = tf.nn.softmax(logits)
        self.disc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(self.labels,2),logits = logits))
        self.prediction = tf.cast(tf.argmax(logits,1),tf.int32)
        self.disc_acc = tf.reduce_mean(tf.cast(tf.equal(self.prediction,self.labels),tf.float32))

        self.params = tf.trainable_variables()

        self.saver = tf.train.Saver(var_list=self.params)
        self.gradients = tf.gradients(self.disc_loss, self.params)
        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, self.grad_clip)

        optimizer = self.d_optimizer(self.learning_rate)
        self.pretrain_updates = optimizer.apply_gradients(
            zip(self.clipped_gradients, self.params))





    def get_rewards(self,sess, x):
        rewards = sess.run(self.pred_train_output,
                           feed_dict={self.enc_inp: x})
        #print('get rewards')
        return rewards
    def get_loss(self,sess, x,y):
        loss,acc = sess.run([self.disc_loss,self.disc_acc],
                           feed_dict={self.enc_inp: x, self.labels: y})
        #print('get rewards')
        return loss,acc
    def d_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer(*args, **kwargs)

    def restore_model(self,sess,savepath):
        self.saver.restore(sess, tf.train.latest_checkpoint(savepath))
        return
    def save_model(self,sess,savepath):
        self.saver.save(sess, savepath + 'my-model-sentence-sen-1024')

        return
    def assign_emb(self, sess, x):
        sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: x})
        return
    def train_step(self, sess, x, y):
        outputs = sess.run([self.pretrain_updates, self.disc_loss,self.disc_acc, self.pred_train_output],
                           feed_dict={self.enc_inp: x, self.labels: y})
        return outputs
def main():
    # Test pre train
    embedding_matrix, x_train, x_val, y_train, y_val, word_index = readFBTask1.create_con(True, MAX_SEQ_LENGTH)
    disc = DiscSentence(MAX_SEQ_LENGTH, word_index, embedding_matrix)
    disc.pretrain(x_train, x_val, y_train, y_val)

    # Test train

    # Test get rewards


if __name__ == '__main__':
    main()
