import numpy as np
from tensorflow.contrib.legacy_seq2seq import embedding_rnn_decoder
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq

from tensorflow.contrib.rnn.python.ops import rnn_cell

import readFBTask1Seq2Seq


import os
import tensorflow as tf



from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec


MAX_SEQUENCE_LENGTH = 200


sess = tf.InteractiveSession()

embedding_matrix,hist_train,hist_val,reply_train,reply_val,word_index = readFBTask1Seq2Seq.create_con(True,MAX_SEQUENCE_LENGTH)



print('Traing and validation set number of positive and negative reviews')

seq_length = 200
rep_seq_length = 20
batch_size = 64

vocab_size = len(word_index) + 1
embedding_dim = len(word_index) + 1

memory_dim = 400

enc_inp = [tf.placeholder(tf.int32, shape=(None,),
                          name="inp%i" % t)
           for t in range(seq_length)]

labels = [tf.placeholder(tf.int32, shape=(None,),
                        name="labels%i" % t)
          for t in range(rep_seq_length)]

weights = [tf.ones_like(labels_t, dtype=tf.float32)
           for labels_t in labels]


# Decoder input: prepend some "GO" token and drop the final
# token of the encoder input
dec_inp = ([tf.zeros_like(labels[0], dtype=np.int32, name="GO")]
           + labels[:-1])

# Initial memory value for recurrence.
prev_mem = tf.zeros((batch_size, memory_dim))
with tf.variable_scope("encoder", reuse=None) as scope:

        cell = tf.contrib.rnn.GRUCell(memory_dim)

        enc_cell = tf.contrib.rnn.EmbeddingWrapper(
            cell, embedding_classes=vocab_size,
            embedding_size=embedding_dim)

        # #
        # basic_cell = tf.contrib.rnn.DropoutWrapper(
        #     encoder_cell,
        #     output_keep_prob=self.keep_prob)

        _, encoder_state = tf.contrib.rnn.static_rnn(enc_cell, enc_inp,prev_mem)
        # encoder_outputs, encoder_state = tf.nn.dynamic_rnn(enc_cell,enc_inp, dtype=tf.float32)


with tf.variable_scope("decoder", reuse=None) as scope:

        cell = tf.contrib.rnn.GRUCell(memory_dim)
        dec_cell = tf.contrib.rnn.OutputProjectionWrapper(cell, vocab_size)


        dec_outputs, dec_memory = embedding_rnn_decoder(dec_inp,
                                                        encoder_state,
                                                        dec_cell,
                                                        vocab_size,
                                                        embedding_dim,
                                                        update_embedding_for_previous=False,
                                                        output_projection=None,
                                                        feed_previous=False)

learning_rate = 0.0005

optimizer = tf.train.AdamOptimizer(learning_rate)
loss = tf.contrib.legacy_seq2seq.sequence_loss(dec_outputs, labels, weights,vocab_size)
train_op = optimizer.minimize(loss)
train_list = []

print("trainable variables//////////////////////////")
for v in tf.trainable_variables():
    if "embeddings:0" not in v.name and "embedding:0" not in v.name:
        train_list.append(v)
        print(v.name,v.get_shape().as_list())


train_op = optimizer.minimize(loss,var_list=train_list, aggregation_method=2)

sess.run(tf.global_variables_initializer())

var_2 = [v for v in tf.global_variables() if v.name == "decoder/embedding_rnn_decoder/embedding:0"][0]
var_3 = [v for v in tf.global_variables() if v.name == "encoder/rnn/embedding_wrapper/embedding:0"][0]

op2 =  var_2.assign(embedding_matrix)
op3 =  var_3.assign(embedding_matrix)

sess.run(op2)
sess.run(op3)

e2 = var_2.eval(session=sess)
e3 = var_3.eval(session=sess)

print(" after emb 2 = " , e2[0:3,0:3])
print(" after emb 3 = " , e3[0:3,0:3])

e2 = var_2.eval(session=sess)
e3 = var_3.eval(session=sess)

idxTrain = np.arange(len(hist_train))
for ep in range(10000):
    np.random.shuffle(idxTrain)

    for j in range(0, hist_train.shape[0] // batch_size):
        X = hist_train[idxTrain[j*batch_size:(j+1)*batch_size],:]
        Y = reply_train[idxTrain[j*batch_size:(j+1)*batch_size],:]

        # Dimshuffle to seq_len * batch_size
        X = np.array(X).T
        Y = np.array(Y).T

        feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
        feed_dict.update({labels[t]: Y[t] for t in range(rep_seq_length)})

        _, loss_t = sess.run([train_op, loss], feed_dict)

        if j %100 ==0:


            X = hist_val[j * batch_size:(j + 1) * batch_size, :]
            s = np.random.randint(0, X.shape[0])
            Y = reply_val[j * batch_size:(j + 1) * batch_size, :]
            X = np.array(X).T
            Y = np.array(Y).T

            feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
            feed_dict.update({labels[t]: Y[t] for t in range(rep_seq_length)})
            dec_outputs_batch = sess.run(dec_outputs, feed_dict)
            Y = Y.T
            # print(Y.shape)

            print("////////////////////////////////////////////")
            print("real output")
            output = ""
            for x in Y[s]:
                if x!=0:
                    output = output + " "+str(x)
            print(output,loss_t)
            output = ""
            print("predicted")
            dec_out = [logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]
            dec_out = np.array(dec_out).T
            # print(dec_out.shape)
            for x in dec_out[s]:
                if x!=0:
                    output = output + " "+str(x)
            print(output,loss_t)
            print("////////////////////////////////////////////")