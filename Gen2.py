import numpy as np
from tensorflow.contrib.legacy_seq2seq import embedding_rnn_decoder
from tensorflow.contrib.legacy_seq2seq import rnn_decoder
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

from Generator import Generator

MAX_SEQUENCE_LENGTH = 200
embedding_matrix,hist_train,hist_val,reply_train,reply_val,reply_in_train,reply_in_val,word_index = readFBTask1Seq2Seq.create_con(True,MAX_SEQUENCE_LENGTH)

EMB_DIM = len(word_index) + 1 # embedding dimension
HIDDEN_DIM = 250 # hidden state dimension of lstm cell
SEQ_LENGTH = 200 # sequence length
REP_SEQ_LENGTH = 20
START_TOKEN = 0
END_TOKEN = word_index.get("eos")
HIST_END_TOKEN = word_index.get("eoh")
PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64





generator = Generator(len(word_index) + 1, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH,REP_SEQ_LENGTH,START_TOKEN,END_TOKEN,HIST_END_TOKEN
                      )
print(tf.VERSION)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

savepathTask = 'Models/AdvGanTask1/'  # best is saved here
saver_all = tf.train.Saver()

if not os.path.exists(savepathTask):
    os.makedirs(savepathTask)

# saver_all.restore(sess, tf.train.latest_checkpoint(savepathTask))
generator.assign_emb(sess,embedding_matrix)



idxTrain = np.arange(len(hist_train))
loss_t = 0
for ep in range(10000000):
    np.random.shuffle(idxTrain)

    for j in range(0, hist_train.shape[0] // BATCH_SIZE):


            X = hist_train[idxTrain[j*BATCH_SIZE:(j+1)*BATCH_SIZE],:]
            Y = reply_train[idxTrain[j*BATCH_SIZE:(j+1)*BATCH_SIZE],:]
            Y_in = np.array(list(reply_train[idxTrain[j * BATCH_SIZE:(j + 1) * BATCH_SIZE], :]))
            Y_in[:, 2:] = 0 # here we put 0 to everzthing other than first input

            _, loss,out = generator.pretrain_step(sess, X, Y)
            if j %20 == 0 and j >0:
                out, out_ids = generator.generate_train(sess, X, Y)
                output, output_ids = generator.generate(sess, X, Y_in)
                train_out_id = np.array(out)[0]
                mc_out_id = np.array(output)[0]
                print("output with argmax") # output by taking arg max of logit lazer of training network and copied generator
                # outputs will match with anz temperature because for every given history there is only one correct reply so after it is trained it does not
                # even require to decode correct sentence since it can easily fidn correct reply given history
                print(np.argmax(train_out_id,axis = 1))
                print(np.argmax(mc_out_id, axis=1))
                print("training sample ids") # training sample ids are always equal to input
                print(np.array(out_ids)[0])
                print("mc sample ids") # depending on the softmax temperature in generator class it can either gives same as ground truth or sample totally random
                #we alwazsa sample next input but if time is lower than sequence length in our case time goes till first yero found in input it uses ground truth otherwise it uses sampled version
                print(np.array(output_ids)[0])

                print("labels") # ground truth of inputs
                print(Y[0])
                print(loss)
                print("///////////////////////////////")
