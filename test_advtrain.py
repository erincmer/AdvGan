import os
import numpy as np

from tensorflow.contrib.legacy_seq2seq import embedding_rnn_decoder
from tensorflow.contrib.legacy_seq2seq import rnn_decoder
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.contrib.rnn.python.ops import rnn_cell
import tensorflow as tf

from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec

import readFBTask1Seq2Seq
from Generator import Generator
from Disc1 import DiscSentence
from Baseline import Baseline


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
MC_NUM = 1


generator = Generator(len(word_index) + 1, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH,REP_SEQ_LENGTH,START_TOKEN,END_TOKEN,HIST_END_TOKEN)
discriminator = DiscSentence(SEQ_LENGTH, word_index, embedding_matrix)
baseline = Baseline(SEQ_LENGTH,REP_SEQ_LENGTH,BATCH_SIZE, word_index, embedding_matrix)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
generator.assign_emb(sess,embedding_matrix)

idxTrain = np.arange(len(hist_train))
loss_t = 0


def concat_hist_reply( histories, replies, word_index):
    disc_inp = np.full((BATCH_SIZE, MAX_SEQUENCE_LENGTH), word_index['eos'])
    counter = 0
    for h, r in zip(histories, replies):

        i = 0
        while i != word_index['eoh']:
            disc_inp[counter, i] = h[i]
            i = i + 1

        disc_inp[counter, i] = word_index['eoh']

        disc_inp[counter, i + 1:i + 21] = r
        counter = counter + 1

    return disc_inp


for ep in range(100):
    np.random.shuffle(idxTrain)

    for j in range(0, hist_train.shape[0] // BATCH_SIZE):
        print("*********************************")
        X = hist_train[idxTrain[j*BATCH_SIZE:(j+1)*BATCH_SIZE],:]
        #Y = reply_train[idxTrain[j*BATCH_SIZE:(j+1)*BATCH_SIZE],:]
        Y = np.ones((BATCH_SIZE, REP_SEQ_LENGTH)) * word_index['eos']

        _,sentence = generator.generate(sess, X, Y)
        disc_in = concat_hist_reply(X,sentence,word_index)

        disc_rewards = discriminator.get_rewards(disc_in)
        print("///////////////////////////////")
        print("Discriminator Rewards for first 3 Sentence = ", disc_rewards[0:3])
        print("///////////////////////////////")
        print("MC sample ids for first 3 Sentence")  # depend
        print(np.array(sentence)[0:3])

        rewards = generator.MC_reward(sess, X, sentence, MC_NUM, discriminator,word_index)

        print("///////////////////////////////")
        print("MC Rewards for first 3 Sentence")  # depend
        print(np.array(rewards)[0:3])



        b = baseline.get_baseline(X,sentence,word_index)

        print("///////////////////////////////")
        print("Baseline Rewards for first 3 Sentence")  # depend
        print(np.array(b)[0:3])


        _,adv_loss =generator.advtrain_step(sess, X, Y, sentence, rewards, b)

        print("///////////////////////////////")
        print("Adverserial Loss = " , adv_loss)
        print("///////////////////////////////")
        baseline_loss = baseline.train(X,sentence,rewards,word_index)
        print("Baseline Loss = " , baseline_loss)
