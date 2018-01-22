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
from keras.models import load_model
import readFBTask1Seq2Seq
from Generator import Generator
from Disc1 import DiscSentence
from Baseline import Baseline


def convert_id_to_text(ids,word_index):

    for id in ids:
        sen = ""
        for i in id:
            if i!=0  and i!= word_index["eos"]:
                sen = sen +" " +list(word_index.keys())[list(word_index.values()).index(i)]
        print(sen)

def concat_hist_reply( histories, replies, word_index):
    disc_inp = np.full((BATCH_SIZE, MAX_SEQUENCE_LENGTH), word_index['eos'])

    counter = 0
    for h, r in zip(histories, replies):

        i = 0
        while h[i] != word_index['eoh']:
            disc_inp[counter, i] = h[i]
            i = i + 1

        disc_inp[counter, i] = word_index['eoh']

        disc_inp[counter, i + 1:i + 21] = r
        counter = counter + 1

    return disc_inp


MAX_SEQUENCE_LENGTH = 200
embedding_matrix,hist_train,hist_val,reply_train,reply_val,x_train,x_test,y_train,y_test,word_index = readFBTask1Seq2Seq.create_con(True,MAX_SEQUENCE_LENGTH)

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

savepath = 'GeneratorModel/'  # best is saved here

saver_all = tf.train.Saver()

if not os.path.exists(savepath):
    os.makedirs(savepath)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
generator.assign_emb(sess,embedding_matrix)



idxTrain = np.arange(len(hist_train))
loss_t = 0



generator.restore_model(sess,savepath)
# discriminator.pretrain(x_train,x_test,y_train,y_test)
# print("before disc restored")
y_train = to_categorical(np.asarray(y_train))
y_test = to_categorical(np.asarray(y_test))
# print(discriminator.model.evaluate(x_test,y_test,verbose = 0))
discriminator.model = load_model("discriminator.h5")



for ep in range(100):
    np.random.shuffle(idxTrain)

    for j in range(0, hist_train.shape[0] // BATCH_SIZE):
        # print("*********************************")
        X = hist_train[idxTrain[j*BATCH_SIZE:(j+1)*BATCH_SIZE],:]
        Y_train = reply_train[idxTrain[j*BATCH_SIZE:(j+1)*BATCH_SIZE],:]

        if ep<2:

            _,g_loss,_ = generator.pretrain_step(sess,X,Y_train)
            if j %100 ==0:
                generator.save_model(sess, savepath)
                print(g_loss)
        else:
            Y = np.ones((BATCH_SIZE, REP_SEQ_LENGTH)) * word_index['eos']

            _,sentence = generator.generate(sess, X, Y)

            rep_inp = np.full((BATCH_SIZE, REP_SEQ_LENGTH), word_index['eos'])
            rep_inp[:, :sentence.shape[1]] = sentence
            sentence = rep_inp
            sentence[sentence ==0] = word_index['eos']

            disc_in = concat_hist_reply(X,sentence,word_index)

            disc_rewards = discriminator.get_rewards(disc_in)

            sen_rand = np.random.random_integers(len(word_index), size=(BATCH_SIZE, REP_SEQ_LENGTH))
            stop_index = np.random.random_integers(REP_SEQ_LENGTH, size=(BATCH_SIZE, 1))
            for i in range(BATCH_SIZE):
                sen_rand[i, stop_index[i][0]:] = word_index["eos"]

            disc_in_rand = concat_hist_reply(X,sen_rand,word_index)
            disc_in_real = concat_hist_reply(X, Y_train, word_index)
            disc_rewards_rand = discriminator.get_rewards(disc_in_rand)
            disc_rewards_real = discriminator.get_rewards(disc_in_real)
            print("///////////////////////////////")
            print("Discriminator Rewards for MC Sentences = ", disc_rewards[0:3,1])
            print("Discriminator Rewards for Random Sentences = ", disc_rewards_rand[0:3,1])
            print("Discriminator Rewards for Real Sentences = ", disc_rewards_real[0:3,1])
            print("///////////////////////////////")
            print("MC sample ids for first 3 Sentence")  # depend
            print(np.array(sentence)[0:3])
            convert_id_to_text(np.array(sentence)[0:3],word_index)
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
