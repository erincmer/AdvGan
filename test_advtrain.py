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

np.set_printoptions(precision=5,suppress=True,linewidth=250)
MAX_SEQUENCE_LENGTH = 200
embedding_matrix,hist_train,hist_test,reply_train,reply_test,x_train,x_test,y_train,y_test,word_index = readFBTask1Seq2Seq.create_con(True,MAX_SEQUENCE_LENGTH)

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
MC_NUM = 3


generator = Generator(len(word_index) + 1, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH,REP_SEQ_LENGTH,START_TOKEN,END_TOKEN,HIST_END_TOKEN)
discriminator = DiscSentence(len(word_index) + 1, BATCH_SIZE,  EMB_DIM, HIDDEN_DIM,SEQ_LENGTH,word_index, END_TOKEN)
baseline = Baseline(SEQ_LENGTH,REP_SEQ_LENGTH,BATCH_SIZE, word_index, embedding_matrix)

# print(word_index)
# input("wait")
savepathG = 'GeneratorModel/'  # best is saved here
savepathD = 'DiscModel/'


if not os.path.exists(savepathG):
    os.makedirs(savepathG)
if not os.path.exists(savepathD):
    os.makedirs(savepathD)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
generator.assign_emb(sess,embedding_matrix)

discriminator.assign_emb(sess,embedding_matrix)






#
try:
    discriminator.restore_model(sess,savepathD)
except:
    print("Disc could not be restored")
    pass
try:

    generator.restore_model(sess, savepathG)
except:
    print("Gen could not be restored")
    pass



idxTrain = np.arange(len(x_train))
idxTest = np.arange(len(x_test))

np.random.shuffle(idxTest)
X = x_test[idxTest[:BATCH_SIZE], :]
Y_train = y_test[idxTest[:BATCH_SIZE]]
d_loss, d_acc = discriminator.get_loss(sess, X, Y_train)
print("Disc Test Loss = ", d_loss, " Accuracy = ", d_acc)
for ep in range(0):
    np.random.shuffle(idxTrain)

    for j in range(0, x_train.shape[0] // BATCH_SIZE):
        # print("*********************************")
        X = x_train[idxTrain[j*BATCH_SIZE:(j+1)*BATCH_SIZE],:]
        Y_train = y_train[idxTrain[j*BATCH_SIZE:(j+1)*BATCH_SIZE]]



        _,d_loss,d_acc,_ = discriminator.train_step(sess,X,Y_train)

        if j %50==0:
                print("Disc Train Loss = ", d_loss, " Accuracy = ",d_acc)
                np.random.shuffle(idxTest)
                X = x_test[idxTest[:BATCH_SIZE],:]
                Y_train = y_test[idxTest[:BATCH_SIZE]]
                d_loss,d_acc= discriminator.get_loss(sess, X, Y_train)
                print("Disc Test Loss = ", d_loss, " Accuracy = ",d_acc)
                discriminator.save_model(sess, savepathD)





idxTrain = np.arange(len(hist_train))
idxTest = np.arange(len(hist_test))
for ep in range(100):
    # np.random.shuffle(idxTrain)

    for j in range(0, hist_train.shape[0] // BATCH_SIZE):
        # print("*********************************")
        j = 0 # TODO: remove it when training Generator

        X = hist_train[idxTrain[j*BATCH_SIZE:(j+1)*BATCH_SIZE],:]
        Y_train = reply_train[idxTrain[j*BATCH_SIZE:(j+1)*BATCH_SIZE],:]

        if ep<0:

            _,g_loss,_ = generator.pretrain_step(sess,X,Y_train)
            if j %1000 ==0:

                print("Gen Train Loss = ", g_loss, ep)
                X = hist_test[idxTest[: BATCH_SIZE], :]
                Y_test = reply_test[idxTest[: BATCH_SIZE], :]
                g_loss = generator.get_pretrain_loss(sess, X, Y_test)
                print("Gen Test Loss = ", g_loss,ep)
                generator.save_model(sess, savepathG)

                _, sentence = generator.generate(sess, X, Y_test)
                sentence[sentence == 0] = word_index['eos']
                print("Generator Predicted Sentences")
                convert_id_to_text(np.array(sentence)[0:5], word_index)
                print("Real Sentences")
                convert_id_to_text(np.array(Y_test)[0:5], word_index)

        else:
            print("*********************************")
            Y = np.ones((BATCH_SIZE, REP_SEQ_LENGTH)) * word_index['eos']
            X_one = np.tile(np.copy(X[0, :]), (BATCH_SIZE, 1))
            convert_id_to_text(np.array(X_one)[0:3, :], word_index)


            gen_proba,sentence = generator.generate(sess, X_one, Y)
            print("FULL SAMPLED SENTENCES")
            convert_id_to_text(np.array(sentence)[:,:], word_index)
            sentence_old = np.copy(np.array(sentence))
            # convert_id_to_text(np.array(Y_train)[0:1, :], word_index)

            gen_proba = np.array(gen_proba)

            gen_proba = gen_proba[31, :, :]
            print("SENTENCES WITH CHOSING WORD PROBABILITIES")
            print(sentence[31, :])
            print(gen_proba[np.arange(len(gen_proba)), sentence[31, :]])
            # input("wait 1")
            # print(gen_proba[0,1, sentence[0,1]])


            # X_one =np.tile(np.copy(X[0,:]),(BATCH_SIZE,1))


            # Sen_one = np.tile(np.copy(sentence[0, :]), (BATCH_SIZE, 1))
            #
            # sentence = Sen_one
            #
            #
            # rep_inp = np.full((BATCH_SIZE, REP_SEQ_LENGTH), word_index['eos'])
            # rep_inp[:, :sentence.shape[1]] = sentence
            # sentence = rep_inp
            # sentence[sentence ==0] = word_index['eos']

            disc_in = concat_hist_reply(X_one,sentence,word_index)
            # print(X_one[0:3,])
            # input("wait 2 ")
            disc_rewards = discriminator.get_rewards(sess,disc_in)

            sen_rand = np.random.random_integers(len(word_index), size=(BATCH_SIZE, REP_SEQ_LENGTH))
            stop_index = np.random.random_integers(REP_SEQ_LENGTH, size=(BATCH_SIZE, 1))
            for i in range(BATCH_SIZE):
                sen_rand[i, stop_index[i][0]:] = word_index["eos"]

            disc_in_rand = concat_hist_reply(X,sen_rand,word_index)
            disc_in_real = concat_hist_reply(X, Y_train, word_index)
            disc_rewards_rand = discriminator.get_rewards(sess,disc_in_rand)
            # disc_rewards_real = discriminator.get_rewards(disc_in_real)
            disc_rewards_real = discriminator.get_rewards(sess,disc_in_real)
            print("///////////////////////////////")
            print("Discriminator Rewards for MC Sentences = ", disc_rewards[30:33,1])
            print("Discriminator Rewards for Random Sentences = ", disc_rewards_rand[0:3,1])
            print("Discriminator Rewards for Real Sentences = ", disc_rewards_real[0:3,1])
            print("///////////////////////////////")
            print("MC sample ids for first 3 Sentence")  # depend
            print(np.array(sentence)[30:33])
            convert_id_to_text(np.array(sentence)[30:33],word_index)
            print("CORRECT SENTENCE")
            convert_id_to_text(Y_train[0:1,:], word_index)

            rewards = generator.MC_reward(sess, X_one, sentence, MC_NUM, discriminator,word_index)

            print("///////////////////////////////")
            print("MC Rewards for first 3 Sentence")  # depend
            print(np.array(rewards)[30:33])
            # input("wait 3 ")

            baseline_loss = baseline.train(X_one, sentence, rewards, word_index)

            b = baseline.get_baseline(X_one,sentence,word_index)

            print("///////////////////////////////")
            print("Baseline Rewards for first 3 Sentence")  # depend
            print(np.array(b)[30:33])

            print("Baseline Loss = " , baseline_loss)
            part0,part1,part2,part3,part4,part5 = generator.get_adv_loss(sess, X_one, Y, sentence, rewards, b)
            #
            #
            # print("one hot encoding")
            # print(np.array(part0).shape)
            # print(np.array(part0)[20:23, :])
            # print("logarithm of action probs")
            # print(np.array(part1).shape)
            # print(np.array(part1)[0:3, :])
            # print(np.argmax(np.array(part1)[0:3, :],1))
            # print(np.array(part1)[20:23, :])
            # print(np.argmax(np.array(part1)[20:23, :],1))
            print("log action multiplied by one hot encoding ")
            print(np.array(part2).shape)
            print(np.array(part2)[626:628,:]) # since word 627 is wrong
            print("reduce sum  ")
            print(np.array(part3).shape)
            print(np.array(part3)[626:628])
            # # input("wait")
            #
            # # c
            # # print(part5)
            # #
            #
            #
            #
            # print(np.array(part4)[20:23])
            _,adv_loss =generator.advtrain_step(sess, X_one, Y, sentence, rewards, b)
            # input("wait")
            print("///////////////////////////////")
            # print("adv loss = " , adv_loss[620:640])
            # print(np.argmax(adv_loss))
            # print("Adverserial Loss = " , adv_loss[:20])
            # print("Adverserial Loss = ", adv_loss[20:40])
            # print("Adverserial Loss = ", adv_loss[40:60])
            # print("Adverserial Loss = ", adv_loss[60:80])

            print("///////////////////////////////AFTER UPDATE/////////////////////////////")
            convert_id_to_text(np.array(X_one)[30:33, :], word_index)
            gen_proba,sentence = generator.generate(sess, X_one, Y)
            convert_id_to_text(np.array(sentence)[30:33,:], word_index)
            # convert_id_to_text(np.array(Y_train)[0:1, :], word_index)

            gen_proba = np.array(gen_proba)

            gen_proba = gen_proba[31, :, :]
            print(sentence[31, :])
            print(gen_proba[np.arange(len(gen_proba)), sentence_old[31, :]])

            print("///////////////////////////////AFTER UPDATE/////////////////////////////")
            input("wait")

            # input("wait")