import os
import numpy as np

from tensorflow.contrib.legacy_seq2seq import embedding_rnn_decoder
from tensorflow.contrib.legacy_seq2seq import rnn_decoder
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.contrib.rnn.python.ops import rnn_cell
import tensorflow as tf

#from keras.utils.np_utils import to_categorical
#from keras.layers import Embedding
#from keras.layers import Dense, Input, Flatten
#from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
#from keras.models import Model
#from keras import backend as K
#from keras.engine.topology import Layer, InputSpec
#from keras.models import load_model
import readFBTask1Seq2Seq
from Generator import Generator
from Disc1 import DiscSentence
from Baseline import Baseline
import header
import tools
import pretrain

np.set_printoptions(precision=5,suppress=True,linewidth=250)

# Pre train if necessary
savepathG = 'GeneratorModel/'  # best is saved here
savepathD = 'DiscModel/'

#if ~header.DO_RESTORE: 
#    pretrain.pretrain(savepathD, savepathG)

# Adversarial training

# Load data
(embedding_matrix,
        hist_train,
        hist_test,
        reply_train,
        reply_test,
        x_train,
        x_test,
        y_train,
        y_test,
        word_index) = readFBTask1Seq2Seq.create_con(False,header.MAX_SEQ_LENGTH)

EMB_DIM = len(word_index) + 1 # embedding dimension
END_TOKEN = word_index.get("eos")
HIST_END_TOKEN = word_index.get("eoh")

# Model
generator = Generator(EMB_DIM,
        header.BATCH_SIZE, 
        EMB_DIM, 
        header.HIDDEN_DIM, 
        header.MAX_SEQ_LENGTH,
        header.REP_SEQ_LENGTH,
        header.START_TOKEN,
        END_TOKEN,
        HIST_END_TOKEN)
discriminator = DiscSentence(EMB_DIM, 
        header.BATCH_SIZE,  
        EMB_DIM, header.HIDDEN_DIM,
        header.MAX_SEQ_LENGTH,
        word_index, 
        END_TOKEN)
baseline = Baseline(header.BATCH_SIZE, 
        header.HIDDEN_DIM,
        header.MAX_SEQ_LENGTH, 
        word_index, 
        learning_rate=0.0004)

# TF setting
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
generator.assign_emb(sess,embedding_matrix)
discriminator.assign_emb(sess,embedding_matrix)

# Restore pre trained models
try:
    discriminator.restore_model(sess,savepathD)
except:
    print("Disc could not be restored")
    pass
try:
     print("Using New Generator")
     generator.restore_model(sess, savepathG)
except:
    print("Gen could not be restored")
    pass

d_steps = 0
g_steps = 1
# Adversarial steps
for ep in range(2,10000):
    # Train discriminator
    for d in range(d_steps):
        np.random.shuffle(idxTrain)
        for j in range(0, hist_train.shape[0] // header.BATCH_SIZE):
            X = hist_train[idxTrain[j*header.BATCH_SIZE:(j+1)*header.BATCH_SIZE],:]
            Y_train = reply_train[idxTrain[j*header.BATCH_SIZE:(j+1)*header.BATCH_SIZE],:]

            # Generate sentence 
            if not teacher_forcing:
                Y = np.ones((header.BATCH_SIZE, header.REP_SEQ_LENGTH)) * word_index['eos']
                teacher_forcing = True
            else:
                # Teacher forcing
                Y = Y_train
                teacher_forcing = False
            _,sentence = generator.generate(sess, X, Y)

            # Pad sentence of variable length to header.REP_SEQ_LENGTH
            rep_inp = np.full((header.BATCH_SIZE, header.REP_SEQ_LENGTH), word_index['eos'])
            rep_inp[:, :sentence.shape[1]] = sentence
            sentence = rep_inp
            sentence[sentence ==0] = word_index['eos']

            # Build a batch of half true and half false sentences
            Y_d = np.zeros((header.BATCH_SIZE, header.REP_SEQ_LENGTH))
            sentence_index = np.random.random_integers((header.BATCH_SIZE-1), size=(int(header.BATCH_SIZE/2),1))

            Y_d[0:int(header.BATCH_SIZE/2),:] = np.squeeze(Y_train[sentence_index,:])
            Y_d[int(header.BATCH_SIZE/2):,:] = np.squeeze(sentence[sentence_index,:])
            X_d = concat_hist_reply(X, Y_d, word_index)
            label_d = np.zeros(header.BATCH_SIZE)
            label_d[int(header.BATCH_SIZE/2):] = 1
             
            _,d_loss,d_acc,_ = discriminator.train_step(sess,X_d, label_d)
            print("Discriminator loss = ", d_loss)

    # Train generator
    max_avg_prob = np.zeros((10,),dtype=np.float)
    for g in range(g_steps):
        ind = ep % 3
        X = hist_train[ind : header.BATCH_SIZE + ind, :]
        Y_train = reply_train[ind : header.BATCH_SIZE + ind, :]
        print("ind ==== ", ind)

            # print("*********************************")
        Y = np.ones((header.BATCH_SIZE, header.REP_SEQ_LENGTH)) * word_index['eos']
        X_one = np.tile(np.copy(X[0, :]), (header.BATCH_SIZE, 1))
        Y_one = np.tile(np.copy(Y_train[0, :]), (header.BATCH_SIZE, 1))
        tools.convert_id_to_text(np.array(X_one)[0:3, :], word_index)

        gen_proba,sentence,_= generator.generate(sess, X_one, Y)
        gen_proba_test, sentence_test = generator.test_generate(sess, X_one, Y)
        gen_proba_test_one =np.copy(gen_proba_test[0,:,:])
        print("Target words probabilities")
        print(gen_proba_test_one[np.arange(len(gen_proba_test_one)), Y_one[0, :]])

        print("Created Sentence with argmax")
        tools.convert_id_to_text(np.array(sentence_test)[0:1, :], word_index)
        print("Target Sentences")
        tools.convert_id_to_text(np.array(Y_one)[0:1, :], word_index)
        # print(Y_train[0, :])
        print("Number of Correct Words  =============================" , np.sum(sentence_test==Y_one))

        temp_av = gen_proba_test_one[np.arange(len(gen_proba_test_one)), Y_one[0, :]]



        temp_av[Y_train[0, :]==word_index["eos"]] = 0
        av_proba = np.sum(temp_av)/ np.sum(Y_one[0, :] != word_index["eos"])
        print("Average Prob for Chosing Right Action  =============================", av_proba)

        if av_proba>max_avg_prob[ind]:
            print("**********************************************MAX FOUND**************************************************************************")
            max_avg_prob[ind] = av_proba
        print(max_avg_prob)
        # print(np.array(gen_proba).shape)

        # print("FULL SAMPLED SENTENCES")
        # tools.convert_id_to_text(np.array(sentence)[:5,:], word_index)
        sentence_old = np.copy(np.array(sentence))
        # tools.convert_id_to_text(np.array(Y_train)[0:1, :], word_index)

        gen_proba = np.array(gen_proba)

        gen_proba = gen_proba[0, :, :]
        # print("SENTENCES WITH CHOSING WORD PROBABILITIES")
        # tools.convert_id_to_text(np.array(sentence_test)[0:1, :], word_index)
        # print(gen_proba[np.arange(len(gen_proba)), sentence[0, :]])
        # input("wait 1")
        # print(gen_proba[0,1, sentence[0,1]])


        # X_one =np.tile(np.copy(X[0,:]),(header.BATCH_SIZE,1))


        # Sen_one = np.tile(np.copy(sentence[0, :]), (header.BATCH_SIZE, 1))
        #
        # sentence = Sen_one
        #
        #
        # rep_inp = np.full((header.BATCH_SIZE, header.REP_SEQ_LENGTH), word_index['eos'])
        # rep_inp[:, :sentence.shape[1]] = sentence
        # sentence = rep_inp
        # sentence[sentence ==0] = word_index['eos']

        disc_in = tools.concat_hist_reply(X_one,sentence,word_index)
        # print(X_one[0:3,])
        # input("wait 2 ")
        disc_rewards = discriminator.get_rewards(sess,disc_in)

        sen_rand = np.random.random_integers(len(word_index), size=(header.BATCH_SIZE, header.REP_SEQ_LENGTH))
        stop_index = np.random.random_integers(header.REP_SEQ_LENGTH, size=(header.BATCH_SIZE, 1))
        for i in range(header.BATCH_SIZE):
            sen_rand[i, stop_index[i][0]:] = word_index["eos"]

        disc_in_rand = tools.concat_hist_reply(X,sen_rand,word_index)
        disc_in_real = tools.concat_hist_reply(X, Y_train, word_index)
        disc_rewards_rand = discriminator.get_rewards(sess,disc_in_rand)
        # disc_rewards_real = discriminator.get_rewards(disc_in_real)
        disc_rewards_real = discriminator.get_rewards(sess,disc_in_real)
        print("///////////////////////////////")
        print("Discriminator Rewards for MC Sentences = ", disc_rewards[0:3,1])
        print("Discriminator Rewards for Random Sentences = ", disc_rewards_rand[0:3,1])
        print("Discriminator Rewards for Real Sentences = ", disc_rewards_real[0:3,1])
        print("///////////////////////////////")
        print("MC sample ids for first 3 Sentence")  # depend
        print(np.array(sentence)[0:3])
        tools.convert_id_to_text(np.array(sentence)[0:3],word_index)
        # print("CORRECT SENTENCE")


        rewards = generator.MC_reward(sess, X_one, sentence, header.MC_NUM, discriminator,word_index)

        print("///////////////////////////////")
        print("MC Rewards for first 3 Sentence")  # depend
        print(np.array(rewards)[0:3])
        # input("wait 3 ")

        #baseline_loss = baseline.train(X_one, sentence, rewards, word_index)
        b = np.tile(np.mean(np.array(rewards),axis = 0 ), (header.BATCH_SIZE, 1))
        #b = baseline.get_baseline(X_one,sentence,word_index)

        print("///////////////////////////////")
        print("Baseline Rewards for first 3 Sentence")  # depend
        print(np.array(b)[30:33])

        # print("Baseline Loss = " , baseline_loss)
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
        # print("log action multiplied by one hot encoding ")
        # print(np.array(part2).shape)
        # print(np.array(part2)[626:628,:]) # since word 627 is wrong
        # print("reduce sum  ")
        # print(np.array(part3).shape)
        # print(np.array(part3)[626:628])
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
        # print("///////////////////////////////")
        # print("adv loss = " , adv_loss[620:640])
        # print(np.argmax(adv_loss))
        # print(adv_loss)
        # print("Adverserial Loss = " , adv_loss[:20])
        # print("Adverserial Loss = ", adv_loss[20:40])
        # print("Adverserial Loss = ", adv_loss[40:60])
        # print("Adverserial Loss = ", adv_loss[60:80])

        # print("///////////////////////////////AFTER UPDATE/////////////////////////////")
        # tools.convert_id_to_text(np.array(X_one)[:3, :], word_index)
        # gen_proba,sentence = generator.generate(sess, X_one, Y)
        # tools.convert_id_to_text(np.array(sentence)[0:3,:], word_index)
        # tools.convert_id_to_text(np.array(Y_train)[0:1, :], word_index)

        # gen_proba = np.array(gen_proba)

        # gen_proba = gen_proba[0, :, :]
        # print(sentence[0, :])
        # print(gen_proba[np.arange(len(gen_proba)), sentence_old[0, :]])

        # print("///////////////////////////////AFTER UPDATE/////////////////////////////")
        # input("wait")

        # input("wait")
