from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf


import readFBTask1Seq2Seq
from GeneratorModel import Generator
from Disc1 import DiscSentence
from Baseline import Baseline
import headerSeq2Seq
import toolsSeq2Seq
import pretrain







np.set_printoptions(precision=5, suppress=True, linewidth=250)



def create_folders():
    savepathG_S1_Seq2Seq = 'GeneratorModel/S1/Seq2Seq/'
    savepathG_S1_ReInf = 'GeneratorModel/S1/Reinforce/'
    savepathG_S1_PPO = 'GeneratorModel/S1/PPO/'
    savepathD_S1 = 'DiscModel/S1/'
    savepathB_S1 = 'BaselineModel/S1/'

    savepathG_S2_Seq2Seq = 'GeneratorModel/S2/Seq2Seq/'
    savepathG_S2_ReInf = 'GeneratorModel/S2/Reinforce/'
    savepathG_S2_PPO = 'GeneratorModel/S2/PPO/'
    savepathD_S2 = 'DiscModel/S2/'
    savepathB_S2 = 'BaselineModel/S2/'


    savepathD_S3 = 'DiscModel/S3/'
    savepathB_S3 = 'BaselineModel/S3/'



    if not os.path.exists(savepathG_S1_Seq2Seq):
        os.makedirs(savepathG_S1_Seq2Seq)
    if not os.path.exists(savepathG_S1_ReInf):
        os.makedirs(savepathG_S1_ReInf)
    if not os.path.exists(savepathG_S1_PPO):
        os.makedirs(savepathG_S1_PPO)
    if not os.path.exists(savepathD_S1):
        os.makedirs(savepathD_S1)
    if not os.path.exists(savepathB_S1):
        os.makedirs(savepathB_S1)

    if not os.path.exists(savepathG_S2_Seq2Seq):
        os.makedirs(savepathG_S2_Seq2Seq)
    if not os.path.exists(savepathG_S2_ReInf):
        os.makedirs(savepathG_S2_ReInf)
    if not os.path.exists(savepathG_S2_PPO):
        os.makedirs(savepathG_S2_PPO)
    if not os.path.exists(savepathD_S2):
        os.makedirs(savepathD_S2)
    if not os.path.exists(savepathB_S2):
        os.makedirs(savepathB_S2)

    if not os.path.exists(savepathD_S3):
        os.makedirs(savepathD_S3)
    if not os.path.exists(savepathB_S3):
        os.makedirs(savepathB_S3)
    return

def create_networks(EMB_DIM,END_TOKEN,word_index):
    generator_s1 = Generator(EMB_DIM,
                             headerSeq2Seq.BATCH_SIZE,
                             EMB_DIM,
                             headerSeq2Seq.HIDDEN_DIM,
                             headerSeq2Seq.MAX_SEQ_LENGTH,
                             headerSeq2Seq.REP_SEQ_LENGTH,
                             headerSeq2Seq.START_TOKEN,
                             END_TOKEN,"gen1")
    discriminator_s1 = DiscSentence(EMB_DIM,
                                    headerSeq2Seq.BATCH_SIZE,
                                    EMB_DIM, headerSeq2Seq.HIDDEN_DIM,
                                    headerSeq2Seq.MAX_SEQ_LENGTH,
                                    word_index,
                                    END_TOKEN,"disc1")
    baseline_s1 = Baseline(headerSeq2Seq.BATCH_SIZE,
                           headerSeq2Seq.HIDDEN_DIM,
                           headerSeq2Seq.REP_SEQ_LENGTH,
                           headerSeq2Seq.MAX_SEQ_LENGTH,
                           word_index,
                           "base1",
                           learning_rate=0.0004)

    generator_s2 = Generator(EMB_DIM,
                             headerSeq2Seq.BATCH_SIZE,
                             EMB_DIM,
                             headerSeq2Seq.HIDDEN_DIM,
                             headerSeq2Seq.MAX_SEQ_LENGTH,
                             headerSeq2Seq.REP_SEQ_LENGTH,
                             headerSeq2Seq.START_TOKEN,
                             END_TOKEN,"gen2")
    discriminator_s2 = DiscSentence(EMB_DIM,
                                    headerSeq2Seq.BATCH_SIZE,
                                    EMB_DIM, headerSeq2Seq.HIDDEN_DIM,
                                    headerSeq2Seq.MAX_SEQ_LENGTH,
                                    word_index,
                                    END_TOKEN,"disc2")
    baseline_s2 = Baseline(headerSeq2Seq.BATCH_SIZE,
                           headerSeq2Seq.HIDDEN_DIM,
                           headerSeq2Seq.REP_SEQ_LENGTH,
                           headerSeq2Seq.MAX_SEQ_LENGTH,
                           word_index,"base2",
                           learning_rate=0.0004)
    discriminator_s3 = DiscSentence(EMB_DIM,
                                    headerSeq2Seq.BATCH_SIZE,
                                    EMB_DIM, headerSeq2Seq.HIDDEN_DIM,
                                    headerSeq2Seq.MAX_SEQ_LENGTH,
                                    word_index,
                                    END_TOKEN,"disc3")
    baseline_s3 = Baseline(headerSeq2Seq.BATCH_SIZE,
                           headerSeq2Seq.HIDDEN_DIM,
                           headerSeq2Seq.REP_SEQ_LENGTH,
                           headerSeq2Seq.MAX_SEQ_LENGTH,
                           word_index,"base3",
                           learning_rate=0.0004)

    return generator_s1,discriminator_s1,baseline_s1,\
           generator_s2,discriminator_s2,baseline_s2,discriminator_s3,baseline_s3


def preTrainS1(gen1,hist_s1,reply_s1,genEp,saveS1):

    hist_train = hist_s1
    reply_train = reply_s1

    idxTrain = np.arange(len(hist_train))


    g_loss = 0
    for ep in range(genEp):
        np.random.shuffle(idxTrain)
        for j in range(hist_train.shape[0] // headerSeq2Seq.BATCH_SIZE):
            X = hist_train[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE], :]
            Y_train = reply_train[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE], :]

            _, g_loss, _ = gen1.pretrain_step(sess, X, Y_train)

            if j == 0:

                print("Gen Train Loss = ", g_loss, ep)

                gen1.save_model(sess, savepathG_S1_Seq2Seq)


def preTrainS2(gen2, hist_s2, reply_s2 ,genEp,saveS2):
    hist_train = hist_s2
    reply_train = reply_s2

    idxTrain = np.arange(len(hist_train))


    for ep in range(genEp):
        np.random.shuffle(idxTrain)
        for j in range(hist_train.shape[0] // headerSeq2Seq.BATCH_SIZE):
            X = hist_train[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE], :]
            Y_train = reply_train[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE], :]

            _, g_loss, _ = gen2.pretrain_step(sess, X, Y_train)

            if j == 0:
                print("Gen Train Loss = ", g_loss, ep)
                if saveS2:
                    gen2.save_model(sess, savepathG_S2_Seq2Seq)




def trainS1(gen1,disc1,base1,hist_s1,reply_s1,d_steps = 2,g_steps = 1,lr=0.000001):


    idxTrain = np.arange(len(hist_s1))
    teacher_forcing = False
    for ep in range(1):
        # Train discriminator

        for d in range(d_steps):
            print("D step for S1 ===", d)

            for j in range(0, hist_s1.shape[0] // headerSeq2Seq.BATCH_SIZE):
                #     jAns = np.random.choice((hist_train.shape[0] // headerSeq2Seq.BATCH_SIZE)-1, 20, replace=False)
                #     for j in jAns:
                X = hist_s1[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE], :]
                Y_train = reply_s1[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE], :]

                # Generate sentence
                if not teacher_forcing:
                    Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
                    teacher_forcing = True
                    _, sentence = gen1.generate(sess, X, Y)
                else:
                    # Teacher forcing
                    # Y = Y_train
                    Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
                    teacher_forcing = False
                    _, sentence = gen1.test_generate(sess, X, Y)

                # Pad sentence of variable length to header.REP_SEQ_LENGTH
                # rep_inp = np.full((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH), word_index['eos'])
                # rep_inp[:, :sentence.shape[1]] = sentence
                # sentence = rep_inp
                # sentence[sentence == 0] = word_index['eos']

                # Build a batch of half true and half false sentences
                Y_d = np.zeros((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH))
                sentence_index = np.random.random_integers((headerSeq2Seq.BATCH_SIZE - 1),
                                                           size=(int(headerSeq2Seq.BATCH_SIZE / 2), 1))

                Y_d[0:int(headerSeq2Seq.BATCH_SIZE / 2), :] = np.squeeze(Y_train[sentence_index, :])
                Y_d[int(headerSeq2Seq.BATCH_SIZE / 2):, :] = np.squeeze(sentence[sentence_index, :])
                X_d = np.copy(X)
                X_d[0:int(headerSeq2Seq.BATCH_SIZE / 2), :] = np.squeeze(np.copy(X[sentence_index, :]))
                X_d[int(headerSeq2Seq.BATCH_SIZE / 2):, :] = np.squeeze(np.copy(X[sentence_index, :]))

                X_d = toolsSeq2Seq.concat_hist_reply(X_d, Y_d, word_index)
                label_d = np.ones(headerSeq2Seq.BATCH_SIZE)
                label_d[int(headerSeq2Seq.BATCH_SIZE / 2):] = 0

                # print(np.sum(np.abs(X_d[0:int(headerSeq2Seq.BATCH_SIZE / 2), :] !=X_d[int(headerSeq2Seq.BATCH_SIZE / 2):, :])))
                # toolsSeq2Seq.convert_id_to_text(X_d,word_index)
                # toolsSeq2Seq.convert_id_to_text(X_d,word_index)
                _, d_loss, d_acc, _ = disc1.train_step(sess, X_d, label_d)
                if j % 50 == 0:
                    print("Discriminator loss = ", d_loss,"Discriminator accuracy = ",d_acc)
                    disc1.save_model(sess, savepathD_S1)
        gen1.assign_lr(sess, lr)
        for g in range(g_steps):
            print("G step for S1 ==== ", g)
            # jAns = np.random.choice((hist_s1.shape[0] // headerSeq2Seq.BATCH_SIZE)-1, 10, replace=False)

            for j in range(0, hist_s1.shape[0] // headerSeq2Seq.BATCH_SIZE):


                X = hist_s1[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE], :]

                # Y_train = reply_s1[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE], :]


                Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']


                gen_proba, sentence = gen1.generate(sess, X, Y)


                # disc_in = toolsSeq2Seq.concat_hist_reply(X, sentence, word_index)
                # disc_rewards = disc1.get_rewards(sess, disc_in)

                rewards = gen1.MC_reward(sess, X, sentence, headerSeq2Seq.MC_NUM, disc1, word_index)
                # b = np.tile(np.mean(np.array(rewards), axis=0), (headerSeq2Seq.BATCH_SIZE, 1))



                base1.train(sess, X, sentence, word_index, rewards)
                b = base1.get_baseline(sess, X, sentence, word_index,)


                _, adv_loss = gen1.advtrain_step(sess, X, sentence, sentence, rewards, b)
                base1.save_model(sess,'BaselineModel/S1/')
                gen1.save_model(sess,'GeneratorModel/S1/Reinforce/')


    return


def trainS2(gen2,disc2,base2,hist_s2,reply_s2,d_steps = 1,g_steps = 1,lr =0.00001):





    discAcc = []
    genRew = []
    wrongCount1 = []
    wrongCount2 = []
    wrongCount3 = []
    idxTrain = np.arange(len(hist_s2))
    teacher_forcing = True
    for ep in range(10):
        for j in range(0, hist_s2.shape[0] // headerSeq2Seq.BATCH_SIZE):

                #     jAns = np.random.choice((hist_train.shape[0] // headerSeq2Seq.BATCH_SIZE)-1, 20, replace=False)
                #     for j in jAns:
                X = hist_s2[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE], :]
                Y_train = reply_s2[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE], :]

                # Generate sentence
                if not teacher_forcing:
                    Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
                    teacher_forcing = True
                    _, sentence = gen2.generate(sess, X, Y)
                else:
                    # Teacher forcing
                    # Y = Y_train
                    Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
                    teacher_forcing = False
                    _, sentence = gen2.test_generate(sess, X, Y)

                # Pad sentence of variable length to header.REP_SEQ_LENGTH
                # rep_inp = np.full((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH), word_index['eos'])
                # rep_inp[:, :sentence.shape[1]] = sentence
                # sentence = rep_inp
                # sentence[sentence == 0] = word_index['eos']

                # Build a batch of half true and half false sentences
                Y_d = np.zeros((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH))
                sentence_index = np.random.random_integers((headerSeq2Seq.BATCH_SIZE - 1),
                                                           size=(int(headerSeq2Seq.BATCH_SIZE / 2), 1))
                # print("sentence indexes")
                # print(sentence_index)

                Y_d[0:int(headerSeq2Seq.BATCH_SIZE / 2), :] = np.squeeze(Y_train[sentence_index, :])
                Y_d[int(headerSeq2Seq.BATCH_SIZE / 2):, :] = np.squeeze(sentence[sentence_index, :])
                # print("Ground Truth Replies")
                # toolsSeq2Seq.convert_id_to_text(Y_d[0:int(headerSeq2Seq.BATCH_SIZE / 2)], word_index)
                # print("Generator Replies")
                # toolsSeq2Seq.convert_id_to_text(Y_d[int(headerSeq2Seq.BATCH_SIZE / 2):], word_index)

                X_d = np.copy(X)
                X_d[0:int(headerSeq2Seq.BATCH_SIZE / 2), :] = np.squeeze(np.copy(X[sentence_index, :]))
                X_d[int(headerSeq2Seq.BATCH_SIZE / 2):, :] = np.squeeze(np.copy(X[sentence_index, :]))


                X_d = toolsSeq2Seq.concat_hist_reply(X_d, Y_d, word_index)
                # print("Ground Truth Concat")
                # toolsSeq2Seq.convert_id_to_text(X_d[0:int(headerSeq2Seq.BATCH_SIZE / 2)], word_index)
                # print("Generator Concat")
                # toolsSeq2Seq.convert_id_to_text(X_d[int(headerSeq2Seq.BATCH_SIZE / 2):], word_index)
                #
                # input("wait")

                label_d = np.ones(headerSeq2Seq.BATCH_SIZE)
                label_d[int(headerSeq2Seq.BATCH_SIZE / 2):] = 0
                # print("Labels for Real Dialogs")
                # print(label_d[:int(headerSeq2Seq.BATCH_SIZE / 2)])
                # print("Labels for Fake Dialogs")
                # print(label_d[int(headerSeq2Seq.BATCH_SIZE / 2):])
                # input("wait")
                # print(np.sum(np.abs(X_d[0:int(headerSeq2Seq.BATCH_SIZE / 2), :] !=X_d[int(headerSeq2Seq.BATCH_SIZE / 2):, :])))
                # toolsSeq2Seq.convert_id_to_text(X_d,word_index)
                # toolsSeq2Seq.convert_id_to_text(X_d,word_index)
                _, d_loss, d_acc, _ = disc2.train_step(sess, X_d, label_d)
                if j % 50 == 0:
                    print("Discriminator loss = ", d_loss, "Discriminator accuracy = ", d_acc)
                    disc2.save_model(sess, savepathD_S2)

    for ep in range(100000000):
        # Train discriminator

        disc_acc = 0
        for ii in range(hist_s2.shape[0] // headerSeq2Seq.BATCH_SIZE):
            X = hist_s2[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]
            Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
            Y_train = reply_s2[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]

            gen_proba_test, sentence_test = gen2.test_generate(sess, X, Y)

            X_d = toolsSeq2Seq.concat_hist_reply(X, sentence_test, word_index)
            label_d = np.zeros(headerSeq2Seq.BATCH_SIZE)
            temp_loss, temp_acc = disc2.get_loss(sess, X_d, label_d)
            disc_acc = disc_acc + temp_acc

            X_d = toolsSeq2Seq.concat_hist_reply(X, Y_train, word_index)
            label_d = np.ones(headerSeq2Seq.BATCH_SIZE)
            temp_loss, temp_acc = disc2.get_loss(sess, X_d, label_d)
            disc_acc = disc_acc + temp_acc

        disc_acc = disc_acc / (2 * (hist_s2.shape[0] // headerSeq2Seq.BATCH_SIZE))
        discAcc.append(disc_acc)

        for d in range(d_steps):
            np.random.shuffle(idxTrain)
            print("D step for S2 ===", d)

            j = np.random.randint((hist_s2.shape[0] // headerSeq2Seq.BATCH_SIZE))


            X = hist_s2[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE], :]
            Y_train = reply_s2[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE], :]



            for _ in range(3):

                    Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
                    _, sentence = gen2.generate(sess, X, Y)
                    Y_d = np.zeros((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH))
                    sentence_index = np.random.random_integers((headerSeq2Seq.BATCH_SIZE - 1),
                                                               size=(int(headerSeq2Seq.BATCH_SIZE / 2), 1))

                    Y_d[0:int(headerSeq2Seq.BATCH_SIZE / 2), :] = np.squeeze(Y_train[sentence_index, :])
                    Y_d[int(headerSeq2Seq.BATCH_SIZE / 2):, :] = np.squeeze(sentence[sentence_index, :])
                    X_d = np.copy(X)
                    X_d[0:int(headerSeq2Seq.BATCH_SIZE / 2), :] = np.squeeze(np.copy(X[sentence_index, :]))
                    X_d[int(headerSeq2Seq.BATCH_SIZE / 2):, :] = np.squeeze(np.copy(X[sentence_index, :]))

                    X_d = toolsSeq2Seq.concat_hist_reply(X_d, Y_d, word_index)
                    label_d = np.ones(headerSeq2Seq.BATCH_SIZE)
                    label_d[int(headerSeq2Seq.BATCH_SIZE / 2):] = 0


                    _, d_loss, d_acc, _ = disc2.train_step(sess, X_d, label_d)







            print("Discriminator loss = ", d_loss,"Discriminator accuracy = ",d_acc)
                        # disc2.save_model(sess, savepathD_S2)


        disc_acc = 0
        for ii in range(hist_s2.shape[0] // headerSeq2Seq.BATCH_SIZE):
            X = hist_s2[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]
            Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
            Y_train = reply_s2[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]

            gen_proba_test, sentence_test = gen2.test_generate(sess, X, Y)

            X_d = toolsSeq2Seq.concat_hist_reply(X, sentence_test, word_index)
            label_d = np.zeros(headerSeq2Seq.BATCH_SIZE)
            temp_loss,temp_acc = disc2.get_loss(sess, X_d,label_d)
            disc_acc = disc_acc + temp_acc

            X_d = toolsSeq2Seq.concat_hist_reply(X, Y_train, word_index)
            label_d = np.ones(headerSeq2Seq.BATCH_SIZE)
            temp_loss, temp_acc = disc2.get_loss(sess, X_d, label_d)
            disc_acc = disc_acc + temp_acc

        disc_acc  = disc_acc/  (2*(hist_s2.shape[0] // headerSeq2Seq.BATCH_SIZE))
        discAcc.append(disc_acc)

        gen_rew = 0
        for ii in range(hist_s2.shape[0] // headerSeq2Seq.BATCH_SIZE):
            X = hist_s2[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]
            Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
            gen_proba_test, sentence_test = gen2.test_generate(sess, X, Y)
            X_d = toolsSeq2Seq.concat_hist_reply(X, sentence_test, word_index)
            temp_rew = disc2.get_rewards(sess, X_d)
            gen_rew = gen_rew + np.sum(temp_rew[:,1])

        genRew.append(gen_rew)


        w1,w2,w3 = testS2_onTest(gen2, hist_s2, reply_s2, test_hist_s2, test_reply_s2, test_hist_s2_OOV, test_reply_s2_OOV)
        wrongCount1.append(w1)
        wrongCount2.append(w2)
        wrongCount3.append(w3)
        gen2.assign_lr(sess, lr)
        for g in range(g_steps):
            np.random.shuffle(idxTrain)
            print("G step for S1 ==== ", g)
            # jAns = np.random.choice((hist_s1.shape[0] // headerSeq2Seq.BATCH_SIZE)-1, 10, replace=False)

            j = np.random.randint((hist_s2.shape[0] // headerSeq2Seq.BATCH_SIZE))

            X = hist_s2[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE], :]

            testS2(gen2,disc2,hist_s2)
            print(j)

            Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']


            gen_proba, sentence = gen2.generate(sess, X, Y)



            rewards = gen2.MC_reward(sess, X, sentence, headerSeq2Seq.MC_NUM, disc1, word_index)
                    # b = np.tile(np.mean(np.array(rewards), axis=0), (headerSeq2Seq.BATCH_SIZE, 1))

            base2.train(sess, X, sentence, word_index, rewards)
            b = base2.get_baseline(sess, X, sentence, word_index,)


            _, adv_loss = gen2.advtrain_step(sess, X, sentence, sentence, rewards, b)
            # base2.save_model(sess,'BaselineModel/S2/')
            # gen2.save_model(sess,'GeneratorModel/S2/Reinforce/')


        gen_rew = 0
        for ii in range(hist_s2.shape[0] // headerSeq2Seq.BATCH_SIZE):
            X = hist_s2[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]
            Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
            gen_proba_test, sentence_test = gen2.test_generate(sess, X, Y)
            X_d = toolsSeq2Seq.concat_hist_reply(X, sentence_test, word_index)
            temp_rew = disc2.get_rewards(sess, X_d)
            gen_rew = gen_rew + np.sum(temp_rew[:,1])
        genRew.append(gen_rew)

        w1,w2,w3 = testS2_onTest(gen2, hist_s2, reply_s2, test_hist_s2, test_reply_s2, test_hist_s2_OOV, test_reply_s2_OOV)
        wrongCount1.append(w1)
        wrongCount2.append(w2)
        wrongCount3.append(w3)
        fig = plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(discAcc, '-b', label='Disc Acc')
        plt.subplot(3, 1, 2)
        plt.plot(genRew, '-b', label='Gen Rewards')
        plt.subplot(3, 1, 3)
        plt.plot(wrongCount1, '-g', label='Train')
        plt.plot(wrongCount2, '-b', label='Test')
        plt.plot(wrongCount3, '-r', label='TestOOV')
        fig.savefig('5GenDisc.pdf')  # save the figure to file
        plt.close(fig)  # close the figure

    return

def trainS3():
    return


def testS1(gen1,disc1,hist_s1):


            reward_list = []

            for ii in range(hist_s1.shape[0] // headerSeq2Seq.BATCH_SIZE):
                X = hist_s1[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]
                Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
                # Y_train = reply_s1[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]

                gen_proba_test, sentence_test = gen1.test_generate(sess, X, Y)

                X_d = toolsSeq2Seq.concat_hist_reply(X, sentence_test, word_index)

                disc_proba = disc1.get_rewards(sess, X_d)

                total_rew = total_rew + np.sum(disc_proba[:,1])
                reward_list.append(disc_proba[:,1])


            reward_list = np.array(reward_list).flatten()
            print("Total Reward for S1 From Disc ===", total_rew, " Time == ", str(datetime.now().time()))
            f = plt.figure()
            plt.hist(reward_list, bins='auto')  # arguments are passed to np.histogram
            plt.title("Histogram S1 Rewards")
            f.savefig("Hist_S1.pdf", bbox_inches='tight')
            f.close()
            return


def testS2(gen2, disc2, hist_s2):
    reward_list = []
    total_rew = 0
    counter = 0
    for ii in range(hist_s2.shape[0] // headerSeq2Seq.BATCH_SIZE):
        X = hist_s2[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]
        Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
        Y_train = reply_s2[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]

        gen_proba_test, sentence_test = gen2.test_generate(sess, X, Y)

        X_d = toolsSeq2Seq.concat_hist_reply(X, sentence_test, word_index)

        disc_proba = disc2.get_rewards(sess, X_d)

        total_rew = total_rew + np.sum(disc_proba[:, 1])
        reward_list.append(disc_proba[:, 1])
        cc = -1
        # print("/////////////////////")
        for d_p in disc_proba[:,1]:
                cc = cc +1
            # if d_p>0.45:
                if np.sum(sentence_test[cc] != Y_train[cc]) >0 and counter<3:
                    temp_proba = gen_proba_test[cc]
                    temp_av = temp_proba[np.arange(len(temp_proba)), Y_train[cc, :]]

                    print("Predicted === ", ii,ii * headerSeq2Seq.BATCH_SIZE + cc,
                          toolsSeq2Seq.convert_return_id_to_text(sentence_test[cc],word_index)," Correct == ", toolsSeq2Seq.convert_return_id_to_text(Y_train[cc],word_index),
                          " Reward ",d_p , " Probas of Correct Words = ", temp_av )


                    counter = counter + 1

    reward_list = np.array(reward_list).flatten()
    print("Total Reward for S2 From Disc ===", total_rew, " Time == ", str(datetime.now().time()))
    f = plt.figure()
    plt.hist(reward_list, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram S2 Rewards")
    f.savefig("Hist_S2.pdf", bbox_inches='tight')
    plt.close()
    return
def testS2_onTest(gen2,hist_s2,reply_s2,test_hist_s2,test_reply_s2,test_hist_s2_OOV,test_reply_s2_OOV):
    total_correct = 0
    total_wrong = 0
    for ii in range(hist_s2.shape[0] // headerSeq2Seq.BATCH_SIZE):
        X = hist_s2[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]

        Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
        Y_train = reply_s2[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]

        gen_proba_test, sentence_test = gen2.test_generate(sess, X, Y)


        for jj in range(headerSeq2Seq.BATCH_SIZE):

            total_correct = total_correct + np.sum(sentence_test == Y_train)
            total_wrong = total_wrong + np.sum(sentence_test != Y_train)

    print("Correct Words for Train === ", total_correct," Wrong Words for Train ======== ", total_wrong, " Time == ", str(datetime.now().time()))
    total_wrong1 = total_wrong
    total_correct = 0
    total_wrong = 0
    for ii in range(test_hist_s2.shape[0] // headerSeq2Seq.BATCH_SIZE):
        X = test_hist_s2[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]

        Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
        Y_train = test_reply_s2[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]

        gen_proba_test, sentence_test = gen2.test_generate(sess, X, Y)
        for jj in range(headerSeq2Seq.BATCH_SIZE):
            total_correct = total_correct + np.sum(sentence_test == Y_train)
            total_wrong = total_wrong + np.sum(sentence_test != Y_train)

    print("Correct Words for Test === ", total_correct, " Wrong Words for Test  ======== ", total_wrong, " Time == ",
          str(datetime.now().time()))
    total_wrong2 = total_wrong
    total_correct = 0
    total_wrong = 0
    for ii in range(test_hist_s2_OOV.shape[0] // headerSeq2Seq.BATCH_SIZE):
        X = test_hist_s2_OOV[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]

        Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
        Y_train = test_reply_s2_OOV[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]

        gen_proba_test, sentence_test = gen2.test_generate(sess, X, Y)


        for jj in range(headerSeq2Seq.BATCH_SIZE):
            total_correct = total_correct + np.sum(sentence_test == Y_train)
            total_wrong = total_wrong + np.sum(sentence_test != Y_train)

    print("Correct Words for Test OOV === ", total_correct, " Wrong Words for Test OOV ======== ", total_wrong, " Time == ",
          str(datetime.now().time()))
    total_wrong3 = total_wrong
    return total_wrong1,total_wrong2,total_wrong3


def testS3(gen1,gen2,disc1,disc2,hist_s3):

    reward_list = []

    for ii in range(hist_s3.shape[0] // headerSeq2Seq.BATCH_SIZE):
        X = hist_s3[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]
        Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
        # Y_train = reply_s1[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]

        gen_proba_test, sentence_test = gen1.test_generate(sess, X, Y)

        X_d = toolsSeq2Seq.concat_hist_reply(X, sentence_test, word_index)

        gen_proba_test_last, sentence_test_last = gen2.test_generate(sess, X_d, Y)

        X_d_last = toolsSeq2Seq.concat_hist_reply(X_d, sentence_test_last, word_index)



        disc_proba = disc2.get_rewards(sess, X_d_last)

        total_rew = total_rew + np.sum(disc_proba[:, 1])
        reward_list.append(disc_proba[:, 1])

    reward_list = np.array(reward_list).flatten()
    print("Total Reward for S3 From Disc ===", total_rew, " Time == ", str(datetime.now().time()))
    f = plt.figure()
    plt.hist(reward_list, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram S3 Rewards")
    f.savefig("Hist_S3.pdf", bbox_inches='tight')
    f.close()
    return

    return


# Pre train if necessary



# if ~header.DO_RESTORE:
#    pretrain.pretrain(savepathD, savepathG)

# Adversarial training

savepathG_S1_Seq2Seq = 'GeneratorModel/S1/Seq2Seq/'
savepathG_S2_Seq2Seq = 'GeneratorModel/S2/Seq2Seq/'
savepathD_S1 = 'DiscModel/S1/'
savepathD_S2 = 'DiscModel/S2/'
savepathG_S2_ReInf = 'GeneratorModel/S2/Reinforce/'
savepathB_S2 = 'BaselineModel/S2/'


# Load data
(embedding_matrix,
train_data,test_data,
 word_index) = readFBTask1Seq2Seq.create_con(False, headerSeq2Seq.MAX_SEQ_LENGTH,headerSeq2Seq.REP_SEQ_LENGTH)


create_folders()

EMB_DIM = len(word_index) + 1  # embedding dimension
END_TOKEN = word_index.get("eos")


# Model

gen1,disc1,base1,gen2,disc2,base2,disc3,base3 = create_networks(EMB_DIM,END_TOKEN,word_index)

# TF setting


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())




gen1.assign_emb(sess, embedding_matrix)
disc1.assign_emb(sess, embedding_matrix)
base1.assign_emb(sess,embedding_matrix)

gen2.assign_emb(sess, embedding_matrix)
disc2.assign_emb(sess, embedding_matrix)
base2.assign_emb(sess,embedding_matrix)

disc3.assign_emb(sess, embedding_matrix)
base3.assign_emb(sess,embedding_matrix)


train_Disc = False
train_Gen = True
discEp = 0
genEp = 5
# Restore pre trained models



#Pretraining of Discriminator and Generator
gen1.assign_lr(sess,0.0004)
gen2.assign_lr(sess,0.0004)


try:
    print("trying to restore Disc")
    # discriminator.restore_model(sess, savepathD)
    # baseline.restore_model(sess, savepathB)
except:
    print("Disc could not be restored")
    pass
try:
    print("trying to restore Gen")
    # generator.restore_model(sess, savepathG)
except:
    print("Gen could not be restored")
    pass


# pretrain.pretrain(sess,disc1,gen1,discEp,genEp,train_Disc,train_Gen,savepathD_S1,savepathG_S1_Seq2Seq)
# pretrain.pretrain(sess,disc2,gen2,discEp,genEp,train_Disc,train_Gen,savepathD_S2,savepathG_S2_Seq2Seq)


hist_s1 = train_data["hist_s1"]
hist_s2 = train_data["hist_s2"]
hist_s3 = train_data["hist_s3"]

lite_hist_s1 = train_data["lite_hist_s1"]
lite_hist_s2 = train_data["lite_hist_s2"]
lite_hist_s3 =train_data["lite_hist_s3"]

reply_s1 = train_data["reply_s1"]
reply_s2 = train_data["reply_s2"]
reply_s3 = train_data["reply_s3"]

lite_reply_s1 =train_data["lite_reply_s1"]
lite_reply_s2 = train_data["lite_reply_s2"]
lite_reply_s3 =  train_data["lite_reply_s3"]

test_hist_s2 = test_data["hist_s2"]
test_hist_s2_OOV = test_data["hist_s2_OOV"]

test_reply_s2 = test_data["reply_s2"]
test_reply_s2_OOV = test_data["reply_s2_OOV"]



gen2.restore_model(sess,savepathG_S2_Seq2Seq)
# preTrainS2(gen2, hist_s2, reply_s2,10,True)

testS2_onTest(gen2,hist_s2,reply_s2,test_hist_s2,test_reply_s2,test_hist_s2_OOV,test_reply_s2_OOV)

# for ii in range(hist_s2.shape[0] // headerSeq2Seq.BATCH_SIZE):
#     X = hist_s2[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]
#     Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
#     Y_train = reply_s2[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]
#
#     gen_proba_test, sentence_test = gen2.test_generate(sess, X, Y)
#
#     X_d = toolsSeq2Seq.concat_hist_reply(X, sentence_test, word_index)
#
#     for cc in range(headerSeq2Seq.BATCH_SIZE):
#
#         print("history")
#         toolsSeq2Seq.convert_id_to_text(X[cc:cc+1],word_index)
#         print("generator reply")
#         toolsSeq2Seq.convert_id_to_text(sentence_test[cc:cc+1], word_index)
#         print("ground truth")
#         toolsSeq2Seq.convert_id_to_text(Y_train[cc:cc+1], word_index)
#         print("concat history with reply ")
#         toolsSeq2Seq.convert_id_to_text(X_d[cc:cc + 1],word_index)
#

# disc2.restore_model(sess,savepathD_S2)
# base2.restore_model(sess,savepathB_S2)
#



# for ep in range(50):
#     preTrainS2(gen2,hist_s2,reply_s2,2)
#     testS2_onTest(gen2, hist_s2, reply_s2, test_hist_s2, test_reply_s2, test_hist_s2_OOV, test_reply_s2_OOV)
#


for e in range(10000):


    trainS2(gen2,disc2,base2,hist_s2,reply_s2,1,20)
    testS2_onTest(gen2,hist_s2,reply_s2,test_hist_s2,test_reply_s2,test_hist_s2_OOV,test_reply_s2_OOV)
    testS2(gen2,disc2,hist_s2)



for e in range(100):

    trainS1(gen1,disc1,base1,hist_s1,reply_s1,2,1,0.000001)
    testS1(gen1,disc1,hist_s1)





#trainS3()

#testS1()
#
#testS3()













# ind = -1
# hist_batch = 8
# max_count = 0
#
#
#             # input("wait")
#     # Train generator
#
#     print("//////////////////////////////////////////////////////////////////////////////////////////////////")
#
#
#     total_rew = 0
#     total_wrong = 0
#     proba_list_all_before = []
#     av_true = 0
#     for ii in range(hist_train.shape[0] // headerSeq2Seq.BATCH_SIZE):
#         X = hist_train[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]
#
#         X_one = np.tile(np.copy(X[0, :]), (headerSeq2Seq.BATCH_SIZE, 1))
#         X = X_one
#
#
#
#         Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
#         Y_train = reply_train[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]
#
#         Y_one = np.tile(np.copy(Y_train[0, :]), (headerSeq2Seq.BATCH_SIZE, 1))
#         Y_train = Y_one
#
#         gen_proba_test, sentence_test = generator.generate(sess, X, Y)
#
#         X_d = toolsSeq2Seq.concat_hist_reply(X, sentence_test, word_index)
#         label_d = np.zeros(headerSeq2Seq.BATCH_SIZE)
#         disc_proba = discriminator.get_rewards(sess, X_d)
#
#         if ii %1 ==0:
#             toolsSeq2Seq.convert_id_to_text(X[0:1], word_index)
#             toolsSeq2Seq.convert_id_to_text(Y_train[0:1], word_index)
#         # for jj in range(headerSeq2Seq.BATCH_SIZE):
#             # toolsSeq2Seq.convert_id_to_text(X[jj:jj + 1], word_index)
#             toolsSeq2Seq.convert_id_to_text(sentence_test[0:10], word_index)
#
#             print("Reward=== ",disc_proba[0:10, 1])
#             input("wait")
#         # input("wait")
#
#         # #     # if np.sum(sentence_test[jj] != Y_train[jj])>0:
#         # #         print(sentence_test[jj])
#         # #         toolsSeq2Seq.convert_id_to_text(X[jj:jj+1],word_index)
#         # #
#         # #         toolsSeq2Seq.convert_id_to_text(Y_train[jj:jj+1],word_index)
#         # #         input("wait")
#         # total_correct = total_correct + np.sum(sentence_test == Y_train)
#         # total_wrong = total_wrong + np.sum(sentence_test != Y_train)
#         #
#         # for p_i in range(gen_proba_test.shape[0]):
#         #     temp_proba = gen_proba_test[p_i]
#         #     temp_av = temp_proba[np.arange(len(temp_proba)), Y_train[p_i, :]]
#         #     temp_av[Y_train[p_i, :] == word_index["eos"]] = 0
#         #     proba_list_all_before.append(temp_av)
#         #
#         #     av_proba = np.sum(temp_av) / np.sum(Y_train[p_i, :] != word_index["eos"])
#         #     av_true = av_true + av_proba
#         #     # print(av_proba)
#         #     # input("wait")
#         total_rew = total_rew + np.sum(disc_proba[:,1])
#     if total_rew > max_count:
#         max_count = total_rew
#     print("Before Total Reward ===", total_rew," Maximum Reward ======== ", max_count, " Time == ", str(datetime.now().time()))
#
#     generator.save_model(sess,savepathG)
#     for g in range(g_steps):
#         print("G step == ", g)
#         jAns = np.random.choice((hist_train.shape[0] // headerSeq2Seq.BATCH_SIZE)-1, 10, replace=False)
#
#         for j in jAns:
#
#
#
#             # toolsSeq2Seq.convert_id_to_text(reply_train[ind:ind+1, :],word_index)
#             # print("////////////////////////////////////////////////////////////")
#             # rewards = np.zeros(((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)))
#             # b = np.zeros(((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)))
#             # for ii in range(int(headerSeq2Seq.BATCH_SIZE//hist_batch)):
#             X = hist_train[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE], :]
#             # X_one = np.tile(np.copy(X[0, :]), (headerSeq2Seq.BATCH_SIZE, 1))
#             # X = X_one
#
#             Y_train = reply_train[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE], :]
#             # Y_one = np.tile(np.copy(Y_train[0, :]), (headerSeq2Seq.BATCH_SIZE, 1))
#             # Y_train = Y_one
#
#             Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
#
#
#             gen_proba, sentence = generator.generate(sess, X, Y)
#
#
#             # gen_proba_test, sentence_test = generator.test_generate(sess, X, Y)
#             #
#             # total_correct = np.sum(sentence_test == Y_train)
#             # total_wrong =  np.sum(sentence_test != Y_train)
#             # print(" Batch Correct Before === ", total_correct, " Batch Wrong === ", total_wrong," Time == ", str(datetime.now().time()))
#
#
#
#             #
#             #
#             # proba_list = []
#             # for p_i in range(gen_proba.shape[0]):
#             #     temp_proba = gen_proba[p_i]
#             #     temp_av = temp_proba[np.arange(len(temp_proba)), sentence[p_i, :]]
#             #     temp_av[sentence[p_i, :] == word_index["eos"]] = 0
#             #     proba_list.append(temp_av)
#             #
#             # # #
#             # #
#             # av_proba = np.sum(np.array(proba_list)) / np.sum(sentence[:, :] != word_index["eos"])
#             # print("Before Training Average Proba = ", av_proba)
#             #
#
#
#
#
#             disc_in = toolsSeq2Seq.concat_hist_reply(X, sentence, word_index)
#             disc_rewards = discriminator.get_rewards(sess, disc_in)
#
#             rewards = generator.MC_reward(sess, X, sentence, headerSeq2Seq.MC_NUM, discriminator, word_index)
#             # b = np.tile(np.mean(np.array(rewards), axis=0), (headerSeq2Seq.BATCH_SIZE, 1))
#
#
#
#             baseline.train(sess, X, sentence, word_index, rewards)
#             b = baseline.get_baseline(sess, X, sentence, word_index,)
#
#
#
#             # b = np.zeros((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH))
#             # toolsSeq2Seq.convert_id_reward_to_text(np.array(sentence)[0:4],disc_rewards[:4,1],rewards[:4],b[:4],np.array(Y_train)[0:4],
#             #                                        word_index)
#
#             # input("wait")
#
#
#             # part0, part1, part2, part3, part4, part5 = generator.get_adv_loss(sess, X, sentence, sentence, rewards, b)
#             # print("*****************************Loses Before***********************")
#             # print(part0)
#             # print("*************Part1*************")
#             # print(part1)
#             # print("*************Part2*************")
#             # print(part2)
#             # print("*************Part3*************")
#             # print(part3)
#             # print("*************Part4*************")
#             # print(part4)
#             # print("*************Part5*************")
#             # print(part5)
#             # print("*****************************Loses Before***********************")
#
#
#             # if ep > 2:
#             # toolsSeq2Seq.convert_id_to_text(X[0:1], word_index)
#             # toolsSeq2Seq.convert_id_to_text(sentence[0:1], word_index)
#             #
#             _, adv_loss = generator.advtrain_step(sess, X, sentence, sentence, rewards, b)
#
#             # part0, part1, part2, part3, part4, part5 = generator.get_adv_loss(sess, X, sentence, sentence, rewards, b)
#             # print("*****************************Loses After***********************")
#             # print(part0)
#             # print("*************Part1*************")
#             # print(part1)
#             # print("*************Part2*************")
#             # print(part2)
#             # print("*************Part3*************")
#             # print(part3)
#             # print("*************Part4*************")
#             # print(part4)
#             # print("*************Part5*************")
#             # print(part5)
#             # print("*****************************Loses After***********************")
#             #
#             #
#             #
#             gen_proba_after, sentence_after = generator.generate(sess, X, sentence)
#
#             # proba_list_after = []
#             # for p_i in range(gen_proba_after.shape[0]):
#             #     temp_proba = gen_proba_after[p_i]
#             #     temp_av = temp_proba[np.arange(len(temp_proba)), sentence[p_i, :]]
#             #     temp_av[sentence[p_i, :] == word_index["eos"]] = 0
#             #     proba_list_after.append(temp_av)
#             #
#             # #
#             # #
#             # av_proba = np.sum(np.array(proba_list_after)) / np.sum(sentence[:, :] != word_index["eos"])
#             # print("After Training Average Proba = ", av_proba)
#             #
#             # for b, a in zip(proba_list,proba_list_after):
#             #     print("Before Proba ")
#             #     print(b)
#             #     print("After Proba ")
#             #     print(a)
#             # #
#             # #
#             # input("wait")
#             # # print("baseline training")
#
#
#
#
#
#
#             baseline.save_model(sess,savepathB)
#
#             # toolsSeq2Seq.convert_id_to_text(X[0:1], word_index)
#             # toolsSeq2Seq.convert_id_to_text(sentence[0:1], word_index)
#             # input("wait")
#
#             # gen_proba_test, sentence_test = generator.test_generate(sess, X, Y)
#             #
#             # total_correct = np.sum(sentence_test == Y_train)
#             # total_wrong =  np.sum(sentence_test != Y_train)
#             # print(" Batch Correct After === ", total_correct, " Batch Wrong === ", total_wrong," Time == ", str(datetime.now().time()))
#
#             # if g %1 ==0:
#             #     total_correct = 0
#             #     total_wrong = 0
#             #     proba_list_all_after = []
#             #     for ii in range(hist_train.shape[0] // headerSeq2Seq.BATCH_SIZE):
#             #         X = hist_train[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]
#             #         Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
#             #         Y_train = reply_train[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]
#             #
#             #         gen_proba_test, sentence_test = generator.test_generate(sess, X, Y)
#             #         for jj in range(headerSeq2Seq.BATCH_SIZE):
#             #              if np.sum(sentence_test[jj] != Y_train[jj])>0:
#             #
#             #
#             #                  # toolsSeq2Seq.convert_id_to_text(X[jj:jj+1],word_index)
#             #                  # toolsSeq2Seq.convert_id_to_text(sentence_test[jj:jj+1],word_index)
#             #                  # toolsSeq2Seq.convert_id_to_text(Y_train[jj:jj+1],word_index)
#             #                  temp_proba = gen_proba_test[jj]
#             #                  temp_av = temp_proba[np.arange(len(temp_proba)), Y_train[jj, :]]
#             #                  temp_av[Y_train[jj, :] == word_index["eos"]] = 0
#             #                  # print("Before Proba")
#             #                  # print(proba_list_all_before[ii * headerSeq2Seq.BATCH_SIZE+jj])
#             #                  # print("After Proba")
#             #                  # print(temp_av)
#             #                  # input("wait")
#             #
#             #
#             #         total_correct = total_correct + np.sum(sentence_test == Y_train)
#             #         total_wrong = total_wrong + np.sum(sentence_test != Y_train)
#
#
#
#                     # Y_train = reply_train[ind: headerSeq2Seq.BATCH_SIZE + ind, :]
#             # print("ind ==== ", ind)
#             #
#             # # print("*********************************")
#             #
#             #
#             # Y_one = np.tile(np.copy(Y_train[0, :]), (headerSeq2Seq.BATCH_SIZE, 1))
#             # toolsSeq2Seq.convert_id_to_text(np.array(X_one)[0:3, :], word_index)
#             #
#             #
#             # gen_proba_test, sentence_test = generator.test_generate(sess, X_one, Y)
#             # gen_proba_test_one = np.copy(gen_proba_test[0, :, :])
#             # print("Target words probabilities")
#             # print(gen_proba_test_one[np.arange(len(gen_proba_test_one)), Y_one[0, :]])
#             #
#             # print("Created Sentence with argmax")
#             # toolsSeq2Seq.convert_id_to_text(np.array(sentence_test)[0:1, :], word_index)
#             # print("Target Sentences")
#             # toolsSeq2Seq.convert_id_to_text(np.array(Y_one)[0:1, :], word_index)
#             # # print(Y_train[0, :])
#             # print("Number of Correct Words  =============================", np.sum(sentence_test == Y_one))
#             #
#             # temp_av = gen_proba_test_one[np.arange(len(gen_proba_test_one)), Y_one[0, :]]
#             #
#             # temp_av[Y_train[0, :] == word_index["eos"]] = 0
#             # av_proba = np.sum(temp_av) / np.sum(Y_one[0, :] != word_index["eos"])
#             # print("Average Prob for Chosing Right Action  =============================", av_proba)
#             #
#             # if av_proba > max_avg_prob[ind]:
#             #     print(
#             #         "**********************************************MAX FOUND**************************************************************************")
#             #     max_avg_prob[ind] = av_proba
#             # print(max_avg_prob)
#             # # print(np.array(gen_proba).shape)
#             #
#             # # print("FULL SAMPLED SENTENCES")
#             # # tools.convert_id_to_text(np.array(sentence)[:5,:], word_index)
#             # sentence_old = np.copy(np.array(sentence))
#             # # tools.convert_id_to_text(np.array(Y_train)[0:1, :], word_index)
#             #
#             # gen_proba = np.array(gen_proba)
#             #
#             # gen_proba = gen_proba[0, :, :]
#             # # print("SENTENCES WITH CHOSING WORD PROBABILITIES")
#             # # tools.convert_id_to_text(np.array(sentence_test)[0:1, :], word_index)
#             # # print(gen_proba[np.arange(len(gen_proba)), sentence[0, :]])
#             # # input("wait 1")
#             # # print(gen_proba[0,1, sentence[0,1]])
#             #
#             #
#             # # X_one =np.tile(np.copy(X[0,:]),(header.BATCH_SIZE,1))
#             #
#             #
#             # # Sen_one = np.tile(np.copy(sentence[0, :]), (header.BATCH_SIZE, 1))
#             # #
#             # # sentence = Sen_one
#             # #
#             # #
#             # # rep_inp = np.full((header.BATCH_SIZE, header.REP_SEQ_LENGTH), word_index['eos'])
#             # # rep_inp[:, :sentence.shape[1]] = sentence
#             # # sentence = rep_inp
#             # # sentence[sentence ==0] = word_index['eos']
#             #
#             #
#             #
#             # sen_rand = np.random.random_integers(len(word_index), size=(headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH))
#             # stop_index = np.random.random_integers(headerSeq2Seq.REP_SEQ_LENGTH, size=(headerSeq2Seq.BATCH_SIZE, 1))
#             # for i in range(headerSeq2Seq.BATCH_SIZE):
#             #     sen_rand[i, stop_index[i][0]:] = word_index["eos"]
#             #
#             # # disc_in_rand = toolsSeq2Seq.concat_hist_reply(X, sen_rand, word_index)
#             # # disc_in_real = toolsSeq2Seq.concat_hist_reply(X, Y_train, word_index)
#             # # disc_rewards_rand = discriminator.get_rewards(sess, disc_in_rand)
#             # # disc_rewards_real = discriminator.get_rewards(disc_in_real)
#             # # disc_rewards_real = discriminator.get_rewards(sess, disc_in_real)
#             # print("///////////////////////////////")
#             # print("Discriminator Rewards for MC Sentences = ", disc_rewards[0:3, 1])
#             # print("Discriminator Rewards for Random Sentences = ", disc_rewards_rand[0:3, 1])
#             # print("Discriminator Rewards for Real Sentences = ", disc_rewards_real[0:3, 1])
#             # print("///////////////////////////////")
#             # print("MC sample ids for first 3 Sentence")  # depend
#             # print(np.array(sentence)[0:3])
#             # toolsSeq2Seq.convert_id_to_text(np.array(sentence)[0:3], word_index)
#             # # print("CORRECT SENTENCE")
#             #
#             #
#             # rewards = generator.MC_reward(sess, X_one, sentence, headerSeq2Seq.MC_NUM, discriminator, word_index)
#             #
#             # print("///////////////////////////////")
#             # print("MC Rewards for first 3 Sentence")  # depend
#             # print(np.array(rewards)[0:3])
#             # # input("wait 3 ")
#             #
#             # # baseline_loss = baseline.train(X_one, sentence, rewards, word_index)
#             # b = np.tile(np.mean(np.array(rewards), axis=0), (headerSeq2Seq.BATCH_SIZE, 1))
#             # # b = baseline.get_baseline(X_one,sentence,word_index)
#             #
#             # print("///////////////////////////////")
#             # print("Baseline Rewards for first 3 Sentence")  # depend
#             # print(np.array(b)[30:33])
#             #
#             # # print("Baseline Loss = " , baseline_loss)
#             # part0, part1, part2, part3, part4, part5 = generator.get_adv_loss(sess, X_one, Y, sentence, rewards, b)
#             # #
#             # #
#             # # print("one hot encoding")
#             # # print(np.array(part0).shape)
#             # # print(np.array(part0)[20:23, :])
#             # # print("logarithm of action probs")
#             # # print(np.array(part1).shape)
#             # # print(np.array(part1)[0:3, :])
#             # # print(np.argmax(np.array(part1)[0:3, :],1))
#             # # print(np.array(part1)[20:23, :])
#             # # print(np.argmax(np.array(part1)[20:23, :],1))
#             # # print("log action multiplied by one hot encoding ")
#             # # print(np.array(part2).shape)
#             # # print(np.array(part2)[626:628,:]) # since word 627 is wrong
#             # # print("reduce sum  ")
#             # # print(np.array(part3).shape)
#             # # print(np.array(part3)[626:628])
#             # # # input("wait")
#             # #
#             # # # c
#             # # # print(part5)
#             # # #
#             # #
#             # #
#             # #
#             # # print(np.array(part4)[20:23])
#             # _, adv_loss = generator.advtrain_step(sess, X_one, Y, sentence, rewards, b)
#             # input("wait")
#             # print("///////////////////////////////")
#             # print("adv loss = " , adv_loss[620:640])
#             # print(np.argmax(adv_loss))
#             # print(adv_loss)
#             # print("Adverserial Loss = " , adv_loss[:20])
#             # print("Adverserial Loss = ", adv_loss[20:40])
#             # print("Adverserial Loss = ", adv_loss[40:60])
#             # print("Adverserial Loss = ", adv_loss[60:80])
#
#             # print("///////////////////////////////AFTER UPDATE/////////////////////////////")
#             # tools.convert_id_to_text(np.array(X_one)[:3, :], word_index)
#             # gen_proba,sentence = generator.generate(sess, X_one, Y)
#             # tools.convert_id_to_text(np.array(sentence)[0:3,:], word_index)
#             # tools.convert_id_to_text(np.array(Y_train)[0:1, :], word_index)
#
#             # gen_proba = np.array(gen_proba)
#
#             # gen_proba = gen_proba[0, :, :]
#             # print(sentence[0, :])
#             # print(gen_proba[np.arange(len(gen_proba)), sentence_old[0, :]])
#
#             # print("///////////////////////////////AFTER UPDATE/////////////////////////////")
#             # input("wait")
#
#             # input("wait")
#
#     print("//////////////////////////////////////////////////////////////////////////////////////////////////")
#
#     if True:
#         total_rew = 0
#         total_wrong = 0
#         proba_list_all_before = []
#         av_true = 0
#         for ii in range(hist_train.shape[0] // headerSeq2Seq.BATCH_SIZE):
#             X = hist_train[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]
#             Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
#             Y_train = reply_train[ii * headerSeq2Seq.BATCH_SIZE:(ii + 1) * headerSeq2Seq.BATCH_SIZE, :]
#
#             gen_proba_test, sentence_test = generator.test_generate(sess, X, Y)
#
#             X_d = toolsSeq2Seq.concat_hist_reply(X, sentence_test, word_index)
#             label_d = np.zeros(headerSeq2Seq.BATCH_SIZE)
#             disc_proba = discriminator.get_rewards(sess, X_d)
#             # # for jj in range(headerSeq2Seq.BATCH_SIZE):
#             # #     # if np.sum(sentence_test[jj] != Y_train[jj])>0:
#             # #         print(sentence_test[jj])
#             # #         toolsSeq2Seq.convert_id_to_text(X[jj:jj+1],word_index)
#             # #         toolsSeq2Seq.convert_id_to_text(sentence_test[jj:jj+1],word_index)
#             # #         toolsSeq2Seq.convert_id_to_text(Y_train[jj:jj+1],word_index)
#             # #         input("wait")
#             # total_correct = total_correct + np.sum(sentence_test == Y_train)
#             # total_wrong = total_wrong + np.sum(sentence_test != Y_train)
#             #
#             # for p_i in range(gen_proba_test.shape[0]):
#             #     temp_proba = gen_proba_test[p_i]
#             #     temp_av = temp_proba[np.arange(len(temp_proba)), Y_train[p_i, :]]
#             #     temp_av[Y_train[p_i, :] == word_index["eos"]] = 0
#             #     proba_list_all_before.append(temp_av)
#             #
#             #     av_proba = np.sum(temp_av) / np.sum(Y_train[p_i, :] != word_index["eos"])
#             #     av_true = av_true + av_proba
#             #     # print(av_proba)
#             #     # input("wait")
#             total_rew = total_rew + np.sum(disc_proba[:,1])
#         if total_rew > max_count:
#             max_count = total_rew
#         print("After Total Reward ===", total_rew," Maximum Reward ======== ", max_count, " Time == ", str(datetime.now().time()))
