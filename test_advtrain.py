import numpy as np


import tensorflow as tf


import readFBTask1Seq2Seq
from GeneratorModel import Generator
from Disc1 import DiscSentence
from Baseline import Baseline
import headerSeq2Seq
import toolsSeq2Seq
import pretrain

np.set_printoptions(precision=5, suppress=True, linewidth=250)

# Pre train if necessary
savepathG = 'GeneratorModel/'  # best is saved here
savepathD = 'DiscModel/'

# if ~header.DO_RESTORE:
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
 word_index) = readFBTask1Seq2Seq.create_con(False, headerSeq2Seq.MAX_SEQ_LENGTH)

EMB_DIM = len(word_index) + 1  # embedding dimension
END_TOKEN = word_index.get("eos")
HIST_END_TOKEN = word_index.get("eoh")

# Model
generator = Generator(EMB_DIM,
                      headerSeq2Seq.BATCH_SIZE,
                      EMB_DIM,
                      headerSeq2Seq.HIDDEN_DIM,
                      headerSeq2Seq.MAX_SEQ_LENGTH,
                      headerSeq2Seq.REP_SEQ_LENGTH,
                      headerSeq2Seq.START_TOKEN,
                      END_TOKEN,
                      HIST_END_TOKEN)
discriminator = DiscSentence(EMB_DIM,
                             headerSeq2Seq.BATCH_SIZE,
                             EMB_DIM, headerSeq2Seq.HIDDEN_DIM,
                             headerSeq2Seq.MAX_SEQ_LENGTH,
                             word_index,
                             END_TOKEN)
baseline = Baseline(headerSeq2Seq.BATCH_SIZE,
                    headerSeq2Seq.HIDDEN_DIM,
                    headerSeq2Seq.REP_SEQ_LENGTH,
                    headerSeq2Seq.MAX_SEQ_LENGTH,
                    word_index,
                    learning_rate=0.0004)

# TF setting
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
generator.assign_emb(sess, embedding_matrix)
discriminator.assign_emb(sess, embedding_matrix)




train_Disc = False
train_Gen = False
discEp = 1
genEp = 20
# Restore pre trained models



#Pretraining of Discriminator and Generator
generator.assign_lr(sess,0.0004)



try:
    discriminator.restore_model(sess, savepathD)
except:
    print("Disc could not be restored")
    pass
try:
    generator.restore_model(sess, savepathG)
except:
    print("Gen could not be restored")
    pass
#pretrain.pretrain(sess,discriminator,generator,discEp,genEp,train_Disc,train_Gen,savepathD,savepathG)

d_steps = 0
g_steps = 1
idxTrain = np.arange(len(hist_train))
idxTest = np.arange(len(hist_test))
# Adversarial steps
teacher_forcing = False
max_avg_prob = np.zeros((10,), dtype=np.float)
ind = -1
hist_batch = 8
max_count = 0
for ep in range(0, 1000000):
    # Train discriminator

    for d in range(d_steps):
        print("D step ===",d)
        np.random.shuffle(idxTrain)
        for j in range(0, hist_train.shape[0] // headerSeq2Seq.BATCH_SIZE):
            X = hist_train[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE], :]
            Y_train = reply_train[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE], :]

            # Generate sentence
            if not teacher_forcing:
                Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
                teacher_forcing = False
            else:
                # Teacher forcing
                Y = Y_train
                teacher_forcing = False
            _, sentence = generator.generate(sess, X, Y)

            # Pad sentence of variable length to header.REP_SEQ_LENGTH
            # rep_inp = np.full((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH), word_index['eos'])
            # rep_inp[:, :sentence.shape[1]] = sentence
            # sentence = rep_inp
            # sentence[sentence == 0] = word_index['eos']

            # Build a batch of half true and half false sentences
            Y_d = np.zeros((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH))
            sentence_index = np.random.random_integers((headerSeq2Seq.BATCH_SIZE - 1), size=(int(headerSeq2Seq.BATCH_SIZE / 2), 1))

            Y_d[0:int(headerSeq2Seq.BATCH_SIZE / 2), :] = np.squeeze(Y_train[sentence_index, :])
            Y_d[int(headerSeq2Seq.BATCH_SIZE / 2):, :] = np.squeeze(sentence[sentence_index, :])
            X_d = np.copy(X)
            X_d[0:int(headerSeq2Seq.BATCH_SIZE / 2), :] =  np.squeeze(np.copy(X[sentence_index, :]))
            X_d[int(headerSeq2Seq.BATCH_SIZE / 2):, :] =  np.squeeze(np.copy(X[sentence_index, :]))

            X_d = toolsSeq2Seq.concat_hist_reply(X_d, Y_d, word_index)
            label_d = np.ones(headerSeq2Seq.BATCH_SIZE)
            label_d[int(headerSeq2Seq.BATCH_SIZE / 2):] = 0

            # print(np.sum(np.abs(X_d[0:int(headerSeq2Seq.BATCH_SIZE / 2), :] !=X_d[int(headerSeq2Seq.BATCH_SIZE / 2):, :])))
            # toolsSeq2Seq.convert_id_to_text(X_d,word_index)
            # toolsSeq2Seq.convert_id_to_text(X_d,word_index)
            _, d_loss, d_acc, _ = discriminator.train_step(sess, X_d, label_d)
            # print("Discriminator loss = ", d_loss)
            # input("wait")
    # Train generator
    generator.assign_lr(sess, 0.00001)
    for g in range(g_steps):
        print("G step == ", g, " ind== ", ind)
        ind = ind + 1
        ind = ind % len(hist_train)
        # toolsSeq2Seq.convert_id_to_text(reply_train[ind:ind+1, :],word_index)
        print("////////////////////////////////////////////////////////////")
        # rewards = np.zeros(((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)))
        # b = np.zeros(((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)))
        # for ii in range(int(headerSeq2Seq.BATCH_SIZE//hist_batch)):

        X_one = np.tile(np.copy(hist_train[ind, :]), (headerSeq2Seq.BATCH_SIZE, 1))
        Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
        gen_proba, sentence = generator.generate(sess, X_one, Y)
        disc_in = toolsSeq2Seq.concat_hist_reply(X_one, sentence, word_index)
        disc_rewards = discriminator.get_rewards(sess, disc_in)

        rewards = generator.MC_reward(sess, X_one, sentence, headerSeq2Seq.MC_NUM, discriminator, word_index)

        b_loss = baseline.train(sess, X_one, sentence, word_index, rewards)
        b = baseline.get_baseline(sess, X_one, sentence, word_index)
        print("baseline baby !: ", b)
        print("baseline loss: ", b_loss)
        #b = np.tile(np.mean(np.array(rewards), axis=0), (headerSeq2Seq.BATCH_SIZE, 1))
        # toolsSeq2Seq.convert_id_reward_to_text(np.array(sentence)[0:10],disc_rewards[:10,1],rewards[:10],b[:10],word_index)







        _, adv_loss = generator.advtrain_step(sess, X_one, Y, sentence, rewards, b)










        # Y_train = reply_train[ind: headerSeq2Seq.BATCH_SIZE + ind, :]
        # print("ind ==== ", ind)
        #
        # # print("*********************************")
        #
        #
        # Y_one = np.tile(np.copy(Y_train[0, :]), (headerSeq2Seq.BATCH_SIZE, 1))
        # toolsSeq2Seq.convert_id_to_text(np.array(X_one)[0:3, :], word_index)
        #
        #
        # gen_proba_test, sentence_test = generator.test_generate(sess, X_one, Y)
        # gen_proba_test_one = np.copy(gen_proba_test[0, :, :])
        # print("Target words probabilities")
        # print(gen_proba_test_one[np.arange(len(gen_proba_test_one)), Y_one[0, :]])
        #
        # print("Created Sentence with argmax")
        # toolsSeq2Seq.convert_id_to_text(np.array(sentence_test)[0:1, :], word_index)
        # print("Target Sentences")
        # toolsSeq2Seq.convert_id_to_text(np.array(Y_one)[0:1, :], word_index)
        # # print(Y_train[0, :])
        # print("Number of Correct Words  =============================", np.sum(sentence_test == Y_one))
        #
        # temp_av = gen_proba_test_one[np.arange(len(gen_proba_test_one)), Y_one[0, :]]
        #
        # temp_av[Y_train[0, :] == word_index["eos"]] = 0
        # av_proba = np.sum(temp_av) / np.sum(Y_one[0, :] != word_index["eos"])
        # print("Average Prob for Chosing Right Action  =============================", av_proba)
        #
        # if av_proba > max_avg_prob[ind]:
        #     print(
        #         "**********************************************MAX FOUND**************************************************************************")
        #     max_avg_prob[ind] = av_proba
        # print(max_avg_prob)
        # # print(np.array(gen_proba).shape)
        #
        # # print("FULL SAMPLED SENTENCES")
        # # tools.convert_id_to_text(np.array(sentence)[:5,:], word_index)
        # sentence_old = np.copy(np.array(sentence))
        # # tools.convert_id_to_text(np.array(Y_train)[0:1, :], word_index)
        #
        # gen_proba = np.array(gen_proba)
        #
        # gen_proba = gen_proba[0, :, :]
        # # print("SENTENCES WITH CHOSING WORD PROBABILITIES")
        # # tools.convert_id_to_text(np.array(sentence_test)[0:1, :], word_index)
        # # print(gen_proba[np.arange(len(gen_proba)), sentence[0, :]])
        # # input("wait 1")
        # # print(gen_proba[0,1, sentence[0,1]])
        #
        #
        # # X_one =np.tile(np.copy(X[0,:]),(header.BATCH_SIZE,1))
        #
        #
        # # Sen_one = np.tile(np.copy(sentence[0, :]), (header.BATCH_SIZE, 1))
        # #
        # # sentence = Sen_one
        # #
        # #
        # # rep_inp = np.full((header.BATCH_SIZE, header.REP_SEQ_LENGTH), word_index['eos'])
        # # rep_inp[:, :sentence.shape[1]] = sentence
        # # sentence = rep_inp
        # # sentence[sentence ==0] = word_index['eos']
        #
        #
        #
        # sen_rand = np.random.random_integers(len(word_index), size=(headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH))
        # stop_index = np.random.random_integers(headerSeq2Seq.REP_SEQ_LENGTH, size=(headerSeq2Seq.BATCH_SIZE, 1))
        # for i in range(headerSeq2Seq.BATCH_SIZE):
        #     sen_rand[i, stop_index[i][0]:] = word_index["eos"]
        #
        # # disc_in_rand = toolsSeq2Seq.concat_hist_reply(X, sen_rand, word_index)
        # # disc_in_real = toolsSeq2Seq.concat_hist_reply(X, Y_train, word_index)
        # # disc_rewards_rand = discriminator.get_rewards(sess, disc_in_rand)
        # # disc_rewards_real = discriminator.get_rewards(disc_in_real)
        # # disc_rewards_real = discriminator.get_rewards(sess, disc_in_real)
        # print("///////////////////////////////")
        # print("Discriminator Rewards for MC Sentences = ", disc_rewards[0:3, 1])
        # print("Discriminator Rewards for Random Sentences = ", disc_rewards_rand[0:3, 1])
        # print("Discriminator Rewards for Real Sentences = ", disc_rewards_real[0:3, 1])
        # print("///////////////////////////////")
        # print("MC sample ids for first 3 Sentence")  # depend
        # print(np.array(sentence)[0:3])
        # toolsSeq2Seq.convert_id_to_text(np.array(sentence)[0:3], word_index)
        # # print("CORRECT SENTENCE")
        #
        #
        # rewards = generator.MC_reward(sess, X_one, sentence, headerSeq2Seq.MC_NUM, discriminator, word_index)
        #
        # print("///////////////////////////////")
        # print("MC Rewards for first 3 Sentence")  # depend
        # print(np.array(rewards)[0:3])
        # # input("wait 3 ")
        #
        # # baseline_loss = baseline.train(X_one, sentence, rewards, word_index)
        # b = np.tile(np.mean(np.array(rewards), axis=0), (headerSeq2Seq.BATCH_SIZE, 1))
        # # b = baseline.get_baseline(X_one,sentence,word_index)
        #
        # print("///////////////////////////////")
        # print("Baseline Rewards for first 3 Sentence")  # depend
        # print(np.array(b)[30:33])
        #
        # # print("Baseline Loss = " , baseline_loss)
        # part0, part1, part2, part3, part4, part5 = generator.get_adv_loss(sess, X_one, Y, sentence, rewards, b)
        # #
        # #
        # # print("one hot encoding")
        # # print(np.array(part0).shape)
        # # print(np.array(part0)[20:23, :])
        # # print("logarithm of action probs")
        # # print(np.array(part1).shape)
        # # print(np.array(part1)[0:3, :])
        # # print(np.argmax(np.array(part1)[0:3, :],1))
        # # print(np.array(part1)[20:23, :])
        # # print(np.argmax(np.array(part1)[20:23, :],1))
        # # print("log action multiplied by one hot encoding ")
        # # print(np.array(part2).shape)
        # # print(np.array(part2)[626:628,:]) # since word 627 is wrong
        # # print("reduce sum  ")
        # # print(np.array(part3).shape)
        # # print(np.array(part3)[626:628])
        # # # input("wait")
        # #
        # # # c
        # # # print(part5)
        # # #
        # #
        # #
        # #
        # # print(np.array(part4)[20:23])
        # _, adv_loss = generator.advtrain_step(sess, X_one, Y, sentence, rewards, b)
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

        if g%25 ==0:
            total_correct = 0
            for j in range(hist_train.shape[0] // headerSeq2Seq.BATCH_SIZE):
                    X = hist_train[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE, :]
                    Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
                    Y_train = reply_train[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE, :]

                    gen_proba_test, sentence_test = generator.test_generate(sess, X, Y)

                    total_correct = total_correct + np.sum(sentence_test == Y_train)
            if total_correct > max_count:
                max_count = total_correct
            print("Total Correct Count ======== ", total_correct ," Maximum Correct Count ======== ", max_count)
