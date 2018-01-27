
import numpy as np
import os

import tensorflow as tf

import readFBTask1Seq2Seq

import headerSeq2Seq as header
import toolsSeq2Seq  as tools


def pretrain(sess,discriminator,generator,discEp,genEp,trainDisc,trainGen,savepathD, savepathG):
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
     word_index) = readFBTask1Seq2Seq.create_con(False, header.MAX_SEQ_LENGTH)


    if not os.path.exists(savepathD):
        os.makedirs(savepathD)
    if not os.path.exists(savepathG):
            os.makedirs(savepathG)


    # Model

    # pre train disc
    if trainDisc:
        idxTrain = np.arange(len(x_train))
        idxTest = np.arange(len(x_test))
        for ep in range(discEp):
            np.random.shuffle(idxTrain)
            for j in range(0, x_train.shape[0] // header.BATCH_SIZE):

                # print("*********************************")
                X = x_train[idxTrain[j * header.BATCH_SIZE:(j + 1) * header.BATCH_SIZE], :]
                Y_train = y_train[idxTrain[j * header.BATCH_SIZE:(j + 1) * header.BATCH_SIZE]]
                _, d_loss, d_acc, _ = discriminator.train_step(sess, X, Y_train)

                if j % 100 == 0:
                    print("Epoch == ", ep, " Minibatch = ", j, ", d_loss, ", "Accuracy = ", d_acc)
                    np.random.shuffle(idxTest)
                    X = x_test[idxTest[:header.BATCH_SIZE], :]
                    Y_train = y_test[idxTest[:header.BATCH_SIZE]]
                    d_loss, d_acc = discriminator.get_loss(sess, X, Y_train)
                    print("Disc Test Loss = ", d_loss, " Accuracy = ", d_acc)
                    discriminator.save_model(sess, savepathD)

    if trainGen:
        # Pre train gen
        idxTrain = np.arange(len(hist_train))
        idxTest = np.arange(len(hist_test))


        for ep in range(genEp):
            np.random.shuffle(idxTrain)
            for j in range(hist_train.shape[0] // header.BATCH_SIZE):
             # j in range(1):
                X = hist_train[idxTrain[j * header.BATCH_SIZE:(j + 1) * header.BATCH_SIZE], :]
                Y_train = reply_train[idxTrain[j * header.BATCH_SIZE:(j + 1) * header.BATCH_SIZE], :]
                # tools.convert_id_to_text(np.array(X)[0:3, :], word_index)
                # tools.convert_id_to_text(np.array(Y_train)[0:3, :], word_index)
                # input("wait")

                _, g_loss, _ = generator.pretrain_step(sess, X, Y_train)
                if j % 100 == 0:
                    print("Gen Train Loss = ", g_loss, ep)
                    X = hist_test[idxTest[: header.BATCH_SIZE], :]
                    Y_real = reply_test[idxTest[: header.BATCH_SIZE], :]
                    # Y_real = Y_train
                    Y_test = np.ones((header.BATCH_SIZE, header.REP_SEQ_LENGTH)) * word_index['eos']

                    g_loss = generator.get_pretrain_loss(sess, X, Y_test)
                    print("Gen Test Loss = ", g_loss, ep)

                    _, sentence = generator.test_generate(sess, X, Y_test)
                    sentence[sentence == 0] = word_index['eos']
                    print("Generator Predicted Sentences")
                    tools.convert_id_to_text(np.array(sentence)[0:5], word_index)
                    print("Real Sentences")
                    tools.convert_id_to_text(np.array(Y_real)[0:5], word_index)

                    generator.save_model(sess, savepathG)




if __name__ == '__main__':
    savepathG = 'GeneratorModel/'  # best is saved here
    savepathD = 'DiscModel/'
    pretrain(savepathD, savepathG)