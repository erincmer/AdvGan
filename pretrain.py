
import os
import numpy as np

from tensorflow.contrib.legacy_seq2seq import embedding_rnn_decoder
from tensorflow.contrib.legacy_seq2seq import rnn_decoder
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.contrib.rnn.python.ops import rnn_cell
import tensorflow as tf

import readFBTask1Seq2Seq
from Generator import Generator
from Disc1 import DiscSentence
from Baseline import Baseline
import header
import tools


def pretrain(savepathD, savepathG):

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


    if not os.path.exists(savepathG):
        os.makedirs(savepathG)
    if not os.path.exists(savepathD):
        os.makedirs(savepathD)

    # pre train disc
    idxTrain = np.arange(len(x_train))
    idxTest = np.arange(len(x_test))
    for ep in range(1):
        np.random.shuffle(idxTrain)
        #for j in range(0, x_train.shape[0] // header.BATCH_SIZE):
        for j in range(0, 1):
            # print("*********************************")
            X = x_train[idxTrain[j*header.BATCH_SIZE:(j+1)*header.BATCH_SIZE],:]
            Y_train = y_train[idxTrain[j*header.BATCH_SIZE:(j+1)*header.BATCH_SIZE]]
            _,d_loss,d_acc,_ = discriminator.train_step(sess,X,Y_train)
    
            if j %50==0:
                print("Disc Train Loss = ", d_loss, " Accuracy = ",d_acc)
                np.random.shuffle(idxTest)
                X = x_test[idxTest[:header.BATCH_SIZE],:]
                Y_train = y_test[idxTest[:header.BATCH_SIZE]]
                d_loss,d_acc= discriminator.get_loss(sess, X, Y_train)
                print("Disc Test Loss = ", d_loss, " Accuracy = ",d_acc)
    discriminator.save_model(sess, savepathD)
    
    # Pre train gen
    idxTrain = np.arange(len(hist_train))
    idxTest = np.arange(len(hist_test))
    max_avg_prob = np.zeros((10,),dtype=np.float)
    T = 1
    for ep in range(T):
        np.random.shuffle(idxTrain)
        #for j in range(hist_train.shape[0] // header.BATCH_SIZE):
        for j in range(1):
            X = hist_train[idxTrain[j * header.BATCH_SIZE:(j + 1) * header.BATCH_SIZE], :]
            Y_train = reply_train[idxTrain[j * header.BATCH_SIZE:(j + 1) * header.BATCH_SIZE], :]
            # tools.convert_id_to_text(np.array(X)[0:3, :], word_index)
            # tools.convert_id_to_text(np.array(Y_train)[0:3, :], word_index)
            # input("wait")
    
            _, g_loss, _ = generator.pretrain_step(sess, X, Y_train)
            if j == (hist_train.shape[0] // header.BATCH_SIZE - 1):
                print("Gen Train Loss = ", g_loss, ep)
                X = hist_test[idxTest[: header.BATCH_SIZE], :]
                Y_real = reply_test[idxTest[: header.BATCH_SIZE], :]
                Y_real = Y_train
                Y_test = np.ones((header.BATCH_SIZE, REP_SEQ_LENGTH)) * word_index['eos']
    
                g_loss = generator.get_pretrain_loss(sess, X, Y_test)
                print("Gen Test Loss = ", g_loss, ep)
    
                _, sentence = generator.test_generate(sess, X, Y_test)
                sentence[sentence == 0] = word_index['eos']
                print("Generator Predicted Sentences")
                tools.convert_id_to_text(np.array(sentence)[0:5], word_index)
                print("Real Sentences")
                tools.convert_id_to_text(np.array(Y_real)[0:5], word_index)
    
    generator.save_model(sess, savepathG)

    sess.close()

if __name__=='__main__':
    savepathG = 'GeneratorModel/'  # best is saved here
    savepathD = 'DiscModel/'
    pretrain(savepathD, savepathG)
