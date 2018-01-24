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


def save_to_file(filename, ids,word_index):
    f = open(filename, 'w+')
    for id in ids:
        sen = ""
        for i in id:
            if i!=0  and i!= word_index["eos"] and i!= word_index["eoh"]:
                sen = sen +" " +list(word_index.keys())[list(word_index.values()).index(i)]
        f.write(sen + '\n')
        #print(sen)


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
embedding_matrix,hist_train,hist_test,reply_train,reply_test,x_train,x_test,y_train,y_test,word_index = readFBTask1Seq2Seq.create_con(False,MAX_SEQUENCE_LENGTH)

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



idxTrain = np.arange(len(hist_train))
idxTest = np.arange(len(hist_test))
for ep in range(100):
    np.random.shuffle(idxTrain)

    for j in range(0, hist_train.shape[0] // BATCH_SIZE):
        print("\n************************")
        print("Episode ", ep)
        print("Step: ", j)
        j = 0 # TODO: remove it when training Generator

        X = hist_train[idxTrain[j*BATCH_SIZE:(j+1)*BATCH_SIZE],:]
        Y_train = reply_train[idxTrain[j*BATCH_SIZE:(j+1)*BATCH_SIZE],:]
        Y = np.ones((BATCH_SIZE, REP_SEQ_LENGTH)) * word_index['eos']
        
        # Generate sentence
        gen_proba,sentence = generator.generate(sess, X, Y)

        # debug
        print("\nInput history:")
        convert_id_to_text(np.array(X)[0:3, :], word_index)
        print("\nExpected reply:")
        convert_id_to_text(np.array(Y_train)[0:3, :], word_index)
        save_to_file("./log/expected_reply_%d_%d.txt" %(ep, j), np.array(Y_train),word_index)
        print("\nOutput reply:")
        convert_id_to_text(np.array(sentence)[0:3,:], word_index)
        save_to_file("./log/output_reply_%d_%d.txt" %(ep, j), np.array(sentence),word_index)
        
        # Save sentence 
        #sentence_old = np.copy(np.array(sentence))
        #gen_proba = np.array(gen_proba)
        #gen_proba = gen_proba[31, :, :]

        # Compute reward for complete sentence
        disc_in = concat_hist_reply(X,sentence,word_index)
        disc_rewards = discriminator.get_rewards(sess,disc_in)
        print("\nRewards for output reply: ", disc_rewards[0:3,1])
        
        # Compute MC rewards
        rewards = generator.MC_reward(sess, X, sentence, MC_NUM, discriminator,word_index)
        print("\nMC Rewards: ")  # depend
        print(np.array(rewards)[0:3])
        
        # Update baseline
        baseline_loss = baseline.train(X, sentence, rewards, word_index)
        b = baseline.get_baseline(X,sentence,word_index)

        print("\nBaseline Rewards")  # depend
        print(np.array(b)[0:3])
        print("Baseline Loss = " , baseline_loss)
        #part0,part1,part2,part3,part4,part5 = generator.get_adv_loss(sess, X_one, Y, sentence, rewards, b)
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
        #print("log action multiplied by one hot encoding ")
        #print(np.array(part2).shape)
        #print(np.array(part2)[626:628,:]) # since word 627 is wrong
        #print("reduce sum  ")
        #print(np.array(part3).shape)
        #print(np.array(part3)[626:628])
        # # input("wait")
        # # c
        # # print(part5)
        # #
        # print(np.array(part4)[20:23])
        #_,adv_loss =generator.advtrain_step(sess, X_one, Y, sentence, rewards, b)
        
        # Adv training
        outputs =generator.advtrain_step_debug(sess, X, sentence, sentence, rewards, b, word_index,gen_proba)
        new_sentence = outputs[1] 

        print("\nOutput reply from adversarial update:")
        convert_id_to_text(np.array(new_sentence)[0:3,:], word_index)
        save_to_file("./log/output_reply_adv_%d_%d.txt" %(ep, j), np.array(new_sentence), word_index)
        
        # print("adv loss = " , adv_loss[620:640])
        # print(np.argmax(adv_loss))
        # print("Adverserial Loss = " , adv_loss[:20])
        # print("Adverserial Loss = ", adv_loss[20:40])
        # print("Adverserial Loss = ", adv_loss[40:60])
        # print("Adverserial Loss = ", adv_loss[60:80])
        
        # Check that sentence has improved after update
        #convert_id_to_text(np.array(X_one)[30:33, :], word_index)
        gen_proba,sentence = generator.generate(sess, X, Y)
        print("\nOutput reply after update:")
        convert_id_to_text(np.array(sentence)[0:3,:], word_index)
        save_to_file("./log/output_reply_update_%d_%d.txt" %(ep, j), np.array(sentence), word_index)

        input("wait")
        exit(0)
