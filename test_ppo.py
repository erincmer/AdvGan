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
 train_data,
 test_data,
 word_index) = readFBTask1Seq2Seq.create_con(False, 
         headerSeq2Seq.MAX_SEQ_LENGTH,
         headerSeq2Seq.REP_SEQ_LENGTH)

EMB_DIM = len(word_index) + 1  # embedding dimension
END_TOKEN = word_index.get("eos")
HIST_END_TOKEN = word_index.get("eoh")

# Model
G1 = Generator(EMB_DIM,
        headerSeq2Seq.BATCH_SIZE,
        EMB_DIM,
        headerSeq2Seq.HIDDEN_DIM,
        headerSeq2Seq.MAX_SEQ_LENGTH,
        headerSeq2Seq.REP_SEQ_LENGTH,
        headerSeq2Seq.START_TOKEN,
        END_TOKEN,
        "G1")

D1 = DiscSentence(EMB_DIM,
        headerSeq2Seq.BATCH_SIZE,
        EMB_DIM, 
        headerSeq2Seq.HIDDEN_DIM,
        headerSeq2Seq.MAX_SEQ_LENGTH,                             
        word_index,
        END_TOKEN,
        "D1")
B1 = Baseline(
        headerSeq2Seq.BATCH_SIZE,
        headerSeq2Seq.HIDDEN_DIM,
        headerSeq2Seq.REP_SEQ_LENGTH,
        headerSeq2Seq.MAX_SEQ_LENGTH,
        word_index,
        "B1",
        learning_rate=0.0004)

G1_old = Generator(EMB_DIM,
        headerSeq2Seq.BATCH_SIZE,
        EMB_DIM,
        headerSeq2Seq.HIDDEN_DIM,
        headerSeq2Seq.MAX_SEQ_LENGTH,
        headerSeq2Seq.REP_SEQ_LENGTH,
        headerSeq2Seq.START_TOKEN,
        END_TOKEN,
        "oldG1")


# TF setting
config = tf.ConfigProto()
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
G1.assign_emb(sess, embedding_matrix)
G1.assign_lr(sess,0.0004)
D1.assign_emb(sess, embedding_matrix)
G1_old.assign_emb(sess, embedding_matrix)
G1_old.assign_lr(sess,0.0004)

hist_train = train_data["hist_s1"]
reply_train = train_data["reply_s1"]
idxTrain = np.arange(len(hist_train))
train_Disc = False
train_Gen = False
discEp = 1
genEp = 20

# Restore pre trained models
#try:
#    D1.restore_model(sess, savepathD)
#except:
#    print("Disc could not be restored")
#    pass
#try:
#    G1.restore_model(sess, savepathG)
#except:
#    print("Gen could not be restored")
#    pass
G1_old.copy(sess, G1)

d_steps = 0
g_steps = 1
idxTrain = np.arange(len(hist_train))
# Adversarial steps
teacher_forcing = False
max_avg_prob = np.zeros((10,), dtype=np.float)
ind = -1
hist_batch = 8
max_count = 0

oldpi_update_step = 5

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
            _, sentence = G1.generate(sess, X, Y)

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

            _, d_loss, d_acc, _ = D1.train_step(sess, X_d, label_d)
    
    # Train generator
    G1.assign_lr(sess, 0.00001)
    update_step = 0
    for g in range(g_steps):
        #for j in range(0, hist_train.shape[0] // headerSeq2Seq.BATCH_SIZE):
        for j in range(10):
            print("\n************************")
            print("Episode ", ep)
            print("Step: ", j)
            j = 0 # TODO: remove it when training Generator
            X = hist_train[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE], :]
            Y_train = reply_train[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE], :]
            Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
            
            # Generate sentence
            gen_proba,sentence = G1.generate(sess, X, Y)

            # TODO concat but
            #disc_in = toolsSeq2Seq.concat_hist_reply(X, sentence, word_index)
            #disc_rewards = D1.get_rewards(sess, disc_in)
            
            disc_rewards = D1.get_rewards(sess, X)
            rewards = np.ones([headerSeq2Seq.BATCH_SIZE,
                headerSeq2Seq.REP_SEQ_LENGTH])
            #rewards = G1.MC_reward(sess, X, sentence, headerSeq2Seq.MC_NUM, D1, word_index)

            #b_loss = B1.train(sess, X, sentence, word_index, rewards)
            #b = B1.get_baseline(sess, X, sentence, word_index)
            b = np.ones([headerSeq2Seq.BATCH_SIZE,
                headerSeq2Seq.REP_SEQ_LENGTH])

            print("baseline baby !: ", b[0,:])
            
            # Use sentence to feed decoder
            distro_old, _ = G1_old.generate(sess, X, sentence)
            
            # Mask on 'eos' words in generated sentence
            mask = np.ones([headerSeq2Seq.BATCH_SIZE,
                headerSeq2Seq.REP_SEQ_LENGTH])
            mask[:,0] = (sentence[:,0] !=
                    word_index['eos']).astype(np.float32)
            for t in range(1,headerSeq2Seq.REP_SEQ_LENGTH): 
                mask[:,t] = (sentence[:,t-1] !=
                        word_index['eos']).astype(np.float32)

            if update_step == oldpi_update_step:
                print("Copy current policy into old policy before update")
                G1_old.copy(sess, G1)
                update_step = 0

            _, adv_loss = G1.ppo_step(sess, X, sentence, sentence, rewards, distro_old, mask)
            update_step +=1




