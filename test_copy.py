import numpy as np
import tensorflow as tf

import readFBTask1Seq2Seq
from GeneratorModel import Generator
from Disc1 import DiscSentence
from Baseline import Baseline
import headerSeq2Seq
import toolsSeq2Seq
import pretrain

def end_of_dialogue(sentence):
    return False

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

# Agent 1 
G1 = Generator(
        EMB_DIM,
        headerSeq2Seq.BATCH_SIZE,
        EMB_DIM,
        headerSeq2Seq.HIDDEN_DIM,
        headerSeq2Seq.MAX_SEQ_LENGTH,
        headerSeq2Seq.REP_SEQ_LENGTH,
        headerSeq2Seq.START_TOKEN,
        END_TOKEN,
        "G1")
D1 = DiscSentence(
        EMB_DIM,
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


# Agent 2
G2 = Generator(
        EMB_DIM,
        headerSeq2Seq.BATCH_SIZE,
        EMB_DIM,
        headerSeq2Seq.HIDDEN_DIM,
        headerSeq2Seq.MAX_SEQ_LENGTH,
        headerSeq2Seq.REP_SEQ_LENGTH,
        headerSeq2Seq.START_TOKEN,
        END_TOKEN,
        "G2")
D2 = DiscSentence(
        EMB_DIM,
        headerSeq2Seq.BATCH_SIZE,
        EMB_DIM, 
        headerSeq2Seq.HIDDEN_DIM,
        headerSeq2Seq.MAX_SEQ_LENGTH,
        word_index,
        END_TOKEN,
        "D2")
B2 = Baseline(
        headerSeq2Seq.BATCH_SIZE,
        headerSeq2Seq.HIDDEN_DIM,
        headerSeq2Seq.REP_SEQ_LENGTH,
        headerSeq2Seq.MAX_SEQ_LENGTH,
        word_index,
        "B2",
        learning_rate=0.0004)

# TF setting
config = tf.ConfigProto()
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
G1.assign_emb(sess, embedding_matrix)
G1.assign_lr(sess,0.0004)
G2.assign_emb(sess, embedding_matrix)
G2.assign_lr(sess,1.0)
D1.assign_emb(sess, embedding_matrix)
D2.assign_emb(sess, embedding_matrix)


hist_train = train_data["hist_s1"]
reply_train = train_data["reply_s1"]
idxTrain = np.arange(len(hist_train))
# Adversarial steps
teacher_forcing = False
max_avg_prob = np.zeros((10,), dtype=np.float)
ind = -1
hist_batch = 8
max_count = 0

num_ep = 1
max_num_round = 5
for ep in range(num_ep):
    for j in range(1):
        print("\n************************")
        print("Episode ", ep)
        print("Step: ", j)
        j = 0 # TODO: remove it when training Generator

        X = hist_train[idxTrain[j*headerSeq2Seq.BATCH_SIZE:(j+1)*headerSeq2Seq.BATCH_SIZE],:]
        Y_train = reply_train[idxTrain[j*headerSeq2Seq.BATCH_SIZE:(j+1)*headerSeq2Seq.BATCH_SIZE],:]
        Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
        
        # Check eq before any change
        gen_G1_old,sen_G1_old = G1.test_generate(sess, X, Y)       
        gen_G2_old,sen_G2_old = G2.test_generate(sess, X, Y)       
        G1_G2_init = np.sum(gen_G1_old == gen_G2_old)
        print("At initialization: G1 == G2: ", G2.test_eq(sess, G1))
        print("At initialization: num of common parameters: ", G1_G2_init)

        # Generate sentence
        gen_proba,sentence = G1.generate(sess, X, Y)       
        #disc_in = toolsSeq2Seq.concat_hist_reply(X, sentence, word_index)
        disc_in = X
        disc_rewards = D1.get_rewards(sess, disc_in)
        #rewards = G1.MC_reward(sess, X, sentence, headerSeq2Seq.MC_NUM, D1, word_index)
        rewards = np.ones([headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH])
        b = np.ones([headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH])
        #b = B1.get_baseline(sess, X, sentence, word_index)
        print("baseline baby !: ", b[0,:])
        toolsSeq2Seq.convert_id_to_text(np.array(sentence)[0:1],word_index)
        
        _, adv_loss = G1.advtrain_step(sess, X, sentence, sentence, rewards, b)
        #_, adv_loss = generator.ppo_step(sess, X_one, Y, sentence, rewards, old_distro)
        
        
        print("Before copy: G1 == G2: ", G2.test_eq(sess, G1))
        gen_G1_old,sen_G1_old = G1.test_generate(sess, X, Y)       
        gen_G2_old,sen_G2_old = G2.test_generate(sess, X, Y)       
        G1_change = np.sum(gen_G1_old == gen_G2_old)
        print("Before copy: num of common proba: ", G1_change)

        ## Copy G1 
        G2.copy(sess, G1)
        print("After copy: G1 == G2: ", G2.test_eq(sess, G1))
        gen_G1_old_copy,sen_G1_old_copy = G1.test_generate(sess, X, Y)       
        gen_G2_old,sen_G2_old = G2.test_generate(sess, X, Y)       
        G2_copy = np.sum(gen_G1_old_copy == gen_G2_old)
        print("After copy: num of common proba: ", G2_copy)

        # Change G1
        _, adv_loss = G1.advtrain_step(sess, X, sentence, sentence, rewards, b)
        print("After G1 update: G1 == G2: ", G2.test_eq(sess, G1))
        gen_G1_old,sen_G1_old = G1.test_generate(sess, X, Y)       
        gen_G2_old,sen_G2_old = G2.test_generate(sess, X, Y)       
        G1_update = np.sum(gen_G1_old == gen_G2_old)
        G1_check = np.sum(gen_G1_old == gen_G1_old_copy)
        print("After G1 update: num of common proba: ", G1_update)
        print("After G1 update: G1 changes: ", G1_check)


