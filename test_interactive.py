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
G2.assign_lr(sess,0.0004)
D1.assign_emb(sess, embedding_matrix)
D2.assign_emb(sess, embedding_matrix)

# Pretraining of Discriminator and Generator
train_Disc = False
train_Gen = False
discEp = 1
genEp = 20
#pretrain.pretrain(sess,discriminator,generator,discEp,genEp,train_Disc,train_Gen,savepathD,savepathG)

# Restore pre trained models
#try:
#    D1.restore_model(sess, savepathD)
#    D2.restore_model(sess, savepathD)
#except:
#    print("Disc could not be restored")
#    pass
#
#try:
#    G1.restore_model(sess, savepathG)
#    G2.restore_model(sess, savepathG)
#except:
#    print("Gen could not be restored")
#    pass


d_steps = 1
g_steps = 1
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

    # Train discriminator on 1 epoch
    for d in range(d_steps):
        np.random.shuffle(idxTrain)
        #for j in range(0, hist_train.shape[0] // headerSeq2Seq.BATCH_SIZE):
        for j in range(1):
            X_train = hist_train[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1) *
                headerSeq2Seq.BATCH_SIZE], :]
            Y_train = reply_train[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1)
                * headerSeq2Seq.BATCH_SIZE], :]

            # Generate dialogues
            end_A1 = False # Did A1 end dialogue ? 
            end_A2 = False # Did A2 end dialogue ? 

            history = np.ones(
                    [headerSeq2Seq.BATCH_SIZE, 
                        headerSeq2Seq.MAX_SEQ_LENGTH]) * word_index['eoh']
            Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
            num_round = 0

            while True:
                print("num_round: ", num_round)
                # Agent 1 generates sentence
                print("Agent 1 generating ...")
                prob1, sentence1 = G1.gen_proba_sentence(sess, history, Y)
                history = toolsSeq2Seq.concat_hist_reply(history, sentence1, word_index)
                
                # Check if it is the end of the dialogue
                if end_of_dialogue(sentence1) or num_round == max_num_round:
                    end_A1 = True
                    break

                # Agent 2 generates sentence
                print("Agent 2 generating ...")
                prob2, sentence2 = G2.gen_proba_sentence(sess, history, Y)
                history = toolsSeq2Seq.concat_hist_reply(history, sentence2, word_index)
                
                # Check if it is the end of the dialogue
                if end_of_dialogue(sentence2) or num_round == max_num_round:
                    end_A2 = True
                    break

                num_round+=1
            
            # Build a batch of half true and half false sentences
            label_d = np.ones(headerSeq2Seq.BATCH_SIZE)
            label_d[int(headerSeq2Seq.BATCH_SIZE / 2):] = 0
            X_d = np.zeros([headerSeq2Seq.BATCH_SIZE,  headerSeq2Seq.MAX_SEQ_LENGTH]) * word_index['eoh']
            X_d[0:int(headerSeq2Seq.BATCH_SIZE / 2), :] = X_train[0:int(headerSeq2Seq.BATCH_SIZE / 2), :]
            X_d[int(headerSeq2Seq.BATCH_SIZE / 2):, :] = history[int(headerSeq2Seq.BATCH_SIZE / 2):, :]

            _, d_loss, d_acc, _ = D1.train_step(sess, X_d, label_d)
    
    # Train generator
    for g in range(g_steps): 
        G1.assign_lr(sess, 0.00001)
        G2.assign_lr(sess, 0.00001)
        
        end_A1 = False # Did A1 end dialogue ? 
        end_A2 = False # Did A2 end dialogue ? 

        history = np.ones(
                [headerSeq2Seq.BATCH_SIZE, 
                    headerSeq2Seq.MAX_SEQ_LENGTH]) * word_index['eoh']
        
        Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.REP_SEQ_LENGTH)) * word_index['eos']
        num_round = 0

        # Loss computation data
        r1 = [] # rewards of agent 1
        r2 = [] # rewards of agent 2
        p1 = [] # proba of sentence output by agent 1
        p2 = [] # proba of sentence output by agent 2

        # Generate dialogue and store data for loss computation
        while True:
            # Agent 1 generates sentence
            print("Agent 1 generating ...")
            prob1, sentence1 = G1.gen_proba_sentence(sess, history, Y)
            print("... done.")

            history = toolsSeq2Seq.concat_hist_reply(history, sentence1, word_index)
            p1.append(prob1)
            print("Agent 1 discriminating ...")
            d1_output = D1.get_rewards(sess, history)
            d1_tmp = np.array([item[1] for item in d1_output])
            r1.append(d1_tmp)
            print("... done.")
            
            # Check if it is the end of the dialogue
            if end_of_dialogue(sentence1) or num_round == max_num_round:
                end_A1 = True
                break

            # Agent 2 generates sentence
            print("Agent 2 generating ...")
            prob2, sentence2 = G2.gen_proba_sentence(sess, history, Y)
            print("... done.")
            history = toolsSeq2Seq.concat_hist_reply(history, sentence2, word_index)
            p2.append(prob2)
            print("Agent 2 discriminating ...")
            d2_output = D2.get_rewards(sess, history)
            d2_tmp = np.array([item[1] for item in d2_output])
            r2.append(d2_tmp)
            print("... done.")
            
            # Check if it is the end of the dialogue
            if end_of_dialogue(sentence2) or num_round == max_num_round:
                end_A2 = True
                break

            num_round+=1

        # Compute baseline
        p1 = np.transpose(np.array(p1))
        p2 = np.transpose(np.array(p2))
        r1 = np.transpose(np.array(r1))
        r2 = np.transpose(np.array(r2))
        
        if end_A1:
            print("Agent 1 finished dialogue")
            b1 = np.zeros(r1.shape)
            b1[:,1:] = r2 # b1 = [0,r2]
            b2 = r1[:,:-1] # b2 = r1[0,T-1]
        else:
            print("Agent 2 finished dialogue")
            b1 = np.zeros(r1.shape)
            b1[:,1:] = r2[:,:-1] # b1 = [0,r2[0,T-1]]
            b2 = r1 # b2 = r1
   
        # Debug
        print("p1.shape: ", p1.shape)
        print("r1.shape: ", r1.shape)
        print("b1.shape: ", b1.shape)
        print("p2.shape: ", p2.shape)
        print("r2.shape: ", r2.shape)
        print("b2.shape: ", b2.shape)
        
        if end_A1:
            print("Training G1 ...")
            _, adv_loss = G1.inter_train_step(sess, history, sentence2, sentence2, p1, r1, b1)
            print("... done")
            print("Training G2 ...")
            _, adv_loss = G2.inter_train_step(sess, history, sentence1, sentence1, p2, r2, b2)
            print("... done")
        else:
            print("Training G1 ...")
            _, adv_loss = G1.inter_train_step(sess, history, sentence2, sentence2, p1, r1, b1)
            print("... done")
            print("Training G2 ...")
            _, adv_loss = G2.inter_train_step(sess, history, sentence1, sentence1, p2, r2, b2)
            print("... done")


