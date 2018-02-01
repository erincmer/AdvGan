import numpy as np
import tensorflow as tf

import readFBTask1Seq2Seq
import headerSeq2Seq
import toolsSeq2Seq

np.set_printoptions(precision=5, suppress=True, linewidth=250)

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

# TF setting
config = tf.ConfigProto()
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


hist_train = train_data["hist_s1"]
reply_train = train_data["reply_s1"]
idxTrain = np.arange(len(hist_train))

TERM_TOKEN = word_index.get("paris")
TERM_STRING = "paris"

for _ in range(1):
    np.random.shuffle(idxTrain)
    #for j in range(0, hist_train.shape[0] // headerSeq2Seq.BATCH_SIZE):
    for j in range(1):
        X = hist_train[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE], :]
        Y_train = reply_train[idxTrain[j * headerSeq2Seq.BATCH_SIZE:(j + 1) * headerSeq2Seq.BATCH_SIZE], :]

        # TEST EOD
        over_lines_test = np.zeros(headerSeq2Seq.BATCH_SIZE)
        
        # Simulate that over_ind_test lines are terminating sentences
        over_ind_test =np.random.random_integers((
            headerSeq2Seq.BATCH_SIZE - 1),
            size=(10, 1))
        over_lines_test[over_ind_test] = 1

        for i in over_ind_test:
            Y_train[i,0] = word_index.get("paris")

        # Compute over_ind
        over_lines = np.zeros(headerSeq2Seq.BATCH_SIZE)
        over_lines = toolsSeq2Seq.EOD(Y_train, over_lines, word_index,"paris")
        over_ind = np.arange(headerSeq2Seq.BATCH_SIZE)[(over_lines==1)]
        
        print("over_ind_test: ", over_ind_test)
        print("over_ind: ", over_ind)
        print("\nover_lines_test:\nover_lintes:\n", )
        print(over_lines_test)
        print(over_lines)

        for i in over_ind:
            print(i)
            toolsSeq2Seq.convert_sentence_to_text(Y_train[i,:], word_index)

        # TODO concat but
        #disc_in = toolsSeq2Seq.concat_hist_reply(X, sentence, word_index)



