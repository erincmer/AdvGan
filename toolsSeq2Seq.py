
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

import headerSeq2Seq


def convert_id_to_text(ids,word_index):

    for id in ids:
        sen = ""
        for i in id:
            if i!=0  and i!= word_index["eos"] and i!=word_index['eoh']:
                sen = sen +" " +list(word_index.keys())[list(word_index.values()).index(i)]
        print(sen)

def convert_sentence_to_text(ids,word_index):
    """
    Print word sentence from token array
    Args:
        ids: sentence token array [rep_length]
        word_index: token:index dictionnary
    """

    sen = ""
    for i in ids:
        if i!=0  and i!= word_index["eos"]: # and i!=word_index['eoh']:
            sen = sen +" " +list(word_index.keys())[list(word_index.values()).index(i)]
    print(sen)

def convert_id_reward_to_text(ids,rewards,d_rewards,b,word_index):

    for id,r,dr,bb in zip(ids,rewards,d_rewards,b):
        sen = ""
        for i in id:
            if i!=0  and i!= word_index["eos"]:
                sen = sen +" " +list(word_index.keys())[list(word_index.values()).index(i)]
        print(sen," ====== ", r)
        print(dr)
        print(bb)

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
    disc_inp = np.full((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.MAX_SEQ_LENGTH), word_index['eos'])

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

def EOD(sentence, over_lines, word_index, word_term="api_call"):
    """
    Compute whether a sentence is the end of the dialogue.
    Based on the api_call criteria. If you use another criteria, change the
    code. 
    Args:
        sentence: Batch of sentence
        over_lines: Indicator on the dialogue termination.
            over_lines[i]==1 if sentence[i,:] is over
        word_index: token:index dictionnary
        word_term: Word on which termination is defined (default: api_call
    """
    
    # Get the lines where there is word_term
    # tmp > 0 for these lines
    tmp = np.sum(sentence==word_index.get(word_term), axis=1)

    # Get the index of the lines for which tmp>0
    ind = np.arange(headerSeq2Seq.BATCH_SIZE)
    over_ind = ind[tmp > 0]

    # Set over_lines to 1 for these index
    over_lines[over_ind] = 1

    return over_lines


def concat_hist_reply_over(histories, replies, word_index, over_lines):
    """ 
    Concatenate histories and replies for dialogues that are not over yet
    Args:
        hitories: history at time t-1 [batch_size x seq_length]
        replies: reply at time t [batch_size x rep_length]
        word_index: token:index dictionnary
        over_lines: lines for which the dialogue is over
    """
    
    # Column where to start concat for each dialogue line
    start_concat = np.sum(histories != word_index['eos'], axis=1)
    print("start_concat: ", start_concat.shape)

    # Column where the reply stops
    stop_replies = np.sum(replies != word_index['eos'], axis=1)
    print("stop_replies: ", stop_replies.shape)
    
    # Check that reply fits in dialogue vector
    can_concat = ((start_concat + stop_replies) < headerSeq2Seq.MAX_SEQ_LENGTH)

    # Check that dialogue did not end at previous step
    ind_not_over=(over_lines==0)

    # Compute the lines of replies to concatenate to histories
    ind_concat =np.arange(headerSeq2Seq.BATCH_SIZE)[(can_concat *
        ind_not_over)]
    print("ind_concat: ", ind_concat.shape)
    print(ind_concat)

    # Copy in case python pass histories by reference
    X = np.copy(histories)

    # Concatenate replies to histories
    for i in ind_concat:
        print("ind_concat[i]: ", ind_concat[i])
        print("start_concat[i]: ", start_concat[i])
        
        print("X[ind_concat[i], start_concat[i]:]: ", 
                X[ind_concat[i], start_concat[i]:].shape)
        
        print("replies[ind_concat[i], :stop_replies[i]]: ", 
                replies[ind_concat[i], :stop_replies[i]].shape)
        
        print("complete history: ",
                convert_sentence_to_text(X[ind_concat[i],:], word_index))
        
        print("Complete sentence: ",
                convert_sentence_to_text(replies[ind_concat[i],:], word_index))


        print("history : ", convert_sentence_to_text(X[ind_concat[i],
            start_concat[i]:], word_index))
        print("sentence: ", convert_sentence_to_text(replies[ind_concat[i],
            :stop_replies[i]], word_index))
            
        X[ind_concat[i], start_concat[i]:] = replies[ind_concat[i], :stop_replies[i]]

    return X

