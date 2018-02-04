
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
        if i!=0  and i!= word_index["eos"] and i!=word_index['eoh']:
            sen = sen +" " +list(word_index.keys())[list(word_index.values()).index(i)]
    return sen

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

def concat_hist_new( histories, replies, word_index):
    """
    Concat history and reply to make a new history (i.e. put eoh after the
    reply)
    Args:
        histories: (end with eoh that you must get rid of)
        replies: replies
        word_index: token:index dictionnary
    """
    disc_inp = np.full((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.MAX_SEQ_LENGTH), word_index['eos'])
    
    #print("eoh: ", word_index['eoh'])
    #print("eos: ", word_index['eos'])
    counter = 0
    for h, r in zip(histories, replies):
        i = 0
        while h[i] != word_index['eoh']:
            disc_inp[counter, i] = h[i]
            i = i + 1
        
        disc_inp[counter, i + 1:i + np.sum(r!=word_index['eos'])+1] = r[r!=word_index['eos']]
        disc_inp[counter, i + np.sum(r!=word_index['eos'])+1] = word_index['eoh']

        #print("first eoh at : ", i)
        #print("history: ", h)

        #ieos = 0
        #print("replies: ", r)
        #while ieos < 20 and r[ieos] !=  word_index['eos']:
        #    ieos+=1

        #print("first eos at : ", ieos)
        #input("wait")

        #disc_inp[counter, i] = word_index['eoh']
        #disc_inp[counter, i : i+ieos] = r[:ieos]
        #disc_inp[counter, i + i+ieos] = word_index['eoh']
        counter = counter + 1

    return disc_inp


#def EOD(sentence, over_lines, word_index, word_term="api_call"):
def EOD(sentence, word_index, word_term="api"):
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
    
    # Get the lines where there starting with api call
    # tmp > 0 for these lines
    is_api = (sentence[:,0] == word_index.get("api"))
    is_call = (sentence[:,1] == word_index.get("call"))

    is_over = is_api * is_call

    # Get the index of the lines for which tmp>0
    ind = np.arange(headerSeq2Seq.BATCH_SIZE)
    over_ind = ind[is_over]
    
    # DEBUG
    #for i in over_ind:
    #    print("i: ", i)
    #    print(convert_sentence_to_text(sentence[int(i),:], word_index))
    #    input("wait")
        
    return over_ind

    ## Get the lines where there is call
    ## tmp > 0 for these lines
    #tmp_call = np.sum(sentence==word_index.get("call"), axis=1)

    ## Get the index of the lines for which tmp>0
    #ind_call = np.arange(headerSeq2Seq.BATCH_SIZE)
    #over_ind_call = ind_call[tmp_call > 0]

    ## Get the lines both having api and call

    ## Set over_lines to 1 for these index
    #over_lines = np.zeros(headerSeq2Seq.BATCH_SIZE, dtype=np.int)
    #over_lines[over_ind] = 1

    #return over_lines


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

def save_to_file(filename, ids,word_index):
    f = open(filename, 'w+')
    for id in ids:
        sen = ""
        for i in id:
            if i!=0  and i!= word_index["eos"] and i!= word_index["eoh"]:
                sen = sen +" " +list(word_index.keys())[list(word_index.values()).index(i)]
        f.write(sen + '\n')

def compute_rewards(sen2_exp, sen2, mode):

    # Reward = 0 if apicall_exp is included in api_call
    # No order restriction
    if mode==1:
        apicall = sen2[:,0:6]
        apicall_exp = sen2_exp[:,2:8]

        print("Expected api call: ", apicall[0,:])
        print("api call: ", apicall_exp[0,:])

        rewards = np.zeros(headerSeq2Seq.BATCH_SIZE)
        for i in range(rewards.shape[0]):
            correct_apicall = np.in1d(apicall[i,:], apicall_exp[i,:])
            num_correct = np.sum(correct_apicall)
            if num_correct > 0:
                rewards[i] = 1

        return rewards

    ## Reward = 0 if apicall_exp is equal  in api_call
    ## No order restricition
    elif mode==2:
        apicall = sen2[:,0:6]
        apicall_exp = sen2_exp[:,2:8]

        rewards = np.zeros(headerSeq2Seq.BATCH_SIZE)
        for i in range(rewards.shape[0]):
            correct_apicall = np.in1d(apicall[i,:], apicall_exp[i,:])
            num_correct = np.sum(correct_apicall)
            if num_correct == apicall[i,:].shape[0] :
                rewards[i] = 1
        
        return rewards

    ## Reward is the ratio bet
    elif mode == 3:
        apicall = sen2[:,0:6]
        apicall_exp = sen2_exp[:,2:8]

        rewards = np.zeros(headerSeq2Seq.BATCH_SIZE)
        for i in range(rewards.shape[0]):
            correct_apicall = np.in1d(apicall[i,:], apicall_exp[i,:])
            num_correct = np.sum(correct_apicall)
            rewards[i] = num_correct / apicall[i,:].shape[0]

        return rewards
