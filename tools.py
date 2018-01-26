
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

import header


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
    disc_inp = np.full((header.BATCH_SIZE, header.MAX_SEQ_LENGTH), word_index['eos'])

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



