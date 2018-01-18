import numpy as np
import os
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec

import readFBTask1

MAX_SEQ_LENGTH = 200

class Discriminator(object):
    def __init__(self, word_index, embedding_matrix):
        self.max_seq_length = MAX_SEQ_LENGTH
        self.word_index = word_index
        self.embedding_matrix = embedding_matrix 
        self.embedding_dim = len(word_index) + 1
        self.embedding_layer = Embedding(len(word_index) + 1,
                                    self.embedding_dim,
                                    weights=[self.embedding_matrix],
                                    input_length=self.max_seq_length,
                                    trainable=True)
        
        sequence_input = Input(shape=(self.max_seq_length,), dtype='int32')
        embedded_sequences = self.embedding_layer(sequence_input)
        l_lstm = Bidirectional(LSTM(100,recurrent_dropout=0.3))(embedded_sequences)
        
        preds = Dense(2, activation='softmax')(l_lstm)
        self.model = Model(sequence_input, preds)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])
        
        print("model fitting - Bidirectional LSTM")
        self.model.summary()

    def pretrain(self, x_train, x_val, y_train, y_val):
        """
        Args:
            embedding_matrix: 
            x_train: Complete train sentences
            x_val: Complete eval sentences
            y_train: Label of train sentences
            y_val: Label of val sentences
            word_index: 
        """

        y_train = to_categorical(np.asarray(y_train))
        y_val = to_categorical(np.asarray(y_val))
        
        print('Traing and validation set number of positive and negative reviews')
        print (y_train.sum(axis=0))
        print(y_val.sum(axis=0))
        
        #EMBEDDING_DIM = len(word_index) + 1
        #embedding_layer = Embedding(len(word_index) + 1,
        #                            EMBEDDING_DIM,
        #                            weights=[embedding_matrix],
        #                            input_length=MAX_SEQUENCE_LENGTH,
        #                            trainable=True)
        
        #sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        #embedded_sequences = embedding_layer(sequence_input)
        #l_lstm = Bidirectional(LSTM(100,recurrent_dropout=0.3))(embedded_sequences)
        #
        #preds = Dense(2, activation='softmax')(l_lstm)
        #model = Model(sequence_input, preds)
        #model.compile(loss='categorical_crossentropy',
        #              optimizer='rmsprop',
        #              metrics=['acc'])
        #
        #print("model fitting - Bidirectional LSTM")
        #model.summary()
        self.model.fit(x_train, y_train, validation_data=(x_val, y_val),
                  epochs=10, batch_size=50)

    def train(self, x_batch, y_batch):
        print('train')

    def get_rewards(self, x_batch, y_batch):
        print('get rewards')


def main():
    embedding_matrix,x_train,x_val,y_train,y_val,word_index = readFBTask1.create_con(True,MAX_SEQ_LENGTH)
    disc = Discriminator(word_index, embedding_matrix)
    disc.pretrain(x_train, x_val, y_train, y_val)

if __name__=='__main__':
    main()
