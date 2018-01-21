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

class Baseline(object):
    def __init__(self, max_seq_length, word_index, embedding_matrix):
        self.trained = False # Has the baseline been trained once ? 
        self.max_seq_length = max_seq_length
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
        preds = Dense(1)(l_lstm)
        
        self.model = Model(sequence_input, preds)
        self.model.compile(loss='mean_squared_error',
                      optimizer='rmsprop')
        
        print("model fitting - Bidirectional LSTM regressor")
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
        
        print('preds', self.model.preds.get_shape())
        exit(1)
        self.model.fit(x_train, y_train, validation_data=(x_val, y_val),
                  epochs=10, batch_size=50)

    def train(self, x_batch, y_batch, batch_size):
        """ 
        Train step
        Args: 
            x_batch: sentence output by G
            y_batch: expected reward (compute with MC rollout)
            batch_size: batch size
        """
        self.trained=True
        self.model.train_on_batch(x_batch, y_batch, batch_size) 
        print('train')

    def get_baseline(self, x_batch):
        """
        Prediction step
        Ags:
            x_batch: 
            y_batch: MC rewards
            batch_size:
        """
        if self.trained == False:
            baseline = np.zeros(x_batch.shape)
        else:
            baseline = self.model.predict_on_batch(x_batch) 
            print('get baseline')
        
        return baseline


def main():
    prin('TODO')
    # Test pre train

    # Test train

    # Test get rewards

if __name__=='__main__':
    main()
