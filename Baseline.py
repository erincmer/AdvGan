import numpy as np
import os
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec


MAX_SEQ_LENGTH = 200


class Baseline(object):
    def __init__(self, max_seq_length,rep_seq_length,batch_size, word_index, embedding_matrix):
        self.trained = True  # Has the baseline been trained once ?
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.rep_seq_length = rep_seq_length
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
        l_lstm = Bidirectional(LSTM(100, recurrent_dropout=0.3))(embedded_sequences)
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
        print(y_train.sum(axis=0))
        print(y_val.sum(axis=0))

        print('preds', self.model.preds.get_shape())
        exit(1)
        self.model.fit(x_train, y_train, validation_data=(x_val, y_val),
                       epochs=10, batch_size=50)

    def train(self, history, sentence,rewards,word_index):
        """
        Train step
        Args:
            x_batch: sentence output by G
            y_batch: expected reward (compute with MC rollout)
            batch_size: batch size
        """
        for t in range(1, self.rep_seq_length+1):
            history_update = np.copy(history)

            # Matrix [batch_size, rep_seq_length]
            # Line l: the first t elements are sentence[l,0:t]
            gen_input_t = np.zeros([self.batch_size, self.rep_seq_length])

            # DEBUG
            # print("t: ", t)
            # print("sentence.shape: ", sentence.shape)
            # print("gen_input_t.shape: ", gen_input_t.shape)
            # print("gen_input_t[:,0:t].shape: ", gen_input_t[:,0:t].shape)
            # print("sentence[:,0:t].shape: ", sentence[:,0:t].shape)

            gen_input_t[:, 0:t] = sentence[:, 0:t]

            # print("word_index['eoh']: ", word_index['eoh'])
            # print("word_index['eos']: ", word_index['eos'])
            # print("self.hist_end_token: ", self.hist_end_token)
            # print("history: ", history)
            # print("start_insert: ", start_insert)
            # print("start_insert.shape: ", start_insert.shape)
            # print("history.shape: ", history.shape)
            # print("history_update.shape: ", history_update.shape)
            # print("complete_sentence.shape: ", complete_sentence.shape)

            # train step
            history_update = self.concat_hist_reply(history_update, gen_input_t, word_index)
            output = self.model.train_on_batch(history_update, rewards[:,t-1])

        self.trained = True

        print('train')
        return output

    def concat_hist_reply(self,histories, replies, word_index):

        disc_inp = np.full((self.batch_size, self.max_seq_length ), word_index['eos'])
        counter = 0
        for h, r in zip(histories, replies):

            i = 0
            while i != word_index['eoh']:
                disc_inp[counter, i] = h[i]
                i = i + 1

            disc_inp[counter, i] = word_index['eoh']

            disc_inp[counter, i + 1:i + 21] = r
            counter = counter + 1

        return disc_inp
    def get_baseline(self, history, sentence, word_index):
        """
        Prediction step
        Ags:
            x_batch:
            y_batch: MC rewards
            batch_size:
        """

        baseline = np.zeros([self.batch_size, self.rep_seq_length])

        for t in range(1, self.rep_seq_length):
            history_update = np.copy(history)

            # Matrix [batch_size, rep_seq_length]
            # Line l: the first t elements are sentence[l,0:t]
            gen_input_t = np.zeros([self.batch_size, self.rep_seq_length])

            # DEBUG
            # print("t: ", t)
            # print("sentence.shape: ", sentence.shape)
            # print("gen_input_t.shape: ", gen_input_t.shape)
            # print("gen_input_t[:,0:t].shape: ", gen_input_t[:,0:t].shape)
            # print("sentence[:,0:t].shape: ", sentence[:,0:t].shape)

            gen_input_t[:, 0:t] = sentence[:, 0:t]

            # print("word_index['eoh']: ", word_index['eoh'])
            # print("word_index['eos']: ", word_index['eos'])
            # print("self.hist_end_token: ", self.hist_end_token)
            # print("history: ", history)
            # print("start_insert: ", start_insert)
            # print("start_insert.shape: ", start_insert.shape)
            # print("history.shape: ", history.shape)
            # print("history_update.shape: ", history_update.shape)
            # print("complete_sentence.shape: ", complete_sentence.shape)

            # Get baseline of these sentences
            history_update = self.concat_hist_reply(history_update, gen_input_t, word_index)
            baseline_val= self.model.predict_on_batch(history_update)
            baseline[:,t-1] = np.squeeze(baseline_val)

        return baseline
        # TODO the input is not ok
        if self.trained == False:
            baseline = np.zeros(x_batch.shape)
        else:
            print('get baseline')


def main():
    print('TODO')
    # Test pre train

    # Test train

    # Test get rewards


if __name__ == '__main__':
    main()