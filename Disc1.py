import numpy as np

import readFBTask1


import os




from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec


MAX_SEQUENCE_LENGTH = 200



embedding_matrix,x_train,x_val,y_train,y_val,word_index = readFBTask1.create_con(True,MAX_SEQUENCE_LENGTH)

y_train = to_categorical(np.asarray(y_train))
y_val = to_categorical(np.asarray(y_val))

print('Traing and validation set number of positive and negative reviews')
print (y_train.sum(axis=0))
print(y_val.sum(axis=0))

# GLOVE_DIR = "~/Testground/data/glove"
# embeddings_index = {}
# f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()
#
# print('Total %s word vectors.' % len(embeddings_index))
#
# embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
# for word, i in word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector
EMBEDDING_DIM = len(word_index) + 1
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_lstm = Bidirectional(LSTM(100,recurrent_dropout=0.3))(embedded_sequences)

preds = Dense(2, activation='softmax')(l_lstm)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Bidirectional LSTM")
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=10, batch_size=50)

