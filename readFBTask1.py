
import random



import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.python.client import device_lib
import math

import pickle

import string
local_device_protos = device_lib.list_local_devices()

print([x.name for x in local_device_protos if (x.device_type == 'GPU' or x.device_type=='CPU')])


def create_dialogs(neg_can,words,num_fake):
    f = open('dialog-bAbI-tasks/dialog-babi-task1-API-calls-trn.txt', encoding='utf-8', errors='ignore')

    lines = f.readlines()
    train_T = []
    label_T = []

    for xx in lines:


            x = xx.split("\t")
            if (len(x) > 1):

                if x[0][0] == "1":
                    sen = " start dialog "

                s1 = x[0][x[0].find(" ") + 1:].rstrip()
                s2=  x[1].rstrip()


                C = 0

                while C < num_fake*2:
                    # for s in range(len(neg_can)):
                    s = np.random.randint(0, len(neg_can))
                    if neg_can[s] != s2:
                        train_T.append(sen + " eoh " + s1 + " " + s2 + " eos ")
                        label_T.append(1)
                        train_T.append(sen + " eoh " + s1 + " " + neg_can[s] + " eos ")
                        label_T.append(0)
                        C = C + 1
                    s = np.random.randint(3, 10)
                    ranWords = np.random.choice(list(words), s,replace=True)
                    negSen = ' '.join(ranWords)
                    if negSen != s2:
                        train_T.append(sen + " eoh " + s1 + " " + s2 + " eos ")
                        label_T.append(1)
                        train_T.append(sen + " eoh " + s1 + " " + negSen + " eos ")
                        label_T.append(0)
                        C = C + 1

                sen = sen + " " + s1 + " " + s2

    f.close()
    return train_T,label_T
def create_can():
    f = open('dialog-bAbI-tasks/dialog-babi-task1-API-calls-trn.txt', encoding='utf-8', errors='ignore')
    lines = f.readlines()

    all_text = []
    neg_can = []
    for xx in lines:
        x = xx.split("\t")
        if (len(x) > 1):


            s1 = x[0][x[0].find(" ") + 1:].rstrip()
            s2 = x[1].rstrip()
            if s1 not in neg_can:
                neg_can.append(s1)
                all_text.append(s1)
            if s2 not in neg_can:
                neg_can.append(s2)
                all_text.append(s2)
    all_text.append(" eoh ")
    all_text.append(" start dialog ")
    f.close()

    print("number of candidates ", len(neg_can))

    return neg_can,all_text
def create_con(create_data,MAX_SEQUENCE_LENGTH = 200):




    num_fake = 1

    if create_data:

    #

        neg_can,all_text = create_can()
        tokenizer = Tokenizer( )
        tokenizer.fit_on_texts(all_text)

        word_index = tokenizer.word_index

        all_dialogs,label_dialogs = create_dialogs(neg_can,word_index,num_fake)
        test_dialogs = all_dialogs [-1000:]
        label_test_dialogs = label_dialogs [-1000:]

        train_dialogs = all_dialogs[:-1000]
        label_train_dialogs = label_dialogs[:-1000]

        print("number of Training Set ",len(train_dialogs))

        print("real dialogues")
        print(train_dialogs[0:4])
        print("fake dialogues")
        print(train_dialogs[0:4])

        embedding_matrix = np.zeros((len(word_index) + 1,len(word_index) + 1))
        print(" dictionary of words ")
        print(word_index)
        for word, i in word_index.items():

            embedding_vector = np.eye(len(word_index) + 1)[i]


            embedding_matrix[i] = embedding_vector

        print("tokenizer done")


        Train = pad_sequences(tokenizer.texts_to_sequences(train_dialogs), maxlen=MAX_SEQUENCE_LENGTH, padding='post')


        print("training token is done")


        Test = pad_sequences(tokenizer.texts_to_sequences(test_dialogs), maxlen=MAX_SEQUENCE_LENGTH, padding='post')



        with open('emb_Task1.pickle', 'wb') as output:
                pickle.dump(embedding_matrix, output,protocol=4)

        with open('Train_Task1.pickle', 'wb') as output:
                pickle.dump(Train, output, protocol=4)

        with open('Test_Task1.pickle', 'wb') as output:
                pickle.dump(Test, output, protocol=4)


        with open('labTrain_Task1.pickle', 'wb') as output:
                pickle.dump(label_train_dialogs, output, protocol=4)

        with open('labTest_Task1.pickle', 'wb') as output:
                pickle.dump(label_test_dialogs, output, protocol=4)

        with open('wi_Task1.pickle', 'wb') as output:
            pickle.dump(word_index, output, protocol=4)

        print("saving finished ")

    else:
        with open('emb_Task1.pickle', 'rb') as output:
            embedding_matrix =pickle.load(output)

        with open('Train_Task1.pickle', 'rb') as output:
            Train =pickle.load(output)
        with open('Test_Task1.pickle', 'rb') as output:
           Test = pickle.load(output)
        with open('labTrain_Task1.pickle', 'rb') as output:
            label_train_dialogs =pickle.load(output)
        with open('labTest_Task1.pickle', 'rb') as output:
            label_test_dialogs = pickle.load(output)

        with open('wi_Task1.pickle', 'rb') as output:
            word_index =pickle.load(output)

        print("loading finished ")

    return embedding_matrix, np.array(Train),np.array(Test), np.array(label_train_dialogs),np.array(label_test_dialogs),word_index

create_con(True,200)