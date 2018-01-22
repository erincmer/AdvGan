


import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


import pickle


def create_fake_dialogs(neg_can,words,num_fake):
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
                        train_T.append(sen + " " + s1 +  " eoh " + s2 + " eos ")
                        label_T.append(1)
                        train_T.append(sen + " " + s1 + " eoh " + neg_can[s] + " eos ")
                        label_T.append(0)
                        C = C + 1
                    s = np.random.randint(3, 10)
                    ranWords = np.random.choice(list(words), s,replace=True)
                    negSen = ' '.join(ranWords)
                    if negSen != s2:
                        train_T.append(sen + " " + s1 + " eoh " + s2 + " eos ")
                        label_T.append(1)
                        train_T.append(sen + " " + s1 + " eoh " + negSen + " eos ")
                        label_T.append(0)
                        C = C + 1

                sen = sen + " " + s1 + " " + s2

    f.close()
    return train_T,label_T

def create_dialogs(neg_can,words,num_fake):
    f = open('dialog-bAbI-tasks/dialog-babi-task1-API-calls-trn.txt', encoding='utf-8', errors='ignore')

    lines = f.readlines()
    hist_T = []
    reply_T = []
    in_reply_T = []
    for xx in lines:


            x = xx.split("\t")
            if (len(x) > 1):

                if x[0][0] == "1":
                    sen = " start dialog "

                s1 = x[0][x[0].find(" ") + 1:].rstrip()
                s2=  x[1].rstrip()



                hist_T.append(sen + " " + s1 + " eoh ")
                reply_T.append(s2 + " eos ")
                in_reply_T.append(" " + s2)



                sen = sen + " " + s1 + " " + s2

    f.close()
    return hist_T,reply_T,in_reply_T
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
    all_text.append(" eos ")
    all_text.append(" gogo ")
    all_text.append(" start dialog ")
    f.close()

    print("number of candidates ", len(neg_can))

    return neg_can,all_text
def create_con(create_data,MAX_SEQUENCE_LENGTH = 200,MAX_REP_SEQUENCE_LENGTH = 20):




    num_fake = 1

    if create_data:

    #

        neg_can,all_text = create_can()
        tokenizer = Tokenizer( )
        tokenizer.fit_on_texts(all_text)

        word_index = tokenizer.word_index

        all_hists,all_replies,all_in_replies = create_dialogs(neg_can,word_index,num_fake)

        all_dialogs, label_dialogs = create_fake_dialogs(neg_can, word_index, num_fake)
        test_histories = all_hists[-1000:]
        test_replies = all_replies [-1000:]
        test_in_replies = all_in_replies[-1000:]

        train_histories =  all_hists[:-1000]
        train_replies = all_replies[:-1000]
        train_in_replies = all_in_replies[:-1000]

        test_dialogs = all_dialogs[-1000:]
        label_test_dialogs = label_dialogs[-1000:]

        train_dialogs = all_dialogs[:-1000]
        label_train_dialogs = label_dialogs[:-1000]


        print("number of Training Set ",len(train_histories))
        print("real dialogues")
        print(train_dialogs[0:4])
        print("fake dialogues")
        print(train_dialogs[0:4])
        print("dialogue histories")
        print(train_histories[0:4])

        print("dialogue replies")
        print(train_replies[0:4])

        print("dialogue input replies")
        print(train_in_replies[0:4])

        embedding_matrix = np.zeros((len(word_index) + 1,len(word_index) + 1))
        print(" dictionary of words ")
        print(word_index)
        for word, i in word_index.items():

            embedding_vector = np.eye(len(word_index) + 1)[i]


            embedding_matrix[i] = embedding_vector

        print("tokenizer done")


        hist_Train = pad_sequences(tokenizer.texts_to_sequences(train_histories), maxlen=MAX_SEQUENCE_LENGTH, padding='post',value= word_index["eoh"])
        hist_Test = pad_sequences(tokenizer.texts_to_sequences(test_histories), maxlen=MAX_SEQUENCE_LENGTH, padding='post',value= word_index["eoh"])


        rep_Train = pad_sequences(tokenizer.texts_to_sequences(train_replies), maxlen=MAX_REP_SEQUENCE_LENGTH, padding='post',value= word_index["eos"])
        rep_Test = pad_sequences(tokenizer.texts_to_sequences(test_replies), maxlen=MAX_REP_SEQUENCE_LENGTH, padding='post',value= word_index["eos"])

        rep_in_Train = pad_sequences(tokenizer.texts_to_sequences(train_in_replies), maxlen=MAX_REP_SEQUENCE_LENGTH, padding='post')
        rep_in_Test = pad_sequences(tokenizer.texts_to_sequences(test_in_replies), maxlen=MAX_REP_SEQUENCE_LENGTH, padding='post')



        Train = pad_sequences(tokenizer.texts_to_sequences(train_dialogs), maxlen=MAX_SEQUENCE_LENGTH, padding='post')



        Test = pad_sequences(tokenizer.texts_to_sequences(test_dialogs), maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        print("training token is done")

        # print(hist_Train[0:3])
        # print(rep_Train[0:3])
        # input("wait")

        with open('emb_Task1.pickle', 'wb') as output:
                pickle.dump(embedding_matrix, output,protocol=4)

        with open('histTrain_Task1.pickle', 'wb') as output:
                pickle.dump(hist_Train, output, protocol=4)

        with open('histTest_Task1.pickle', 'wb') as output:
                pickle.dump(hist_Test, output, protocol=4)


        with open('repTrain_Task1.pickle', 'wb') as output:
                pickle.dump(rep_Train, output, protocol=4)
        with open('repInTrain_Task1.pickle', 'wb') as output:
                pickle.dump(rep_in_Train, output, protocol=4)

        with open('repTest_Task1.pickle', 'wb') as output:
                pickle.dump(rep_Test, output, protocol=4)
        with open('repInTest_Task1.pickle', 'wb') as output:
                pickle.dump(rep_in_Test, output, protocol=4)

        with open('wi_Task1.pickle', 'wb') as output:
            pickle.dump(word_index, output, protocol=4)

        print("saving finished ")

    else:
        with open('emb_Task1.pickle', 'rb') as output:
            embedding_matrix =pickle.load(output)

        with open('histTrain_Task1.pickle', 'rb') as output:
            hist_Train =pickle.load(output)
        with open('histTest_Task1.pickle', 'rb') as output:
            hist_Test = pickle.load(output)
        with open('repTrain_Task1.pickle', 'rb') as output:
            rep_Train =pickle.load(output)
        with open('repInTrain_Task1.pickle', 'rb') as output:
            rep_in_Train =pickle.load(output)
        with open('repTest_Task1.pickle', 'rb') as output:
            rep_Test = pickle.load(output)
        with open('repInTest_Task1.pickle', 'rb') as output:
            rep_in_Test = pickle.load(output)
        with open('wi_Task1.pickle', 'rb') as output:
            word_index =pickle.load(output)

        print("loading finished ")

    return embedding_matrix, np.array(hist_Train),np.array(hist_Test), np.array(rep_Train),np.array(rep_Test),\
           np.array(rep_in_Train),np.array(rep_in_Test),np.array(Train),np.array(Test), np.array(label_train_dialogs),np.array(label_test_dialogs),word_index

if __name__ == "__main__" :
    create_con(True, MAX_SEQUENCE_LENGTH=200, MAX_REP_SEQUENCE_LENGTH=20)