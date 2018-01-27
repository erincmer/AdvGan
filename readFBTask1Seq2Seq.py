
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


import pickle


def create_fake_dialogs(neg_can ,words ,num_fake):
    """
    Returns set of true and false pair of sentence. s1 (customer) is always true and s2
    (restaurant) may be false.
    Args:
        neg_can: All sentence in the dataset (without tags)
        words: dictionnary of token:index
        num_fake:
    """

    f = open('dialog-bAbI-tasks/dialog-babi-task1-API-calls-trn.txt', encoding='utf-8', errors='ignore')
    lines = f.readlines() # get all lines of all dialogues
    train_T = []
    label_T = []

    for xx in lines:
        x = xx.split("\t") # x:  ['1 hi', 'hello what can i help you with today\n']
        if (len(x) > 1):
            if x[0][0] == "1":
                sen = " start dialog "
            s1 = x[0][x[0].find(" ") + 1:].rstrip() # sentence 1 'hi'
            s2=  x[1].rstrip( )  # sentence 2 'hello what can i help you with today'

            # Get rid of api_call
            # if s2.split()[0] == 'api_call':
            #     continue

            C = 0
            while C < num_fake and "api_call" not in s2:
                # Choose randomly another sentence in the dataset instead of
                # the ground truth one
                cansAns = np.random.choice(len(neg_can), 6, replace=False)
                for s in cansAns :
                    if neg_can[s] != s2  :
                        train_T.append(sen + " " + s1 +  " eoh " + s2 + " eos ")
                        label_T.append(1)
                        train_T.append(sen + " " + s1 + " eoh " + neg_can[s] + " eos ")
                        label_T.append(0)


                for _ in range(2):
                    s = np.random.  randint(3, 20) # Sample random size of sentence
                    # Sample random sentence of size s
                    ranWords = np.random.choice(list( words), s,  replace=True) # array of words
                    negSen = ' '.join  (ranWords) # string of word
                    if negSen != s2:
                        train_T.append(sen + " " + s1 + " eoh " + s2 + " eos ")
                        label_T.append(1)
                        train_T.append(sen + " " + s1 + " eoh " + negSen + " eos ")
                        label_T.append(0)

                # print("\nRandom sentence of the same size")
                # print("s2: ", s2)
                for _ in range(2):
                    # Subsample s2 and shuffles it
                    s = np.random.randint(1, len(s2.split()))
                    ranWords = np.random.choice(list(s2. split()), s,replace=True)
                    negSen = ' '.join(ranWords)
                    # print("negSen: ", negSen)
                    if negSen != s2:
                        train_T.append(sen + " " + s1 + " eoh " + s2 + " eos ")
                        label_T.append(1)
                        train_T.append(sen + " " + s1 + " eoh " + negSen + " eos ")
                        label_T.append(0)
                #
                # print("\nRandom shuffle")
                # print("s2: ", s2)
                repeat = []
                for _ in range(2):
                    # Shuffle randomly s2
                    ranWords = np.array(s2.split())
                    np.random.shuffle(ranWords)
                    negSen = ' '.join(ranWords)
                    if negSen not in repeat and negSen != s2:
                        repeat.append(negSen)
                        # print("negSen: ", negSen)
                for negSen in repeat:
                    train_T.append(sen + " " + s1 + " eoh " + s2 + " eos ")
                    label_T.append(1)
                    train_T.append(sen + " " + s1 + " eoh " + negSen + " eos ")
                    label_T.append(0)
                # # print("\nRandom repeat")
                # # print("s2: ", s2)
                repeat = []
                for _ in range(2):
                    # Select randomly a token and pad the end sentence with it
                    ranWords = np.array(s2.split())
                    s_repeat = np. random.randint(1,ranWords.shape[0])
                    ranWords[s_repeat:] = ranWords[s_repeat]
                    negSen = ' '.join(ranWords)
                    if negSen not in repeat and negSen != s2:
                        repeat.append(negSen)
                        # print("negSen: ", negSen)
                for negSen in repeat:
                    train_T.append(sen + " " + s1 + " eoh " + s2 + " eos ")
                    label_T.append(1)
                    train_T.append(sen + " " + s1 + " eoh " + negSen + " eos ")
                    label_T.append(0)
                # print("\nRandom repeat maximum")
                # print("s2: ", s2)
                repeat = []
                for _ in range(2):
                    # Select randomly a token and pad the end sentence with it
                    # with max length
                    ranWords = s2.split()
                    s_repeat = np. random.randint(1,len(  s2.split())) # sample random word
                    for i in range( len(ranWords),19):
                        ranWords.append('')
                    ranWords = np.array(ranWords)
                    ranWords[s_repeat:] = ranWords[s_repeat]
                    # print(ranWords)
                    negSen = ' '.join(ranWords)
                    if negSen not in repeat and negSen != s2:
                        repeat.append(negSen)
                        # print("negSen: ", negSen)
                for negSen in repeat:
                    train_T.append(sen + " " + s1 + " eoh " + s2 + " eos ")
                    label_T.append(1)
                    train_T.append(sen + " " + s1 + " eoh " + negSen + " eos ")
                    label_T.append(0)
                #
                # print("\nRandom repeat maximum with new word")
                # print("s2: ", s2)
                repeat = []
                for _ in range(2):
                    # Select randomly a token and pad the end sentence with it
                    # with max length
                    ranWords = s2.split()
                    s_repeat = np. random.randint(1,len(words))
                    word_index_list = list(words)
                    word_pad =  word_index_list[s_repeat] # word to repeat
                    idx_repeat = np. random.randint(1,len(  s2.split())) # index to start padding
                    for i in range(idx_repeat,len(s2.split())):
                        ranWords[i] = word_pad
                    for _ in range(len (s2.split()),19):
                        ranWords.append(word_pad)
                    negSen = ' '.join(ranWords)
                    if negSen not in repeat and negSen != s2:
                        repeat.append(negSen)
                        # print("negSen: ", negSen)
                for negSen in repeat:
                    train_T.append(sen + " " + s1 + " eoh " + s2 + " eos ")
                    label_T.append(1)
                    train_T.append(sen + " " + s1 + " eoh " + negSen + " eos ")
                    label_T.append(0)

                C = C + 1

            sen = sen + " " + s1 + " " + s2

    f.close()
    return train_T,label_T


def create_dialogs( neg_can,words):
    """
    Returns all possible history finishing with customer sentence, sentences
    from the restaurant with eos, sentences from restaurant without eos
    Args:
        neg_can: All sentence in the dataset (without tags)
        words: dictionnary of token:index
    """
    f = open('dialog-bAbI-tasks/dialog-babi-task1-API-calls-trn.txt', encoding='utf-8', errors='ignore')
    lines = f.readlines()
    hist_T = [] # history finishing with customer sentence
    reply_T = [] # sentence from the restaurant which finishes with eos
    in_reply_T = [] # sentence from restaurant without eos
    # count=0
    for xx in lines:
        x = xx.split(  "\t") # # x:  ['1 hi', 'hello what can i help you with today\n']
        if (len(x) > 1):
            if x[0][0] == "1":
                sen = " start dialog "

            s1 = x[0][x[0].find(" ") + 1:].  rstrip() # sentence 1 'hi'
            s2=  x[1].  rstrip() # sentence 2 'hello what can i help you with today'

            # Get rid of api_call
            if s2.split()[0] == 'api_call':
                continue

            hist_T.append(sen + " " + s1 +  " eoh ")  # [' start dialog  hi eoh ']
            reply_T.append(s2 +  " eos ")  # ['hello what can i help you with today eos ']
            in_reply_T.append(" " + s2)  # [' hello what can i help you with today']

            # Reply with one wrong wor
            ranWords = s2.split()
            s_replace = np.random. randint(1,len(words))
            word_index_list = list(words)
            word_replace = word_index_list[  s_replace] # word that replaces
            idx_replace = np.random. randint(1,len(s2.  split())) # index of word to replace
            # print("idx_replace: ", idx_replace)
            ranWords[idx_replace] = word_replace
            negSen = ' '.join(ranWords)
            # print("s2: ", s2)
            # print("negSen: ", negSen)
            if negSen != s2:
                # reply_T.append(negSen + " eos ")
                in_reply_T.append(" " + negSen)
            sen = sen + " " + s1 + " " + s2


    f.close()
    return hist_T, reply_T, in_reply_T

def create_can():
    """
    Returns unique set of sentence present the dataset
    """
    f = open('dialog-bAbI-tasks/dialog-babi-task1-API-calls-trn.txt', encoding='utf-8', errors='ignore')
    lines = f.  readlines() # get all lines
    all_text = [] # contains all sentence in dataset + tags
    neg_can = [  ] # contains all sentence in dataset
    count = 0
    for xx in lines:
        x = xx.split(  "\t") # x:  ['1 hi', 'hello what can i help you with today\n']
        if (len(x) > 1):
            s1 = x[0][x[0].find(" ") + 1:].  rstrip() # sentence 1 'hi'
            s2 = x[1].  rstrip() # sentence 2 'hello what can i help you with today'
            # if s1 not in neg_can:
                # neg_can.append(s1)
            all_text.append(s1)
            if s2 not in neg_can:
                # Get rid of api_call
                if s2.split()[0] != 'api_call':
                    neg_can.append(s2)
                    all_text.append(s2)
    all_text.append(" eoh ")
    all_text.append(" eos ")
    all_text.append(" start dialog ")
    f.close()

    # Remark: neg_can == all_text \{eoh, eos, start_dialog}
    print("number of candidates ", len(neg_can))
    return neg_can,all_text


def create_con( create_data,MAX_SEQUENCE_LENGTH = 200,MAX_REP_SEQUENCE_LENGTH = 20):
    """

    Args:
        create_data: Set to False to load data from file
        MAX_SEQUENCE_LENGTH: dialogue max length
        MAX_REP_SEQUENCE_LENGTH: sentence max length 20
    """

    num_fake = 2

    if create_data:
        neg_can,all_text =  create_can() # Get all possible unique sentences
        tokenizer =Tokenizer( )
        tokenizer.fit_on_texts(all_text)
        word_index = tokenizer.  word_index # Dictionnary of token:index
        # print("word_index: ", word_index)

        # history ending with customer sentence
        # restaurant reply with eos
        # restaurant reply without eos
        print("create_dialogs ...")
        all_hists,all_replies ,all_in_replies = create_dialogs(neg_can ,word_index)

        # Set of 2 sentences with 1st always true and the 2nd maybe false
        print("create_fake_dialogs ...")
        all_dialogs, label_dialogs = create_fake_dialogs(neg_can, word_index, num_fake)

        test_histories = all_hists[-100:] # (..., cust) last 1000 histories
        test_replies = all_replies [-100:] # (cust, eos)
        test_in_replies = all_in_replies[-100:]  # (cust)

        train_histories =  all_hists[:-100]  # (..., cust)
        train_replies = all_replies[:-100] # (cust, eos)
        train_in_replies = all_in_replies[:-100]  # (cust)

        test_dialogs = all_dialogs[-1000:]  # (cust, rest)
        label_test_dialogs = label_dialogs[-1000:]  # (is rest true or not)

        train_dialogs = all_dialogs[:-1000]  # (cust, rest)
        label_train_dialogs = label_dialogs[:-1000]  # (is rest true or not)

        print(label_train_dialogs)
        print("number of Disc Training Set ",len(train_dialogs))
        print("number of Seq2Seq Training Set ", len(train_histories))
        print("real dialogues")
        print(train_dialogs[0:4])
        print("fake dialogues")
        print(train_dialogs[0:4])
        print("dialogue histories")
        print(train_histories[0:4])
        print("dialogue replies")
        print(train_replies[0:4])
        # print("dialogue input replies")
        # print(train_in_replies[0:4])

        embedding_matrix = np.zeros((len(word_index) + 1 ,len(word_index) + 1))
        # print(" dictionary of words ")
        # print(word_index)
        # TODO: What is that ?
        for word, i in word_index.items():
            embedding_vector = np.eye(len(word_index) + 1)[i]
            embedding_matrix[i] = embedding_vector

        # print('embedding_matrix: ', embedding_matrix)

        # print("tokenizer done")

        hist_Train = pad_sequences(tokenizer.texts_to_sequences(train_histories), maxlen=MAX_SEQUENCE_LENGTH, padding='post' ,value= word_index["eoh"])
        hist_Test = pad_sequences(tokenizer.texts_to_sequences(test_histories), maxlen=MAX_SEQUENCE_LENGTH, padding='post' ,value= word_index["eoh"])

        rep_Train = pad_sequences(tokenizer.texts_to_sequences(train_replies), maxlen=MAX_REP_SEQUENCE_LENGTH, padding='post' ,value= word_index["eos"])
        rep_Test = pad_sequences(tokenizer.texts_to_sequences(test_replies), maxlen=MAX_REP_SEQUENCE_LENGTH, padding='post' ,value= word_index["eos"])

        # rep_in_Train = pad_sequences(tokenizer.texts_to_sequences(train_in_replies), maxlen=MAX_REP_SEQUENCE_LENGTH, padding='post')
        # rep_in_Test = pad_sequences(tokenizer.texts_to_sequences(test_in_replies), maxlen=MAX_REP_SEQUENCE_LENGTH, padding='post')

        Train = pad_sequences(tokenizer.texts_to_sequences(train_dialogs), maxlen=MAX_SEQUENCE_LENGTH, padding='post'
                              ,value= word_index["eos"])

        Test = pad_sequences(tokenizer.texts_to_sequences(test_dialogs), maxlen=MAX_SEQUENCE_LENGTH, padding='post'
                             ,value= word_index["eos"])
        print("training token is done")

        # print(Train[0:3])
        # exit(0)
        # print(hist_Train[0:3])
        # print(rep_Train[0:3])
        # input("wait")

        with open('emb_Task1.pickle', 'wb') as output:
            pickle.dump(embedding_matrix, output ,protocol=4)
        with open('histTrain_Task1.pickle', 'wb') as output:
            pickle.dump(hist_Train, output, protocol=4)
        with open('histTest_Task1.pickle', 'wb') as output:
            pickle.dump(hist_Test, output, protocol=4)
        with open('repTrain_Task1.pickle', 'wb') as output:
            pickle.dump(rep_Train, output, protocol=4)
        with open('Train_Task1.pickle', 'wb') as output:
                pickle.dump(Train, output, protocol=4)
        with open('Test_Task1.pickle', 'wb') as output:
                pickle.dump(Test, output, protocol=4)
        with open('repTest_Task1.pickle', 'wb') as output:
            pickle.dump(rep_Test, output, protocol=4)
        with open('labTrain_Task1.pickle', 'wb') as output:
            pickle.dump(label_train_dialogs, output, protocol=4)
        with open('labTest_Task1.pickle', 'wb') as output:
            pickle.dump(label_test_dialogs, output, protocol=4)
        # with open('repInTest_Task1.pickle', 'wb') as output:
        #        pickle.dump(rep_in_Test, output, protocol=4)
        with open('wi_Task1.pickle', 'wb') as output:
            pickle.dump(word_index, output, protocol=4)
        with open('label_train_dialogs.pickle', 'wb') as output:
            pickle.dump(label_train_dialogs, output, protocol=4)
        with open('label_test_dialogs.pickle', 'wb') as output:
            pickle.dump(label_test_dialogs, output, protocol=4)

        print("saving finished ")

    else:
        with open('emb_Task1.pickle', 'rb') as output:
            embedding_matrix =pickle.load(output)

        with open('histTrain_Task1.pickle', 'rb') as output:
            hist_Train = pickle.load(output)
        with open('histTest_Task1.pickle', 'rb') as output:
            hist_Test = pickle.load(output)
        with open('repTrain_Task1.pickle', 'rb') as output:
            rep_Train = pickle.load(output)
        with open('Train_Task1.pickle', 'rb') as output:
           Train =pickle.load(output)
        with open('repTest_Task1.pickle', 'rb') as output:
            rep_Test = pickle.load(output)
        with open('labTrain_Task1.pickle', 'rb') as output:
            label_train_dialogs = pickle.load(output)
        with open('labTest_Task1.pickle', 'rb') as output:
            label_test_dialogs = pickle.load(output)
        with open('Test_Task1.pickle', 'rb') as output:
           Test = pickle.load(output)
        with open('wi_Task1.pickle', 'rb') as output:
            word_index = pickle.load(output)
        with open('label_train_dialogs.pickle', 'rb') as output:
            label_train_dialogs = pickle.load(output)
        with open('label_test_dialogs.pickle', 'rb') as output:
            label_test_dialogs = pickle.load(output)

        print("loading finished ")

    # embedding matrix: ?
    # hist_train: history (..., cust)
    # rep_train: (restaurant reply)
    # Train: (cust, rest)
    # label_train_dialog: is rest in Train true or not
    return embedding_matrix, np.array(hist_Train), np.array(hist_Test), np.array(rep_Train), np.array(
        rep_Test), np.array(Train), np.array(Test), np.array(label_train_dialogs), np.array(
        label_test_dialogs), word_index


if __name__ == "__main__":
    create_con(True, MAX_SEQUENCE_LENGTH=200, MAX_REP_SEQUENCE_LENGTH=20)
