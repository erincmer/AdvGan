
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import pickle


# def create_fake_dialogs(neg_can ,words ,num_fake):
#     """
#     Returns set of true and false pair of sentence. s1 (customer) is always true and s2
#     (restaurant) may be false.
#     Args:
#         neg_can: All sentence in the dataset (without tags)
#         words: dictionnary of token:index
#         num_fake:
#     """
#
#     f = open('dialog-bAbI-tasks/dialog-babi-task1-API-calls-trn.txt', encoding='utf-8', errors='ignore')
#     lines = f.readlines() # get all lines of all dialogues
#     train_T = []
#     label_T = []
#
#     for xx in lines:
#         x = xx.split("\t") # x:  ['1 hi', 'hello what can i help you with today\n']
#         if (len(x) > 1):
#             if x[0][0] == "1":
#                 sen = " start dialog "
#             s1 = x[0][x[0].find(" ") + 1:].rstrip() # sentence 1 'hi'
#             s2=  x[1].rstrip( )  # sentence 2 'hello what can i help you with today'
#
#             # ############### First Agent##############
#             C = 0
#             while C < num_fake :
#                 # Choose randomly another sentence in the dataset instead of
#                 # the ground truth one
#                 cansAns = np.random.choice(len(neg_can), 2, replace=False)
#                 for s in cansAns:
#                     if neg_can[s] != s1:
#                         train_T.append(sen + " " + " eoh " + s1 + " eos ")
#                         label_T.append(1)
#                         train_T.append(sen + " " + " eoh " + neg_can[s] + " eos ")
#                         label_T.append(0)
#
#                 for _ in range(2):
#                     s = np.random.randint(3, 20)  # Sample random size of sentence
#                     # Sample random sentence of size s
#                     ranWords = np.random.choice(list(words), s, replace=True)  # array of words
#                     negSen = ' '.join(ranWords)  # string of word
#                     if negSen != s1:
#                         train_T.append(sen + " " + " eoh " + s1 + " eos ")
#                         label_T.append(1)
#                         train_T.append(sen + " " + " eoh " + negSen + " eos ")
#                         label_T.append(0)
#
#                 # print("\nRandom sentence of the same size")
#                 # print("s2: ", s2)
#                 # for _ in range(2):
#                 #     # Subsample s2 and shuffles it
#                 #     s = np.random.randint(1, len(s2.split()))
#                 #     ranWords = np.random.choice(list(s2. split()), s,replace=True)
#                 #     negSen = ' '.join(ranWords)
#                 #     # print("negSen: ", negSen)
#                 #     if negSen != s2:
#                 #         train_T.append(sen + " " + s1 + " eoh " + s2 + " eos ")
#                 #         label_T.append(1)
#                 #         train_T.append(sen + " " + s1 + " eoh " + negSen + " eos ")
#                 #         label_T.append(0)
#                 # #
#                 # # print("\nRandom shuffle")
#                 # # print("s2: ", s2)
#                 # repeat = []
#                 # for _ in range(2):
#                 #     # Shuffle randomly s2
#                 #     ranWords = np.array(s2.split())
#                 #     np.random.shuffle(ranWords)
#                 #     negSen = ' '.join(ranWords)
#                 #     if negSen not in repeat and negSen != s2:
#                 #         repeat.append(negSen)
#                 #         # print("negSen: ", negSen)
#                 # for negSen in repeat:
#                 #     train_T.append(sen + " " + s1 + " eoh " + s2 + " eos ")
#                 #     label_T.append(1)
#                 #     train_T.append(sen + " " + s1 + " eoh " + negSen + " eos ")
#                 #     label_T.append(0)
#                 # # # print("\nRandom repeat")
#                 # # # print("s2: ", s2)
#                 # repeat = []
#                 # for _ in range(2):
#                 #     # Select randomly a token and pad the end sentence with it
#                 #     ranWords = np.array(s2.split())
#                 #     s_repeat = np. random.randint(1,ranWords.shape[0])
#                 #     ranWords[s_repeat:] = ranWords[s_repeat]
#                 #     negSen = ' '.join(ranWords)
#                 #     if negSen not in repeat and negSen != s2:
#                 #         repeat.append(negSen)
#                 #         # print("negSen: ", negSen)
#                 # for negSen in repeat:
#                 #     train_T.append(sen + " " + s1 + " eoh " + s2 + " eos ")
#                 #     label_T.append(1)
#                 #     train_T.append(sen + " " + s1 + " eoh " + negSen + " eos ")
#                 #     label_T.append(0)
#                 # # print("\nRandom repeat maximum")
#                 # # print("s2: ", s2)
#                 # repeat = []
#                 # for _ in range(2):
#                 #     # Select randomly a token and pad the end sentence with it
#                 #     # with max length
#                 #     ranWords = s2.split()
#                 #     s_repeat = np. random.randint(1,len(  s2.split())) # sample random word
#                 #     for i in range( len(ranWords),19):
#                 #         ranWords.append('')
#                 #     ranWords = np.array(ranWords)
#                 #     ranWords[s_repeat:] = ranWords[s_repeat]
#                 #     # print(ranWords)
#                 #     negSen = ' '.join(ranWords)
#                 #     if negSen not in repeat and negSen != s2:
#                 #         repeat.append(negSen)
#                 #         # print("negSen: ", negSen)
#                 # for negSen in repeat:
#                 #     train_T.append(sen + " " + s1 + " eoh " + s2 + " eos ")
#                 #     label_T.append(1)
#                 #     train_T.append(sen + " " + s1 + " eoh " + negSen + " eos ")
#                 #     label_T.append(0)
#                 # #
#                 # # print("\nRandom repeat maximum with new word")
#                 # # print("s2: ", s2)
#                 # repeat = []
#                 # for _ in range(2):
#                 #     # Select randomly a token and pad the end sentence with it
#                 #     # with max length
#                 #     ranWords = s2.split()
#                 #     s_repeat = np. random.randint(1,len(words))
#                 #     word_index_list = list(words)
#                 #     word_pad =  word_index_list[s_repeat] # word to repeat
#                 #     idx_repeat = np. random.randint(1,len(  s2.split())) # index to start padding
#                 #     for i in range(idx_repeat,len(s2.split())):
#                 #         ranWords[i] = word_pad
#                 #     for _ in range(len (s2.split()),19):
#                 #         ranWords.append(word_pad)
#                 #     negSen = ' '.join(ranWords)
#                 #     if negSen not in repeat and negSen != s2:
#                 #         repeat.append(negSen)
#                 #         # print("negSen: ", negSen)
#                 # for negSen in repeat:
#                 #     train_T.append(sen + " " + s1 + " eoh " + s2 + " eos ")
#                 #     label_T.append(1)
#                 #     train_T.append(sen + " " + s1 + " eoh " + negSen + " eos ")
#                 #     label_T.append(0)
#
#                 C = C + 1
#                 # ############### First Agent##############
#             # Get rid of api_call
#             # if s2.split()[0] == 'api_call':
#             #     continue
#             # ############### Second Agent##############
#             # C = 0
#             # while C < num_fake and "api_call5" not in s2:
#             #     # Choose randomly another sentence in the dataset instead of
#             #     # the ground truth one
#             #     cansAns = np.random.choice(len(neg_can), 2, replace=False)
#             #     for s in cansAns :
#             #         if neg_can[s] != s2  :
#             #             train_T.append(sen + " " + s1 +  " eoh " + s2 + " eos ")
#             #             label_T.append(1)
#             #             train_T.append(sen + " " + s1 + " eoh " + neg_can[s] + " eos ")
#             #             label_T.append(0)
#             #
#             #
#             #     for _ in range(2):
#             #         s = np.random.  randint(3, 20) # Sample random size of sentence
#             #         # Sample random sentence of size s
#             #         ranWords = np.random.choice(list( words), s,  replace=True) # array of words
#             #         negSen = ' '.join  (ranWords) # string of word
#             #         if negSen != s2:
#             #             train_T.append(sen + " " + s1 + " eoh " + s2 + " eos ")
#             #             label_T.append(1)
#             #             train_T.append(sen + " " + s1 + " eoh " + negSen + " eos ")
#             #             label_T.append(0)
#
#                 # print("\nRandom sentence of the same size")
#                 # print("s2: ", s2)
#                 # for _ in range(2):
#                 #     # Subsample s2 and shuffles it
#                 #     s = np.random.randint(1, len(s2.split()))
#                 #     ranWords = np.random.choice(list(s2. split()), s,replace=True)
#                 #     negSen = ' '.join(ranWords)
#                 #     # print("negSen: ", negSen)
#                 #     if negSen != s2:
#                 #         train_T.append(sen + " " + s1 + " eoh " + s2 + " eos ")
#                 #         label_T.append(1)
#                 #         train_T.append(sen + " " + s1 + " eoh " + negSen + " eos ")
#                 #         label_T.append(0)
#                 # #
#                 # # print("\nRandom shuffle")
#                 # # print("s2: ", s2)
#                 # repeat = []
#                 # for _ in range(2):
#                 #     # Shuffle randomly s2
#                 #     ranWords = np.array(s2.split())
#                 #     np.random.shuffle(ranWords)
#                 #     negSen = ' '.join(ranWords)
#                 #     if negSen not in repeat and negSen != s2:
#                 #         repeat.append(negSen)
#                 #         # print("negSen: ", negSen)
#                 # for negSen in repeat:
#                 #     train_T.append(sen + " " + s1 + " eoh " + s2 + " eos ")
#                 #     label_T.append(1)
#                 #     train_T.append(sen + " " + s1 + " eoh " + negSen + " eos ")
#                 #     label_T.append(0)
#                 # # # print("\nRandom repeat")
#                 # # # print("s2: ", s2)
#                 # repeat = []
#                 # for _ in range(2):
#                 #     # Select randomly a token and pad the end sentence with it
#                 #     ranWords = np.array(s2.split())
#                 #     s_repeat = np. random.randint(1,ranWords.shape[0])
#                 #     ranWords[s_repeat:] = ranWords[s_repeat]
#                 #     negSen = ' '.join(ranWords)
#                 #     if negSen not in repeat and negSen != s2:
#                 #         repeat.append(negSen)
#                 #         # print("negSen: ", negSen)
#                 # for negSen in repeat:
#                 #     train_T.append(sen + " " + s1 + " eoh " + s2 + " eos ")
#                 #     label_T.append(1)
#                 #     train_T.append(sen + " " + s1 + " eoh " + negSen + " eos ")
#                 #     label_T.append(0)
#                 # # print("\nRandom repeat maximum")
#                 # # print("s2: ", s2)
#                 # repeat = []
#                 # for _ in range(2):
#                 #     # Select randomly a token and pad the end sentence with it
#                 #     # with max length
#                 #     ranWords = s2.split()
#                 #     s_repeat = np. random.randint(1,len(  s2.split())) # sample random word
#                 #     for i in range( len(ranWords),19):
#                 #         ranWords.append('')
#                 #     ranWords = np.array(ranWords)
#                 #     ranWords[s_repeat:] = ranWords[s_repeat]
#                 #     # print(ranWords)
#                 #     negSen = ' '.join(ranWords)
#                 #     if negSen not in repeat and negSen != s2:
#                 #         repeat.append(negSen)
#                 #         # print("negSen: ", negSen)
#                 # for negSen in repeat:
#                 #     train_T.append(sen + " " + s1 + " eoh " + s2 + " eos ")
#                 #     label_T.append(1)
#                 #     train_T.append(sen + " " + s1 + " eoh " + negSen + " eos ")
#                 #     label_T.append(0)
#                 # #
#                 # # print("\nRandom repeat maximum with new word")
#                 # # print("s2: ", s2)
#                 # repeat = []
#                 # for _ in range(2):
#                 #     # Select randomly a token and pad the end sentence with it
#                 #     # with max length
#                 #     ranWords = s2.split()
#                 #     s_repeat = np. random.randint(1,len(words))
#                 #     word_index_list = list(words)
#                 #     word_pad =  word_index_list[s_repeat] # word to repeat
#                 #     idx_repeat = np. random.randint(1,len(  s2.split())) # index to start padding
#                 #     for i in range(idx_repeat,len(s2.split())):
#                 #         ranWords[i] = word_pad
#                 #     for _ in range(len (s2.split()),19):
#                 #         ranWords.append(word_pad)
#                 #     negSen = ' '.join(ranWords)
#                 #     if negSen not in repeat and negSen != s2:
#                 #         repeat.append(negSen)
#                 #         # print("negSen: ", negSen)
#                 # for negSen in repeat:
#                 #     train_T.append(sen + " " + s1 + " eoh " + s2 + " eos ")
#                 #     label_T.append(1)
#                 #     train_T.append(sen + " " + s1 + " eoh " + negSen + " eos ")
#                 #     label_T.append(0)
#
#                 # C = C + 1
#             # ############### Second Agent##############
#             sen = sen + " " + s1 + " " + s2
#
#     f.close()
#     return train_T,label_T
#

def create_test_s1_interactive():
    """
    Returns all the unique api_call of test set
    """
    f = open('dialog-bAbI-tasks/api_call_test.txt', encoding='utf-8', errors='ignore')
    lines = f.readlines()

    hist_s1 = []

    counter = 0
    for xx in lines:
        #print("xx: ", xx)
        x = xx.split("\t")  # # x:  ['1 hi', 'hello what can i help you with today\n']
        #print("len(x): ", len(x))
        if (len(x) == 1):
            #print("x: ", x)
            if x[0][0] == "1":
                sen = " start dialog"

            s1 = x[0][x[0].find(" "):].rstrip()  # sentence 1 'hi'
            #print("s1: ", s1)
            #print(len(s1))
            #input("wait")
            if len(s1) != 0:
                hist_s1.append(sen + " " + s1 + " eoh ")

    f.close()
    #print("READ: ", hist_s1)
    return  hist_s1


def create_test_dialogs():
    """
    Returns all possible history finishing with customer sentence, sentences
    from the restaurant with eos, sentences from restaurant without eos
    Args:
        neg_can: All sentence in the dataset (without tags)
        words: dictionnary of token:index
    """
    f = open('dialog-bAbI-tasks/dialog-babi-task1-API-calls-tst.txt', encoding='utf-8', errors='ignore')
    lines = f.readlines()

    hist_s2 = []
    hist_s2_OOV = []

    reply_s2 = []
    reply_s2_OOV = []


    counter = 0
    for xx in lines:
        x = xx.split("\t")  # # x:  ['1 hi', 'hello what can i help you with today\n']
        if (len(x) > 1):
            if x[0][0] == "1":
                sen = " start dialog"


            s1 = x[0][x[0].find(" ") + 1:].rstrip()  # sentence 1 'hi'
            s2 = x[1].rstrip()  # sentence 2 'hello what can i help you with today'



            hist_s2.append(sen + " " + s1 + " eoh ")
            reply_s2.append(s2 + "  " + " eos ")

            sen = sen + " " + s1 + " " + s2
    f.close()
    f = open('dialog-bAbI-tasks/dialog-babi-task1-API-calls-tst-OOV.txt', encoding='utf-8', errors='ignore')
    lines = f.readlines()
    for xx in lines:
        x = xx.split("\t")  # # x:  ['1 hi', 'hello what can i help you with today\n']
        if (len(x) > 1):
            if x[0][0] == "1":
                sen = " start dialog"


            s1 = x[0][x[0].find(" ") + 1:].rstrip()  # sentence 1 'hi'
            s2 = x[1].rstrip()  # sentence 2 'hello what can i help you with today'



            hist_s2_OOV.append(sen + " " + s1 + " eoh ")
            reply_s2_OOV.append(s2  + " eos ")

            sen = sen + " " + s1 + " " + s2

    f.close()
    return  hist_s2, reply_s2,hist_s2_OOV,reply_s2_OOV



def create_dialogs():
    """
    Returns all possible history finishing with customer sentence, sentences
    from the restaurant with eos, sentences from restaurant without eos
    Args:
        neg_can: All sentence in the dataset (without tags)
        words: dictionnary of token:index
    """
    f = open('dialog-bAbI-tasks/dialog-babi-task1-API-calls-trn.txt', encoding='utf-8', errors='ignore')
    lines = f.readlines()
    hist_s1 = [] # history finishing with customer sentence
    hist_s2 = []
    hist_s3 = []
    reply_s1 = []
    reply_s2 = []
    reply_s3 = []

    lite_hist_s1 = [] # history finishing with customer sentence
    lite_hist_s2 = []
    lite_hist_s3 = []
    lite_reply_s1 = []
    lite_reply_s2 = []
    lite_reply_s3 = []


    counter = 0
    for xx in lines:
        x = xx.split(  "\t") # # x:  ['1 hi', 'hello what can i help you with today\n']
        if (len(x) > 1):
            if x[0][0] == "1":
                sen = " start dialog"
                prev_sen = "notSet"
                prev_s2  = "notSet"
                counter = counter + 1

            s1 = x[0][x[0].find(" ") + 1:].  rstrip() # sentence 1 'hi'
            s2=  x[1].  rstrip() # sentence 2 'hello what can i help you with today'

            if counter <50:
                lite_hist_s1.append(sen + " eoh ")
                lite_reply_s1.append(s1  + " eos ")

                lite_hist_s2.append(sen + " " + s1 + " eoh ")
                lite_reply_s2.append(s2 + " eos ")

                lite_hist_s3.append(sen + " eoh ")
                lite_reply_s3.append(s1 + "  " + s2 + " " + " eos ")

                if prev_sen != "notSet" and prev_s2 != "notSet":
                    lite_hist_s3.append(prev_sen)
                    lite_reply_s3.append(prev_s2 + "  " + s1 + "  " + " eos ")




            hist_s1.append(sen + " eoh ")
            reply_s1.append(s1 + "  " + " eos ")

            hist_s2.append(sen + " " + s1 +  " eoh ")
            reply_s2.append(s2 + " " + " eos ")

            hist_s3.append(sen + " eoh ")
            reply_s3.append(s1 + " " +s2 + " " + " eos ")


            if prev_sen != "notSet" and prev_s2 != "notSet":

                hist_s3.append(prev_sen)
                reply_s3.append(prev_s2 + " " + s1 + " " + " eos ")
            prev_sen =  sen + " " + s1 + " eoh "
            prev_s2 = s2
            sen = sen + " " + s1 + " " + s2


    f.close()
    return hist_s1,reply_s1,hist_s2,reply_s2,hist_s3,reply_s3,lite_hist_s1,lite_reply_s1,lite_hist_s2,lite_reply_s2,lite_hist_s3,lite_reply_s3

def create_can():
    """
    Returns unique set of sentence present the dataset
    """
    f = open('dialog-bAbI-tasks/dialog-babi-task1-API-calls-trn.txt', encoding='utf-8', errors='ignore')
    lines = f.  readlines() # get all lines
    all_text = [] # contains all sentence in dataset + tags

    neg_can_s1 = [] # contains all sentence in dataset
    neg_can_s2 = []
    neg_can_s2_no_apicall = []

    for xx in lines:
        x = xx.split(  "\t") # x:  ['1 hi', 'hello what can i help you with today\n']
        if (len(x) > 1):
            s1 = x[0][x[0].find(" ") + 1:].  rstrip() # sentence 1 'hi'
            s2 = x[1].  rstrip() # sentence 2 'hello what can i help you with today'
            # if s1 not in neg_can:
                # neg_can.append(s1)

            all_text.append(s1)
            all_text.append(s2)
            ############### First Agent##############

            if s1 not in neg_can_s1:


                    neg_can_s1.append(s1)

            ############### First Agent##############

            # ############### Second Agent##############

            if s2 not in neg_can_s2:
                if s2.split()[0] != 'api_call':
                    neg_can_s2_no_apicall.append(s2)

                neg_can_s2.append(s2)

            # ############### Second Agent##############
    f.close()
    f = open('dialog-bAbI-tasks/dialog-babi-task1-API-calls-tst-OOV.txt', encoding='utf-8', errors='ignore')
    # max_len = 0
    # for xx in all_text:
    #     if  len(xx.split())>max_len:
    #         max_len   = len(xx.split())
    #         print(len(xx.split()))
    #         print(xx)
    #
    #
    # input("wait")

    for xx in lines:
        x = xx.split(  "\t") # x:  ['1 hi', 'hello what can i help you with today\n']
        if (len(x) > 1):
            s1 = x[0][x[0].find(" ") + 1:].  rstrip() # sentence 1 'hi'
            s2 = x[1].  rstrip() # sentence 2 'hello what can i help you with today'
            # if s1 not in neg_can:
                # neg_can.append(s1)

            all_text.append(s1)
            all_text.append(s2)

    f.close()
    all_text.append(" start dialog ")
    all_text.append(" eoh ")

    all_text.append(" eos ")


    # Remark: neg_can == all_text \{eoh, eos, start_dialog}
    print("number of candidates ", len(neg_can_s1),len(neg_can_s2))
    return neg_can_s1,neg_can_s2,all_text, neg_can_s2_no_apicall


def create_con( create_data,MAX_SEQUENCE_LENGTH,MAX_REP_SEQUENCE_LENGTH):
    """

    Args:
        create_data: Set to False to load data from file
        MAX_SEQUENCE_LENGTH: dialogue max length
        MAX_REP_SEQUENCE_LENGTH: sentence max length 20
    """

    num_fake = 1

    if create_data:
        neg_can_S1,neg_can_S2,all_text, neg_can_S2_no_apicall =  create_can() # Get all possible unique sentences
        tokenizer =Tokenizer( )
        tokenizer.fit_on_texts(all_text)
        word_index = tokenizer.word_index # Dictionnary of token:index
        # print("word_index: ", word_index)

        # history ending with customer sentence
        # restaurant reply with eos
        # restaurant reply without eos
        print("create_dialogs ...")
        hist_s1, reply_s1, hist_s2, reply_s2, hist_s3, reply_s3, lite_hist_s1, lite_reply_s1, lite_hist_s2, lite_reply_s2, lite_hist_s3, lite_reply_s3 = create_dialogs()
        test_hist_s2, test_reply_s2, test_hist_s2_OOV, test_reply_s2_OOV = create_test_dialogs()
        
        test_hist_s1 = create_test_s1_interactive()

        train_data = {}
        test_data = {}
        train_data["hist_s1"] = pad_sequences(tokenizer.texts_to_sequences(hist_s1), maxlen=MAX_SEQUENCE_LENGTH, padding='post',value=word_index["eos"])
        train_data["hist_s2"] = pad_sequences(tokenizer.texts_to_sequences(hist_s2), maxlen=MAX_SEQUENCE_LENGTH,padding='post', value=word_index["eos"])
        train_data["hist_s3"] = pad_sequences(tokenizer.texts_to_sequences(hist_s3), maxlen=MAX_SEQUENCE_LENGTH,padding='post', value=word_index["eos"])

        train_data["lite_hist_s1"] = pad_sequences(tokenizer.texts_to_sequences(lite_hist_s1), maxlen=MAX_SEQUENCE_LENGTH, padding='post',value=word_index["eos"])
        train_data["lite_hist_s2"] = pad_sequences(tokenizer.texts_to_sequences(lite_hist_s2), maxlen=MAX_SEQUENCE_LENGTH,padding='post', value=word_index["eos"])
        train_data["lite_hist_s3"] = pad_sequences(tokenizer.texts_to_sequences(lite_hist_s3), maxlen=MAX_SEQUENCE_LENGTH,padding='post', value=word_index["eos"])



        train_data["reply_s1"] = pad_sequences(tokenizer.texts_to_sequences(reply_s1), maxlen=MAX_REP_SEQUENCE_LENGTH,padding='post', value=word_index["eos"])
        train_data["reply_s2"] = pad_sequences(tokenizer.texts_to_sequences(reply_s2), maxlen=MAX_REP_SEQUENCE_LENGTH,padding='post', value=word_index["eos"])
        train_data["reply_s3"] = pad_sequences(tokenizer.texts_to_sequences(reply_s3), maxlen=MAX_REP_SEQUENCE_LENGTH,padding='post', value=word_index["eos"])

        train_data["lite_reply_s1"] = pad_sequences(tokenizer.texts_to_sequences(lite_reply_s1), maxlen=MAX_REP_SEQUENCE_LENGTH,padding='post', value=word_index["eos"])
        train_data["lite_reply_s2"] = pad_sequences(tokenizer.texts_to_sequences(lite_reply_s2), maxlen=MAX_REP_SEQUENCE_LENGTH,padding='post', value=word_index["eos"])
        train_data["lite_reply_s3"] = pad_sequences(tokenizer.texts_to_sequences(lite_reply_s3), maxlen=MAX_REP_SEQUENCE_LENGTH,padding='post', value=word_index["eos"])

        test_data["hist_s2"] = pad_sequences(tokenizer.texts_to_sequences(test_hist_s2), maxlen=MAX_SEQUENCE_LENGTH, padding='post',value=word_index["eos"])
        test_data["hist_s2_OOV"] = pad_sequences(tokenizer.texts_to_sequences(test_hist_s2_OOV), maxlen=MAX_SEQUENCE_LENGTH,padding='post', value=word_index["eos"])

        test_data["reply_s2"] = pad_sequences(tokenizer.texts_to_sequences(test_reply_s2), maxlen=MAX_REP_SEQUENCE_LENGTH,padding='post', value=word_index["eos"])
        test_data["reply_s2_OOV"] = pad_sequences(tokenizer.texts_to_sequences(test_reply_s2_OOV), maxlen=MAX_REP_SEQUENCE_LENGTH,padding='post', value=word_index["eos"])

        test_data["hist_s1"] = pad_sequences(
                tokenizer.texts_to_sequences(test_hist_s1), 
                maxlen=MAX_SEQUENCE_LENGTH,
                padding='post', 
                value=word_index["eos"])
        
        train_data["all_sen_s1"] = pad_sequences(
                tokenizer.texts_to_sequences(neg_can_S1), 
                maxlen=MAX_REP_SEQUENCE_LENGTH,
                padding='post', 
                value=word_index["eos"])

        train_data["all_sen_s2"] = pad_sequences(
                tokenizer.texts_to_sequences(neg_can_S2), 
                maxlen=MAX_REP_SEQUENCE_LENGTH,
                padding='post', 
                value=word_index["eos"])

        train_data["all_sen_s2_no_apicall"] = pad_sequences(
                tokenizer.texts_to_sequences(neg_can_S2_no_apicall), 
                maxlen=MAX_REP_SEQUENCE_LENGTH,
                padding='post', 
                value=word_index["eos"])



        print("s1 dialogues")
        print(hist_s1[0:4])
        print(reply_s1[0:4])

        print("s2 dialogues")
        print(hist_s2[0:4])
        print(reply_s2[0:4])

        print("s3 dialogues")
        print(hist_s3[0:4])
        print(reply_s3[0:4])

        print("test s2 dialogues")
        print(test_hist_s2[0:4])
        print(test_reply_s2[0:4])

        print("test OOV s2 dialogues")
        print(test_hist_s2_OOV[0:4])
        print(test_reply_s2_OOV[0:4])


        embedding_matrix = np.zeros((len(word_index) + 1 ,len(word_index) + 1))
        for word, i in word_index.items():
            embedding_vector = np.eye(len(word_index) + 1)[i]
            embedding_matrix[i] = embedding_vector




        with open('emb_Task1.pickle', 'wb') as output:
            pickle.dump(embedding_matrix, output ,protocol=4)
        with open('wi_Task1.pickle', 'wb') as output:
            pickle.dump(word_index, output, protocol=4)


        with open('Train_Task1.pickle', 'wb') as output:
            pickle.dump(train_data, output, protocol=4)

        with open('Test_Task1.pickle', 'wb') as output:
            pickle.dump(test_data, output, protocol=4)


        print("saving dataset is finished ")

    else:
        with open('emb_Task1.pickle', 'rb') as output:
            embedding_matrix =pickle.load(output)
        with open('wi_Task1.pickle', 'rb') as output:
            word_index = pickle.load(output)


        with open('Train_Task1.pickle', 'rb') as output:
            train_data = pickle.load(output)
        with open('Test_Task1.pickle', 'rb') as output:
            test_data = pickle.load(output)


        print("loading finished ")

    # embedding matrix: ?
    # hist_train: history (..., cust)
    # rep_train: (restaurant reply)
    # Train: (cust, rest)
    # label_train_dialog: is rest in Train true or not
    return embedding_matrix, train_data,test_data, word_index


def add_noise(history, sentence_true, sentence_gen, word_index, all_sentences,
        all_sentences_no_apicall):
    """
    Args:
        all_sentences: token version of all sentences in dialogue
    """
     
    history_noise = []
    sen_noise = []
    #sen_noise = sentence_gen[0,:]  
    #history_noise = history[0,:]

    # Choose randomly another sentence in the dataset instead of
    # the ground truth one
    for i in range(1,sentence_true.shape[0]):
        cansAns = np.random.choice(len(all_sentences), 1, replace=False)
        not_equal_true = np.sum(all_sentences[cansAns[0],:] == sentence_true[i,:])!=np.prod(all_sentences[cansAns[0],:].shape)
        not_equal_gen = np.sum(all_sentences[cansAns[0],:] == sentence_gen[i,:])!=np.prod(all_sentences[cansAns[0],:].shape)

        if not_equal_true and not_equal_gen:
            sen_noise.append(all_sentences[cansAns[0],:])
            history_noise.append(history[i,:])
    
    # NO api call version
    # Choose randomly another sentence in the dataset instead of
    # the ground truth one
    for i in range(1,sentence_true.shape[0]):
        cansAns = np.random.choice(len(all_sentences_no_apicall), 1, replace=False)
        #print(cansAns.shape)
        #print(cansAns[0])
        #print(cansAns[1])

        not_equal_true = np.sum(all_sentences_no_apicall[cansAns[0],:] == sentence_true[i,:])!=np.prod(all_sentences_no_apicall[cansAns[0],:].shape)
        not_equal_gen = np.sum(all_sentences_no_apicall[cansAns[0],:] == sentence_gen[i,:])!=np.prod(all_sentences_no_apicall[cansAns[0],:].shape)

        #print(all_sentences[cansAns[0],:])
        #print(all_sentences[cansAns[0],:].shape)
        #print(sentence_true[i,:])
        #print(sentence_true[i,:].shape) 
        #print(sentence_true_noise[i,:].shape)
        if not_equal_true and not_equal_gen:
            sen_noise.append(all_sentences_no_apicall[cansAns[0],:])
            history_noise.append(history[i,:])
   
    # random sentence of random length
    sen_length = sentence_true.shape[1]
    for i in range(sentence_true.shape[0]):
        # sample random length
        length = np.random.randint(1, sen_length)
        # sample $length random words
        ranWords = np.random.choice(len(word_index), length,replace=False)
        # Pad it with eos
        replace_true = np.ones(sen_length) * word_index.get("eos")
        replace_true[:length] = ranWords
        
        not_equal_true = np.sum(replace_true == sentence_true[i,:])!=np.prod(replace_true.shape)
        not_equal_gen = np.sum(replace_true == sentence_gen[i,:])!=np.prod(replace_true.shape)
        if not_equal_true and not_equal_gen:
            sen_noise.append(replace_true)
            history_noise.append(history[i,:])

    history_noise = np.array(history_noise)
    sen_noise = np.array(sen_noise)

    return (history_noise,
            sen_noise)


if __name__ == "__main__":
    create_con(True, MAX_SEQUENCE_LENGTH=200, MAX_REP_SEQUENCE_LENGTH=20)
    #create_test_s1_interactive()
