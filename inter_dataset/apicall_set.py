
# Switch api_call to the beginning of the dialogue

import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import pickle


def api_call():
    """
    Returns all possible history finishing with customer sentence, sentences
    from the restaurant with eos, sentences from restaurant without eos
    Args:
        neg_can: All sentence in the dataset (without tags)
        words: dictionnary of token:index
    """
    f = open('dialog-bAbI-tasks_prev/dialog-babi-task1-API-calls-tst-OOV.txt', encoding='utf-8', errors='ignore')
    lines = f.readlines()

    f_new = open('dialog-bAbI-tasks/api_call_first.txt', 'w', encoding='utf-8', errors='ignore')
    counter = 0
    new_d = []
    for xx in lines:
        l = xx
        x = l.split("\t")  # # x:  ['1 hi', 'hello what can i help you with today\n']
        if(len(x)>1):
            #print("xx: ", xx)
            #print("l[0]: ", l[0])
            #input("wait")
            s1 = x[0][x[0].find(" ") + 1:].rstrip()  # sentence 1 'hi'
            s2 = x[1].rstrip()  # sentence 2 'hello what can i help you with today'
            #print("l[0]: ", l[0])
            if l[0]=="1":
                new_d = []
                #input("new dialogue")
            new_d.append(xx)
            
            if s2.split()[0] == 'api_call':
                #input("api_call sentence")
                #print("xx api_call: ", xx)
                to_write = "1 " + s1 + "\t" + s2 + "\n"
                for i in range(len(new_d)-1):
                    x = new_d[i].split("\t")  # # x:  ['1 hi', 'hello what can i help you with today\n']
                    s1 = x[0][x[0].find(" ") + 1:].rstrip()  # sentence 1  'hi'
                    s2 = x[1].rstrip()  # sentence 2 'hello whatcan i help you with today'
                    to_write = to_write + str(i+2) + " " + s1 + "\t" + s2 + "\n"
                
                to_write = to_write + "\n"
                #print(to_write)
                f_new.write(to_write)
                #input("to_write")

    f.close()
    f_new.close()


if __name__ == "__main__":
    api_call()
