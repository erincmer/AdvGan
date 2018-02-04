

#Extract all api_call lines from test set to make a test set for interactive
#training

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
    f = open('dialog-bAbI-tasks_prev/dialog-babi-task1-API-calls-tst.txt', encoding='utf-8', errors='ignore')
    lines = f.readlines()

    f_new = open('dialog-bAbI-tasks/api_call_test.txt', 'w', encoding='utf-8', errors='ignore')
    counter = 0
    api_call = []
    for xx in lines:
        l = xx
        x = l.split("\t")  # # x:  ['1 hi', 'hello what can i help you with today\n']
        if(len(x)>1):
            s1 = x[0][x[0].find(" ") + 1:].rstrip()  # sentence 1 'hi'
            s2 = x[1].rstrip()  # sentence 2 'hello what can i help you with today'
            
            if s2.split()[0] == 'api_call':
                if s2 not in api_call:
                    api_call.append(s2)

    for line in api_call:
        to_write = "1 " + line + "\n\n"
        f_new.write(to_write)

    f.close()
    f_new.close()


if __name__ == "__main__":
    api_call()
