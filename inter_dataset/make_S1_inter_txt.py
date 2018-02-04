
# Make S1 seq2seq dataset for interactive training

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
    f = open('S1_trn_tmp.txt', encoding='utf-8', errors='ignore')
    lines = f.readlines()

    f_new = open('S1_trn.txt', 'w', encoding='utf-8', errors='ignore')
    counter = 0
    new_d = []
    for xx in lines:
        l = xx
        x = l.split("\t")  # # x:  ['1 hi', 'hello what can i help you with today\n']
        print("xx: ", xx)
        print("x: ", x)
        #input("wait")

        if(len(x)==1) and l[0]=="1":
            new_d = []
            s1 = x[0][x[0].find(" ") + 1:].rstrip()  # sentence 1 'hi'
            print("api_call: ", s1)
            #input("wait")
            new_d.append(s1)
        elif (len(x)>1):
            #print("l[0]: ", l[0])
            #input("wait")
            s1 = x[0][x[0].find(" ") + 1:].rstrip()  # sentence 1 'hi'
            s2 = x[1].rstrip()  # sentence 2 'hello what can i help you with today'
            #print("l[0]: ", l[0])
            new_d.append(s1)
            new_d.append(s2)
            print("s1: ", s1)
            print("s2: ", s2)
            
            # If restaurant is done
            if s2 == "ok let me look into some options for you":
                print("build new dialogue")
                # Build dialogue
                line_num = 1
                i=0
                to_write = str(line_num) + " " + new_d[i] + "\t" + new_d[i+1]  + "\n"
                i+=2
                line_num+=1
                while i < len(new_d)-1:
                    to_write = to_write + str(line_num) + " " + new_d[i] + "\t" + new_d[i+1]  + "\n"
                    line_num+=1
                    i+=2

                to_write = to_write + str(line_num) + " " + new_d[len(new_d)-1] + "\t" + "<WAIT>\n"
                
                to_write = to_write + "\n"
                print(to_write)
                f_new.write(to_write)
                #input("to_write")

    f.close()
    f_new.close()


if __name__ == "__main__":
    api_call()
