
#How to make S1_trn.txt:

call apicall_set.py on original dataset txt file dialogig...trn.txt -> S1_trn_tmp.txt
Manually replace all <SILENCE> by '' in S1_trn_tmp.txt
call make_S1_inter_data.py -> S1_trn.txt


S1_trn.txt: dialogue training set for 1
S2_trn.txt: dialogue training set for 2
api_call_test.txt: api_call different from the training set to give to S1 to
instantiate dialogue
S2_tst.txt: test dialogue for S2 (same as fb test)
S2_tst_OOV.txt: test dialogue for S2 (same as fb test OOV)
