import numpy as np
import os

import tensorflow as tf

from Generator import Generator
from Disc1 import DiscSentence
import readFBTask1Seq2Seq
import toolsSeq2Seq
import headerSeq2Seq

# Set to true to allow debug printing
DEBUG = False

class Baseline(object):
    def __init__(self,batch_size, hidden_dim, rep_seq_length, 
                 max_seq_length, word_index, learning_rate=0.0004):
        """
        Args:
            batch_size: 
            hidden_dim: hidden state dimension
            max_seq_length: max dialogue length
            word_index: token index dictionnary
        """
        self.batch_size = batch_size # 64
        self.rep_seq_length = rep_seq_length # 200
        self.max_seq_length = max_seq_length # 200
        self.hidden_dim = hidden_dim # 250
        self.word_index = word_index #token dict
        self.embedding_dim = len(word_index) + 1
        self.end_token = word_index.get("eos")
        
        # Set embedding
        self.d_embeddings = tf.Variable(
                tf.constant(0.0, shape=[self.embedding_dim, self.embedding_dim]),
                trainable=False, name="W")
        self.embedding_placeholder = tf.placeholder(
                tf.float32, [self.embedding_dim, self.embedding_dim])
        self.embedding_init = self.d_embeddings.assign(self.embedding_placeholder)
 
        # Optimization parameter
        self.grad_clip = 5.0
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
       
        # Data
        self.b_mask = tf.placeholder(tf.float32, shape=[self.batch_size],
                                     name="b_mask") # 
        self.b_truth = tf.placeholder(tf.float32, shape=[self.batch_size],
                                     name="b_truth") # ground truth baseline
        self.enc_inp = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_seq_length],
                                      name="encoderInputs") # input history
        self.input_lengths = tf.reduce_sum(
                tf.to_int32(tf.not_equal(self.enc_inp, self.end_token)), 
                1) # Length of the dialogue without eos
        self.prev_mem = tf.zeros((self.batch_size, self.hidden_dim)) # hidden state
        
        # Embed history
        with tf.device("/cpu:0"):
            processed_x = tf.nn.embedding_lookup(self.d_embeddings, self.enc_inp)

        # Model
        with tf.variable_scope("baseline", reuse=None) as scope:
            self.enc_cell = tf.contrib.rnn.GRUCell(self.hidden_dim)
            # encoder_outputs [batch_size, max_seq_length, cell.output_size]
            # encoder_state [batch_size, cell.state_size] i.e. last
            # hidden state
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                    self.enc_cell, processed_x, self.input_lengths, self.prev_mem)
            out = tf.layers.dense(self.encoder_state,100,activation=tf.nn.relu)
            logits = tf.layers.dense(out, 1) # reward
        
        # Loss definition 
        self.b = tf.squeeze(tf.nn.sigmoid(logits))
        self.b_loss = tf.losses.mean_squared_error(
                self.b_truth, 
                self.b, 
                self.b_mask)  
        
        # Set saver
        self.params = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=self.params)
        
        # Optimization definition
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.gradients = tf.gradients(self.b_loss, self.params)
        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, self.grad_clip)
        self.update = self.optimizer.apply_gradients(
            zip(self.clipped_gradients, self.params))


    def restore_model(self,sess,savepath):
        """
        Save pre trained model
        Args:
            sess: tf session
            savepath: file path of the model
        """
        self.saver.restore(sess, tf.train.latest_checkpoint(savepath))
    
    def save_model(self,sess,savepath):
        """
        Save pre trained model
        Args:
            sess: tf session
            savepath: file path of the model
        """
        self.saver.save(sess, savepath + 'my-model-sentence-sen-1024')

    def assign_emb(self, sess, x):
        """
        Args:
            sess: tf session
            x: 
        """
        sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: x})
    
    def get_baseline(self, sess, X, sentence, word_index):
        """
        Compute baseline
        """
        baseline = np.zeros([self.batch_size, self.rep_seq_length])
        print("eos token: ", word_index['eos'])
        for t in range(1, sentence.shape[1]):
            history_update = np.copy(X)
            gen_input_t = np.ones([headerSeq2Seq.BATCH_SIZE, 
                headerSeq2Seq.REP_SEQ_LENGTH]) * word_index['eos']
            gen_input_t[:,0:t] = sentence[:,0:t]
            
            if DEBUG:
                print("sentence[:, 0:t]: ")
                toolsSeq2Seq.convert_id_to_text(np.array(sentence)[0:1, 0:t], word_index)
                input("wait")
            
            history_update = toolsSeq2Seq.concat_hist_reply(history_update,
                    gen_input_t, word_index)

            # Compute mask
            # Take into account the first eos but mask all the others
            if t==1:
                b_mask = (sentence[:,t-1] !=
                    word_index['eos']).astype(np.float32)
            else:
                b_mask = (sentence[:,t-2] !=
                    word_index['eos']).astype(np.float32)

            b_tmp = sess.run([self.b],
                           feed_dict={self.enc_inp: history_update})
            
            baseline[:, t-1] = np.squeeze(np.array(b_tmp)) * b_mask
            
            if DEBUG:
                print("get_baseline")
                print("t: ", t)
                print("b_mask[0]: ", b_mask[0]) # mask at timestep t for all sentences
                print("t-th token: ", sentence[0,(t-1):t])
                print("t-th token: ",
                        list(word_index.keys())[list(word_index.values()).index(sentence[0,(t-1):t])] )
                print("History: ")
                toolsSeq2Seq.convert_id_to_text(np.array(X)[0:1], word_index)
                print("Sentence")
                toolsSeq2Seq.convert_id_to_text(np.array(sentence)[0:1, :t], word_index)

                print("Baseline: ", np.squeeze(np.array(b_tmp))[0])
                print(baseline[:, t - 1].shape)
                print(np.squeeze(np.array(b_tmp).shape))
                input("wait")
            

        return baseline



    def train_step(self, sess, x, y, b_mask):
        """ 
        Training step on one batch
        Args:
            sess: tf session
            x: 
            y:
            b_mask
        """
        _, loss= sess.run([self.update, self.b_loss],
                           feed_dict={self.enc_inp: x, self.b_truth: y,
                               self.b_mask: b_mask})
        return loss
    
    def train(self, sess, X, sentence, word_index, rewards):
        """ 
        Train on rep_seq_length batch
        """
        loss = 0
        # train

        for t in range(1, sentence.shape[1]):
            history_update = np.copy(X)
            gen_input_t = np.ones([headerSeq2Seq.BATCH_SIZE, 
                headerSeq2Seq.REP_SEQ_LENGTH]) * word_index['eos']
            gen_input_t[:,0:t] = sentence[:,0:t]
            if DEBUG:
                print("sentence[:, 0:t]: ")
                toolsSeq2Seq.convert_id_to_text(np.array(sentence)[0:1, 0:t], word_index)
                input("wait")
            
            history_update = toolsSeq2Seq.concat_hist_reply(history_update,
                    gen_input_t, word_index)

            # Compute mask
            # Take into account the first eos but mask all the others
            if t==1:
                b_mask = (sentence[:,t-1] !=
                    word_index['eos']).astype(np.float32)
            else:
                b_mask = (sentence[:,t-2] !=
                    word_index['eos']).astype(np.float32)

            if DEBUG:
                print("get_baseline")
                print("t: ", t)
                print("b_mask[0]: ", b_mask[0]) # mask at timestep t for all sentences
                print("t-th token: ", sentence[0,(t-1):t])
                print("t-th token: ",
                        list(word_index.keys())[list(word_index.values()).index(sentence[0,(t-1):t])] )
                print("History: ")
                toolsSeq2Seq.convert_id_to_text(np.array(X)[0:1], word_index)
                print("Sentence")
                toolsSeq2Seq.convert_id_to_text(np.array(sentence)[0:1, :t], word_index)

                input("wait")
 
            loss  += self.train_step(sess, history_update, rewards[:,t-1], b_mask)

        return loss





def main():
    # Test pre train
    (embedding_matrix,
            hist_train,
            hist_test,
            reply_train,
            reply_test,
            x_train,
            x_test,
            y_train,
            y_test,
            word_index) = readFBTask1Seq2Seq.create_con(False,headerSeq2Seq.MAX_SEQ_LENGTH)

    EMB_DIM = len(word_index) + 1 # embedding dimension
    END_TOKEN = word_index.get("eos")
    HIST_END_TOKEN = word_index.get("eoh")
    headerSeq2Seq.PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs

    generator = Generator(
            headerSeq2Seq.BATCH_SIZE, 
            EMB_DIM, 
            headerSeq2Seq.HIDDEN_DIM,
            headerSeq2Seq.MAX_SEQ_LENGTH,
            headerSeq2Seq.REP_SEQ_LENGTH,
            headerSeq2Seq.START_TOKEN,
            END_TOKEN,
            HIST_END_TOKEN)
    discriminator = DiscSentence(
            headerSeq2Seq.BATCH_SIZE,  
            headerSeq2Seq.HIDDEN_DIM,
            headerSeq2Seq.MAX_SEQ_LENGTH, 
            word_index, 
            END_TOKEN)
    baseline = Baseline(
            headerSeq2Seq.BATCH_SIZE, 
            headerSeq2Seq.HIDDEN_DIM, 
            headerSeq2Seq.MAX_SEQ_LENGTH, 
            word_index, 
            END_TOKEN,
            learning_rate=0.0004)
 
    # Tf session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    generator.assign_emb(sess,embedding_matrix)
    discriminator.assign_emb(sess,embedding_matrix)
    savepathG = 'GeneratorModel/'  # best is saved here
    savepathD = 'DiscModel/'
    try:
        generator.restore_model(sess, savepathG)
        discriminator.restore_model(sess,savepathD)
    except:
        print("Disc and Gen could not be restored")
        exit(1)
    pass

    idxTrain = np.arange(len(hist_train))
    idxTest = np.arange(len(hist_test))
    for ep in range(10):
        np.random.shuffle(idxTrain)

        for j in range(0, hist_train.shape[0] // headerSeq2Seq.BATCH_SIZE):

            X = hist_train[idxTrain[j*headerSeq2Seq.BATCH_SIZE:(j+1)*headerSeq2Seq.BATCH_SIZE],:]
            Y_train = reply_train[idxTrain[j*headerSeq2Seq.BATCH_SIZE:(j+1)*headerSeq2Seq.BATCH_SIZE],:]
            Y = np.ones((headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.MAX_SEQ_LENGTH)) * word_index['eos']
            gen_proba,sentence, sentence_proba = generator.generate(sess, X, Y)
            rewards = generator.MC_reward(sess, X, sentence, headerSeq2Seq.MC_NUM,
                    discriminator,word_index)

            # train
            for t in range(1, sentence.shape[1]):
                history_update = np.copy(X)
                gen_input_t = np.zeros([headerSeq2Seq.BATCH_SIZE, headerSeq2Seq.MAX_SEQ_LENGTH])
                gen_input_t[:,0:t-1] = sentence[:,0:t-1]
                history_update = toolsSeq2Seq.concat_hist_reply(history_update,
                        gen_input_t, word_index)
                b_mask = (sentence[:,(t-1)] !=
                        word_index['eos']).astype(np.float32)
                # test mask
                print("b_mask.shape: ", b_mask.shape)
                print("b_mask[0,:]: ", b_mask)
                print("sentence tokens: \n", sentence[:,(t-1)])
                toolsSeq2Seq.convert_id_to_text(np.array(sentence)[0:3], word_index)

                loss = baseline.train(sess, history_update, rewards[:,t-1], b_mask)                

            print("Epoch: %d, loss: %.3f" %(ep, loss))

if __name__ == '__main__':
    main()
