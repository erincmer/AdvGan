
import Disc1
import readFBTask1

# RNN macros
MAX_SEQ_LENGTH = 200


def pretrain_D1():
    """ 
    Pretrain restaurant disc with the corresponding dataset
    """
    embedding_matrix, x_train, x_val, y_train, y√val, word_ind = readFBTask1.create_con(TRUE, MAX_SEQ_LENGTH)
    D1 = Disc1(MAX_SEQ_LENGTH, word_index, y_val)
    D1.pretrain(x_train, x_val, y_train, y_val)
    # TODO: Save model

def pretrain_D2():
     """ 
    Pretrain customer disc with the corresponding dataset
    """
    # Pretrain customer disc
    embedding_matrix, x_train, x_val, y_train, y√val, word_ind = readFBTask1.create_con(TRUE, MAX_SEQ_LENGTH)
    D1 = Disc1(MAX_SEQ_LENGTH, word_index, y_val)
    D1.pretrain(x_train, x_val, y_train, y_val)
    # TODO: Save model

def pretrain_G1():
    """ 
    Pretrain restaurant gen with the corresponding dataset
    """


def pretrain_G2():
    """ 
    Pretrain cust gen with the corresponding dataset
    """


def adv_train_1():
    """
    Adversarial training for G1 and D1
    """
    # Load G1 and D1
    g_steps = 1
    d_steps = 5
    train_set_size = 5000
    batch√size 50
    max_seq_length = 200
    mc_num = 5

    embedding_matrix, hist_train, hist_val, reply_train, reply_val,
    word_index = readFBTask1Seq2Seq.crate_con(True, max_seq_length)

    embedding_dim = len(word_index+1)

    G1 = Generator()
    D1 = Discriminator(max_seq_length, word_index, embedding_train)
    rollout = WordRollout(G1)
    baseline = Baseline

    for i in range(g_steps):
        for j in range(train_set_size/batch_size):
            X = hist_train[idxTrain[j*batch_size:(j+1)*batch_size],:]
            Y = reply_train[idxTrain[j*batch_size:(j+1)*batch_size],:]
            
            # Shuffle
            X = np.array(X).T
            Y = np.array(Y).T

            feed_dict = {enc_input[t]: X[t] for t in rqnge(seq_length)}
            
            # if batch_size = 1
            # This would be a matrix of embedding_dim * rep_seq_length size
            #¬The first column would correspond to a vector of size
            # embedding_dim representing the word distribution 
            sentence_distro = generator.get_sentence_distro(feed_dict)
            sentence = distro2sentence(sentence_distro)

            # TODO: Specify generator input
            # Sequence of token index
            # Sequence of one hot representation of token
            
            rewards = rollout.get_rewards(sentence, mc_num,
                    discriminator)
            baseline = baseline.get_values(sentence)

            generator.adv_update(sentence, sentence_distro, rewards, baseline)
            # Expected implementation
            #log_p_yt = tf.reduce_sum(tf.one_hot(tf.to_int32(tf.reshape(sentence, [-1])), embedding_dim,1.0, 0.0)
            #            * tf.log(tf.clip_by_value(tf.reshape(sentence_distro, [-1,embedding_dim]),1e-20, 1.0)),1)
            #        
            #unbiased_r = (tf.reshape(self.rewards,[-1]) -tf.reshape(self.baseline[-1]))
            #adv_g_loss = -tf.reduce_sum(log_p_yt * unbiased_r)

    for i in range(d_steps):
        X = hist_train[idxTrain[j*batch_size:(j+1)*batch_size],:]
        Y+ = reply_train[idxTrain[j*batch_size:(j+1)*batch_size],:]
        
        # Shuffle
        X = np.array(X).T
        Y = np.array(Y).T
 
        feed_dict = {enc_input[t]: X[t] for t in rqnge(seq_length)}
        
        sentence_distro = generator.get_sentence_distro(feed_dict)
        Y_ = distro2sentence(sentence_distro)

        # Make 2 batches out of Y_ and Y+
        X1, X1, Y1, Y2 = make_batches(X,Y+,Y_)
        disc.train(X1, Y1, batch_size)
        disc.train(X2, Y2, batch_size)
            

def adv_train_2():


def interactive_train():


