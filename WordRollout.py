import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import numpy as np


class WordRollout(object):
    def __init__(self, generator, update_rate):
        # TODO: define generator update strategy. 
        """
        Args:
            generator: It is good to have generator as attribute so that the
            rollout doesn't affect the generator to train.
            update_rate: how to update the rollout gen with respect to the
            generator to train ? 
        """
        self.generator = tf.identity(generator)
        #self.update_rate = update_rate
        #self.vocab_size = self.generator.vocab_size
        #self.batch_size = self.generator.batch_size
        #self.embedding_dim = self.generator.embedding_dim 
        #self.hidden_dim = self.generator.embedding_dim 
        #self.rep_seq_length = self.generator.rep_seq_length 

    def get_reward(self, input_x, mc_num, discriminator):
        """
        Compute rewards for every sub sentence of input_x input_x_{0:t} 
        Args:
            input_x: complete sentence from G
            mc_num: How many times you do MC rollout
            discriminator: D
        """
        rewards = []
        # ~for t in [0,T-1] 
        for t in range(1, self.rep_seq_length): 
            # Make batch of size mc_steps of sentences with the first t tokens
            # of input_x and the rest to 0
            gen_input_t = np.zeros([mc_steps, req_seq_length])
            gen_input_t[:,0:t] = np.tile(input_x[0:t], (mc_steps,1))
            # Ask gen to output a sentence using the first t tokens
            # of the complete sentence input_x
            complete_sentence = generator.get_sentence(gen_input_t)
            
            # Ask disc to reward these sentences
            disc_proba=discriminator.get_rewards(complete_sentence)
            disc_reward=np.array([item[1] for item in disc_proba])
            if i == 0:
                rewards.append(disc_reward)
            else:
                rewards[t - 1] += disc_reward
        # At this point, for t in [0,T-1], rewards[t] = \sum_i R_D(y_{0:t})

        # Reward for the complete sentence
        disc_proba=discriminator.get_rewards(input_x)
        disc_reward = np.array([item[1] for item in disc_proba])
        if i == 0:
            rewards.append(disc_reward)
        else:
            rewards[19] += disc_reward

        # At this point, for t in [0,T-1], rewards[t] = \sum_{rollout} R_D(y_{0:t})
        # reward[T] = R_D(y_{0:T})

        # Average
        rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
        return rewards

    def update_params(self):
        # TODO: how to update the rollout generator ?
        generator = tf.identify(self.generator)
