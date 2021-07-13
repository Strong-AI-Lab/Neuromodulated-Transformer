'''
File name: FeedForwardNetwork.py
Author: Kobe Knowles
Date created: 05/07/21
Data last modified: 05/07/21
Python Version: 3.6
Tensorflow version: 2
'''

import tensorflow as tf
import numpy as np

# TODO: implement more of these for Reading strategies if needed and  for NM_gating.
def init_vanilla_ffn(d_model, dff):
    '''
    Function: init_vanilla_ffn \n
    Description: Initializes the feed-forward network for a transformer layer. \n
    Input:
        d_model: (int) The dimension of the transformer model. \n
        dff: (int) The dimension of the feed-forward network.
    Return:
         (tf.keras.Sequential)
    '''
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model, activation=None)
    ])

def init_attn_gate_ffn(dff, seq_len):
    '''
    Function: init_attn_gate_ffn \n
    Description: Initializes a feed-forward network for a transformer layer. \n
    Input:
        dff: (int) The dimension of the feed-forward network. \n
        seq_len: (int) The dimension of the last weight (W_2) in the feed-forward network.
            (this should be equal to the maximum sequence length of the nm_encoder {primiary use case})
    Return:
         (tf.keras.Sequential)
    '''
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(seq_len, activation=None)
    ])

def init_metacognition_sequence_ffn(dff):
    '''
    Function: init_metacognition_sequence_ffn \n
    Description: Initializes a feed-forward network for a transformer layer.
        It generates a score between 0 and 1 for each word in the input. \n
    Input:
        dff: (int) The dimension of the feed-forward network. \n
    Return:
         (tf.keras.Sequential)
    '''
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

def init_metacognition_single_ffn(dff):
    '''
    Function: init_metacognition_single_ffn \n
    Description: Initializes a feed-forward network for a transformer layer.
        It generates a (single) score between 0 and 1. \n
    Input:
        dff: (int) The dimension of the feed-forward network. \n
    Return:
         (tf.keras.Sequential)
    '''
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(1, activation='relu'), # TODO: put relu here?
        tf.keras.layers.Permute((2,1)),
        tf.keras.layers.Dense(1, activation='sigmoid') # output is (batch_size, 1, 1) squeeze manually after call.
    ])

class FeedForwardNetwork(tf.keras.layers.Layer):
    '''
    Class: FeedForwardNetwork \n
    Description: Implementation of the feed-forward layer in a transformer. \n
    Attributes:
        ffn: The feed forward network of type tf.keras.Model.
    '''

    def __init__(self, ffn):
        '''
        Function: __init__ \n
        Description: Initializes a feed forward layer with the passed parameters. \n
        Input:
            ffn: (tf.keras.Sequential|tf.keras.Model)
        '''
        super(FeedForwardNetwork, self).__init__()
        self.ffn = ffn

    def call(self, input):
        '''
        Function: call \n
        Description: Overrides parent class's call function (i.e. on run though the feed-forward network layer). \n
        Input:
            input: (tf.Tensor; [batch_size, (max_)seq_len, d_model]
        Return:
            (tf.Tensor; [batch_size, (max_)seq_len, d_model]
        '''
        assert len(input.shape) == 3, f"The number of dimensions of the input should be 3, got {input.shape}!"

        return self.ffn(input)

if __name__ == "__main__":
    d_model, max_seq_len, dff, batch_size = 10, 8, 20, 4
    c = init_vanilla_ffn(d_model, dff)
    ffn = FeedForwardNetwork(c)

    inp =  tf.random.uniform((batch_size, max_seq_len, d_model))
    tens = ffn(inp)

    print(f"ffn output tensor: {tens} \n"
          f"shape: {tens.shape}")
