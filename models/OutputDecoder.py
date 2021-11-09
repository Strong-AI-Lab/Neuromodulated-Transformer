import tensorflow as tf
import numpy as np

import sys
sys.path.append("..")

from models.MultiHeadAttention import MultiHeadAttention
from models.FeedForwardNetwork import FeedForwardNetwork
from models.PositionEncoding import get_angles, positional_encoding
from models.Decoder import DecoderLayer

class OutputDecoder(tf.keras.layers.Layer):
    '''
    Description: Implementation of the OutputDecoder of a transformer (all Z layers).
    '''

    def __init__(self, num_layers, d_model, num_heads, dff, max_seq_len=768,
                 mask_strategy='default', rate=0.1, name=None):
        '''
        :param num_layers: (int) An integer specifying the number of decoder layers.
        :param d_model: (int) An integer specifying the dimension of the decoder layers (and the transformer as a whole).
        :param num_heads: (int) An integer specifying the number of heads in the multi-head attention module.
        :param dff: (int) An integer specifying the dimension of the feed forward layer.
        :param max_seq_len: (int) An integer specifying the maximum sequence length of the input tensors.
        :param mask_strategy: (str) A string specifying the masking strategy.
        :param rate: (float) A floating point number that represents the dropout rate of dropout layers.
        '''
        if name is not None: super(OutputDecoder, self).__init__(name=name)
        else: super(OutputDecoder, self).__init__()

        # this sets a requirement that the name isn't None.
        if name is None: raise Exception(f"The name parameter should not be None but it is!")

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.mask_strategy = mask_strategy
        self.max_seq_len = max_seq_len # this will include the auxiliary tokens and <cls> tokens, however, they may be removed.
        self.rate = rate

        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, mask_strategy, rate=rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate, input_shape=(d_model,))

    def call(self, x, training, mask):
        '''
        :param x: (tf.Tensor; [batch_size, seq_len, d_model]) A tensor representing the input to the decoder component.
        :param training: (bool) A boolean value specifying if we are in training mode for layers which have differnt modes
            for training an non-training.
        :param mask: (tf.Tensor; [batch_size, ..., ..., seq_len]) A tensor representing the mask to be used in multi-head attention.
        :return: (tf.Tensor; [batch_size, max_seq_len, d_model]) |
            (dict of tf.Tensor; [batch_size, num_heads, seq_len, seq_len (can vary)])
        '''

        attention_weights = dict()
        for i, layer in enumerate(self.decoder_layers):
            x, block1 = layer(x=x, training=training, mask=mask)
            attention_weights[f"output_decoder_layer_{i+1}_block1"] = block1

        x = self.dropout(x)
        return x, attention_weights
