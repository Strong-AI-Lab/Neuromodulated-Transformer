import tensorflow as tf
import numpy as np

import sys
sys.path.append("..")

from models.MultiHeadAttention import MultiHeadAttention
from models.FeedForwardNetwork import FeedForwardNetwork
from models.PositionEncoding import get_angles, positional_encoding
from models.Decoder import DecoderLayer

class DecoderLayerPostLN(tf.keras.layers.Layer):
    '''
    Description: Implementation of a transformer decoder layer.
    '''

    def __init__(self, d_model, num_heads, dff, mask_strategy, rate=0.1, name=None):
        '''
        :param d_model: (int) An integer that represents the dimension of the decoder layer (or the transformer as a whole).
        :param num_heads: (int) An integer that represents the number of heads in the multi-head attention component.
        :param dff: (int) An integer that represents the dimension of the feed forward layer.
        :param mask_strategy: (str) A string that represents what mask is to be used in this layer.
        :param rate: (float) A floating point number that represents the dropout rate of dropout layers.
        :param name: (None | str) A NoneType object or string if the name of this layer is needed to be specified.
        '''
        if name is not None: super(DecoderLayerPostLN, self).__init__(name=name)
        else: super(DecoderLayerPostLN, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.mask_strategy = mask_strategy
        self.rate = rate

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, dff, "vanilla")

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, input_shape=(d_model,))
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, input_shape=(d_model,))

        self.dropout1 = tf.keras.layers.Dropout(rate=rate, input_shape=(d_model,))
        self.dropout2 = tf.keras.layers.Dropout(rate=rate, input_shape=(d_model,))


    def call(self, x, training, mask):
        '''
        :param x: (tf.Tensor; [batch_size, seq_len, d_model]) Input to the model in tensor form.
        :param training: (bool) Boolean value representing if the dropout layers are to be in training mode or not.
        :param mask: (tf.Tensor; [batch_size, ..., ..., seq_len]) A tensor representing the mask to be used in multi-head attention.
        :return: (tf.Tensor; [batch_size, seq_len, d_model])
        '''
        # note: that if aux tokens are meant to be removed from the beginning of the sequence, do it before passing as input to this function.

        # block 1
        #x_ = self.layernorm1(x)
        attn1, attention_weights = self.mha(x, x, x, mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1) # (batch_size, seq_len, d_model)

        # block 2
        #out1_ = self.layernorm2(out1)
        out2 = self.ffn(out1)
        out2 = self.dropout2(out2, training=training)
        out2 = self.layernorm2(out1 + out2)

        return out2, attention_weights

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

        self.decoder_layers = [DecoderLayerPostLN(d_model, num_heads, dff, mask_strategy, rate=rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate, input_shape=(d_model,))

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-05, input_shape=(d_model,))

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

        x = self.dropout(x, training=training)
        x = self.layernorm(x)
        return x, attention_weights
