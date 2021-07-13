'''
File name: NMEncoder.py
Author: Kobe Knowles
Date created: 05/07/21
Data last modified: 08/07/21
Python Version: 3.6
Tensorflow version: 2
'''

import tensorflow as tf
import numpy as np

import sys
sys.path.append("..")

from models.MultiHeadAttention import MultiHeadAttention
from models.FeedForwardNetwork import * #FeedForwardNetwork
from models.PositionEncoding import get_angles, positional_encoding

class NMEncoderLayer(tf.keras.layers.Layer):
    '''
    Class: NMEncoderLayer \n
    Description: Implementation of a neuromodulation encoder layer. \n
    Attributes:
        max_seq_len: (int) The maximum sequence length of the input. (always pad to this length) \n
        mha: Multi-head attention (attends solely to itself). \n
        ffn: Feed forward network. \n
        layernorm1: Layernormalization layer, occuring after the multi-head attention layer (mha). \n
        layernorm2: Layernormalization layer, occuring after the feed-forward network (ffn). \n
        dropout1: Dropout layer which occurs after the multi-head attention layer and before the residual connection
            and layer normalization. \n
        dropout2: Dropout layer which occurs after the feed-forward layer and before the residual connection
            and layer normalization.
    '''
    def __init__(self, d_model, num_heads, dff, max_seq_len, rate=0.1):
        #TODO: note that this may be similar to EncoderLayer, the differences may be done in the NMEncoder class (i.e. the additional layers at the end.)
        '''
        Function: __init__ \n
        Description: Initializes a neuromodulation encoder layer with the passed parameters. \n
        Input:
            d_model: (int) The dimension of the transformer layer. \n
            num_heads: (int) The number of heads in the multi-head attention component. \n
            dff: (int) The dimension of the feed forward network layer. \n
            max_seq_len: (int) The maximum sequence length to be passed as input. \n
            rate: (float) The dropout rate for dropout layers throughout the layer.
                Defaults to 0.1. \n
        '''
        super(NMEncoderLayer, self).__init__()

        self.max_seq_len = max_seq_len

        self.mha = MultiHeadAttention(d_model, num_heads, max_seq_len, nm_gating=False)
        self.ffn = FeedForwardNetwork(init_vanilla_ffn(d_model, dff))

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for multi-head attention layer.
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for feed forward network.

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        '''
        Function: call \n
        Description: Overrides parent class's call function (i.e. one run through EncoderLayer). \n
        Input:
            x: (tf.Tensor; [batch_size, max_seq_len(_target), d_model]) Input tensor to the decoder layer. \n
            training: (bool) True if Dropout layers are to be in training mode; False otherwise. \n
            mask: (tf.Tensor) Mask for the multi-head attention layer. \n
        Return:
            out2: (tf.Tensor; [batch_size, max_seq_len, d_model])
            attn_weights: (tf.Tensor; [batch_size, num_heads, (max_)seq_len, (max_)seq_len])
        '''
        assert self.max_seq_len == x.shape[1], f"x.shape[1] should equal {self.max_seq_len}, got {x.shape[1]}!"

        attn1, attn_weights = self.mha(x, x, x, nm_inp_gating=nm_inp_gating_attn, mask=mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        out2 = self.ffn(out1)
        out2 = self.dropout2(out2, training=training)
        out2 = self.layernorm2(out1 + out2)

        return out2, attn_weights

class NMEncoder(tf.keras.layers.Layer):
    '''
    Class: NMEncoder \n
    Description: Implementation of the neuromodulation encoder in a transformer. \n
    Attributes:
        num_layers: (int) The number of layers of the encoder (i.e. number of encoder layers). \n
        d_model: (int) The dimension of the encoder|transformer. \n
        max_seq_len: (int) The maximum sequence length of the input. (always pad to this length) \n
        mode: (string) Whether or not to process each layer one by one with multiple calls 'one', or all at once 'n_layers'. \n
        counter: (int) Current layer -1 that we are to process next, resets to 0 when finish processing the final layer. \n
        embedding: (tf.keras.layers.Embedding) The embedding layer which convers the input from ids to vectors. \n
        pos_encoding: (tf.Tensor) The positional embedding tensor to append to the input vectors to provide positional information
            (i.e. bring the vector of adjacent words closer to one another and distance words further away). \n
        encoder_layers: (list; DecoderLayer) A list of {num_layer} decoder layers. \n
        dropout: (tf.keras.layers.Dropout) A dropout layer to be applied the the input embeddings after positional encoding
            has been applied.
    '''
    def __init__(self, num_layers, d_model, num_heads, dff, max_seq_len, input_vocab_size, max_position_encoding=10000,
                 rate=0.1, parallel_layers={}, restrictions=None):
        '''
        Function: __init__ \n
        Description: Initialization of the encoder class. \n
        Input:
            num_layers: (int) The number of encoder layers in the encoder. \n
            d_model: (int) The dimension of the transformer layer. \n
            num_heads: (int) The number of heads in the multi-head attention component. \n
            dff: (int) The dimension of the feed forward network layer. \n
            max_seq_len: (int) The maximum sequence length to be passed as input. \n
            input_vocab_size: (int) The vocabulary size of the input (language). \n
            max_position_encoding: (int) The maximum position encoding to be generated (along sequence length dimension).
                It should greater than max_seq_len. Defaults to 10000. \n
            rate: (float) The dropout rate for dropout layers throughout the layer.
                Defaults to 0.1. \n
            parallel_layers: (dict) Dictionary containing layer names, and the pre-initialized layer.
                sample layer:
                    {
                    "nm_gate_attention": e.g. DecoderLayer
                    "nm_gate_eol": e.g. EncoderLayer
                    }
                Valid keys include nm_gate_attention_lm, nm_gate_eol_lm, nm_gate_attention, nm_gate_eol, + metacognition_layers...
            restrictions: (dict) A set of restrictions for certain auxiliary tokens.
                example:
                    {
                    "<dec>": [<aoint>,...] # this means that if <dec> then don't process the layer that corresponds to metacognition for the aoint.
                    }
        '''
        super(Encoder, self).__init__()

        assert max_position_encoding >= max_seq_len, f"The max_position_encoding ({max_position_encoding}) should be" \
                                                     f"greater than max_seq_len ({max_seq_len})!"

        self.num_layers = num_layers
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # possible values are ["n_layers", "one"]
        self.mode = "n_layers"
        self.counter = 0

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_position_encoding, d_model)

        self.encoder_layers = [EncoderLayer(d_model, num_heads, dff, max_seq_len,
                                            rate, nm_attn, nm_eol) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

        self.parallel_layers = parallel_layers # Parallel layers at the end of self.encoder_layers output (i.e. after they all have been processed).
        self.valid_names = [
                            "nm_attn_gate_lm",
                            "nm_eol_gate_lm",
                            "nm_attn_gate",
                            "nm_eol_gate",
                            "unk_reading_strategy",
                            "highlighting_reading_strategy",
                            "aoi_reading_strategy",
                            "re_read_reading_strategy",
                            "paraphrase_reading_strategy",
                            "summarization_reading_strategy"
                            ]

        if restrictions is None:
            self.restrictions = restrictions
        else:
            self.restrictions = None # TODO baseline restrictions I will initialize myself, they will always be initialized if they are None.

    def call(self, x, training, mask, aux_tokens):
        '''
        Function: call \n
        Description: Overrides the parent class' call function (i.e. run through the encoder). \n
        Input:
            x: (tf.Tensor [int]; [batch_size, max_seq_len(_input)]) Input tensor to the decoder layer. \n
            training: (bool) True if Dropout layers are to be in training mode; False otherwise. \n
            mask: (tf.Tensor) Mask for multi-head attention layer 1. \n
            aux_tokens: (list; string) List of the auxiliary tokens in the current input. Used for checking restrictions.
        Return:
            x_dict: (dict; tf.Tensor; [batch_size, varies, varies]) \n
            attention_weights: (dict; tf.Tensor; [batch_size, num_heads, max_seq_len, max_seq_len])
        '''
        assert x.shape[1] == self.max_seq_len, f"The tensor x should have a dimension 1 (python indices) size of {self.max_seq_len}." \
                                           f"Got {x.shape[1]} instead!"
        seq_len = x.shape[1]

        if self.counter == 0:
            x = self.embedding(x)
            # increase the embedding values so that position encoding doesn't completely remove the meaning of the words.
            # in practice for a dim of 512, 22.63 is what x is multiplied by.
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.dtypes.float32))
            x += self.pos_encoding[:, :seq_len, :]
            # TODO: consider if normalization needs to occur here?
            x = self.dropout(x, training=training)

        attention_weights = dict()
        if self.mode == "n_layers":
            for i in range(self.num_layers):
                x, block1 = self.encoder_layers[i](x, training, mask, nm_inp_gating_attn, nm_inp_gating_eol)
                attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            self.reset_counter()  # make sure that the counter is 0 for the next time this class is called.
        elif self.mode == "one":
            x, block1 = self.encoder_layers[i](x, training, mask, nm_inp_gating_attn, nm_inp_gating_eol)
            attention_weights[f'decoder_layer{self.counter + 1}_block1'] = block1
            self.increment_counter()  # note: when at the final layer, this function resets it.
        else:
            raise Exception(f"Invalid mode parameter. Got mode: {self.mode}! \n"
                            f"It should be equal to \"n_layers\" or \"one\"!")
        x_dict = dict()
        if self.counter == 0: # i.e. we are at the end and it has been reset to zero.
            for key, value in self.parallel_layers.items():
                if not check_restrictions(key, aux_tokens): continue
                x_dict[key] = value(x, training, mask) # TODO when initializeing the layers, make sure their call has no nm_inp...
            if len(x_dict.keys()) == 0: x_dict["default"] = x # i.e. return x if the dictionary is empty.
        return x_dict, attention_weights

    def check_restrictions(self, key, aux_tok): # TODO need a better way to do this...
        if key not in self.valid_names: return False # the key (layer name) needs to be a valid name as defined during initialization.
        for k, v in self.restrictions.items(): # iterate though each key:(list)
            if k in aux_tok: # if k is an auxiliary token in the current input then check restrictions.
                for restr in v: # for each restriction in v if it is equal to key, then return False as the key can't be equal to this restriction.
                    if restr == key: return False
        return True


    def increment_counter(self):
        self.counter += 1
        if self.counter == self.num_layers:
            self.reset_counter()

    def reset_counter(self):
        self.counter = 0