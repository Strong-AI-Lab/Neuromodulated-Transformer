'''
File name: Decoder.py
Author: Kobe Knowles
Date created: 05/07/21
Data last modified: 12/08/21
Python Version: 3.6
Tensorflow version: 2
'''

import tensorflow as tf
import numpy as np

import sys
sys.path.append("..")

from models.NMMultiHeadAttention import NMMultiHeadAttention
from models.FeedForwardNetwork import * #FeedForwardNetwork
from models.PositionEncoding import get_angles, positional_encoding
from models.Encoder import EncoderLayer

class DecoderLayer(tf.keras.layers.Layer):
    '''
    Class: DecoderLayer \n
    Description: Implementation of a single decoder layer with support for neuromodulation network context dependant gating. \n
    Attributes:
        max_seq_len: (int) The maximum sequence length of the input. (always pad to this length) \n
        nm_attn: (bool) If we are to gate in a context dependant manner during the calculation of attention. \n
        nm_eol: (bool) If we are to gate in a context dependant manner at the end of each layer. \n
        stop_grad_gating: (bool) True if the gradient is only allowed to backpropagate through the last layer for gating;
                False otherwise all places where gating occurs is allowed to be backpropagated, not just the last layer. \n
        mha1: Multi-head attention 1 (no encoder influence; attends solely to itself). \n
        mha2: (NOTE: has been removed) Multi-head attention 1 (encoder influence; attends the decoder's input to the encoder's output). \n
        ffn: Feed forward network. \n
        layernorm1: Layernormalization layer, occuring after the first multi-head attention layer (mha1). \n
        layernorm2: Layernormalization layer, occuring after the feed-forward-network (ffn). \n
        dropout1: Dropout layer which occurs after the first multi-head attention layer and before the residual connection
            and layer normalization. \n
        dropout2: Dropout layer which occurs after the feed-forward-network layer and before the residual connection
            and layer normalization.
    '''

    def __init__(self, d_model, num_heads, num_layers_gating, dff, max_seq_len, rate=0.1, nm_attn=False, nm_eol=False, rel_pos_emb=True):
        '''
        Function: __init__ \n
        Description: Initializes a decoder layer with the passed parameters. \n
        Input:
            d_model: (int) The dimension of the transformer layer. \n
            num_heads: (int) The number of heads in the multi-head attention component. \n
            dff: (int) The dimension of the feed forward network layer. \n
            max_seq_len: (int) The maximum sequence length to be passed as input. \n
            rate: (float) The dropout rate for dropout layers throughout the layer.
                Defaults to 0.1. \n
            nm_attn: (bool) True if context dependant gating is to occur for the attention calculation (from nm_network);
                False otherwise. Defaults to False. \n
            nm_eol: (bool) True if context dependant gating is to occur at the end of each layer (eol) (from nm_network);
                False otherwise. Defaults to False. \n
            rel_pos_emb: (bool) True if relative position embeddings are to be used; False otherwise (i.e. absolute position embeddings)
        '''
        super(DecoderLayer, self).__init__()

        self.max_seq_len = max_seq_len
        self.nm_attn = nm_attn
        self.nm_eol = nm_eol

        # d_model, num_heads, max_seq_len, nm_gating=False
        self.mha1 = NMMultiHeadAttention(d_model, num_heads, max_seq_len, nm_gating=nm_attn)

        self.ffn = FeedForwardNetwork(init_vanilla_ffn(d_model, dff))

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6) # TODO: check what epsilon actually does.
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        #self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        #self.dropout3 = tf.keras.layers.Dropout(rate)

        # Don't be fooled by the name EncoderLayer, it is the same as a DecoderLayer, but without any gating support.
        # It can act as a decoder with look ahead padding, and an encoder without.
        if self.nm_eol:
            self.eol_mini_tf = [EncoderLayer(d_model, num_heads, dff, max_seq_len, rate,
                                             nm_attn=False, nm_eol=False, rel_pos_emb=rel_pos_emb) for _ in range(num_layers_gating)] # This is a mini transformer.
        if self.nm_attn:
            self.attn_mini_tf = [EncoderLayer(d_model, num_heads, dff, max_seq_len, rate,
                                             nm_attn=False, nm_eol=False, rel_pos_emb=rel_pos_emb) for _ in range(num_layers_gating)] # This is a mini transformer.


    def call(self, x, training, mask, nm_encoder_input=None):
        '''
        Function: call \n
        Description: Overrides parent class's call function (i.e. one run through DecoderLayer). \n
        Input:
            x: (tf.Tensor; [batch_size, max_seq_len(_target), d_model]) Input tensor to the decoder layer. \n
            training: (bool) True if Dropout layers are to be in training mode; False otherwise. \n
            mask: (tf.Tensor) Mask for the multi-head attention layer. \n
            nm_inp_gating_attn: (tf.Tensor; [batch_size, nm_max_seq_len, nm_max_seq_len]) Context dependant gating tensor (in logit form) from the neuromodulated encoder for attn weights. \n
            nm_inp_gating_eol: (tf.Tensor; [batch_size, max_seq_len, d_model] Context dependant gating tensor (in logit form) from the neuromodulated encoder for end of layer (eol) gating.
        Return:
            out3: (tf.Tensor; [batch_size, max_seq_len, d_model])
            attn_weights_block1: (tf.Tensor; [batch_size, num_heads, (max_)seq_len, (max_)seq_len])
            attn_weights_block2: (tf.Tensor; [batch_size, num_heads, (max_)seq_len(_q), (max_)seq_len(_k)])
        '''
        assert self.max_seq_len == x.shape[1], f"x.shape[1] should equal {self.max_seq_len}, got {x.shape[1]}!" # TODO remove.

        if nm_encoder_input is not None:
            if not(self.nm_eol or self.nm_attn): raise Exception(f"One of nm_eol ({self.nm_eol}) or nm_attn ({self.nm_attn}) should be True")
        #elif nm_encoder_input is None:
        else:
            assert not(self.nm_eol and self.nm_attn), f"Both of nm_eol ({self.nm_eol}) or nm_attn ({self.nm_attn}) should be False"

        attention_weights = dict()

        nm_attn_mini_tf = None
        if self.nm_attn:
            nm_attn_mini_tf = nm_encoder_input[:,-x.shape[1]:,:] # with above input checks this
            for m, layer in enumerate(self.attn_mini_tf):
                nm_attn_mini_tf, attn_nm_attn = layer(nm_attn_mini_tf, training, mask) # it shares the same mask (e.g. if look ahead mask then it is applied to both)
                attention_weights[f"attn_nm_attn_layer_{str(m)}"] = attn_nm_attn

        nm_eol_mini_tf = None
        if self.nm_eol:
            nm_eol_mini_tf = nm_encoder_input[:, -x.shape[1]:, :]  # with above input checks this
            for m, layer in enumerate(self.eol_mini_tf):
                nm_eol_mini_tf, attn_nm_eol = layer(nm_eol_mini_tf, training, mask)  # it shares the same mask (e.g. if look ahead mask then it is applied to both)
                attention_weights[f"attn_nm_eol_layer_{str(m)}"] = attn_nm_eol

        x_ = self.layernorm1(x)
        attn1, attn_weights_block1 = self.mha1(x_, x_, x_, nm_inp_gating=nm_attn_mini_tf, mask=mask)
        attention_weights[f"attn_encoder_block1"] = attn_weights_block1
        attn1 = self.dropout1(attn1, training=training)
        out1 = x + attn1

        out1_ = self.layernorm2(out1)
        out2 = self.ffn(out1_) # change to out2 instead of out1 if take input from the encoder.
        out2 = self.dropout2(out2, training=training)
        out2 = out1 + out2

        if nm_eol_mini_tf is not None:
            nm_eol_mini_tf = tf.math.sigmoid(nm_eol_mini_tf)
            out2 = nm_eol_mini_tf * out2

        return out2, attention_weights

class Decoder(tf.keras.layers.Layer):
    '''
    Class: Decoder \n
    Description: Implementation of the decoder in a transformer. \n
    Attributes:
        num_layers: (int) The number of layers of the decoder (i.e. number of decoder layers). \n
        d_model: (int) The dimension of the decoder|transformer. \n
        max_seq_len: (int) The maximum sequence length of the input. (always pad to this length) \n
        mode: (string) Whether or not to process each layer one by one with multiple calls 'one', or all at once 'n_layers'. \n
        counter: (int) Current layer -1 that we are to process next, resets to 0 when finish processing the final layer. \n
        embedding: (tf.keras.layers.Embedding) The embedding layer which convers the input from ids to vectors. \n
        pos_encoding: (tf.Tensor) The positional embedding tensor to append to the input vectors to provide positional information
            (i.e. bring the vector of adjacent words closer to one another and distance words further away). \n
        decoder_layers: (list; DecoderLayer) A list of {num_layer} decoder layers. \n
        dropout: (tf.keras.layers.Dropout) A dropout layer to be applied the the input embeddings after positional encoding
            has been applied.
    '''

    def __init__(self, num_layers, num_layers_gating, d_model, num_heads, dff, max_seq_len, target_vocab_size, max_position_encoding=10000,
                 rate=0.1, nm_attn=False, nm_eol=False, rel_pos_emb=True):
        '''
        Function: __init__ \n
        Description: Initialization of the decoder class. \n
        Input:
            num_layers: (int) The number of decoder layers in the decoder. \n
            d_model: (int) The dimension of the transformer layer. \n
            num_heads: (int) The number of heads in the multi-head attention component. \n
            dff: (int) The dimension of the feed forward network layer. \n
            max_seq_len: (int) The maximum sequence length to be passed as input. \n
            target_vocab_size: (int) The vocabulary size of the target (language). \n
            max_position_encoding: (int) The maximum position encoding to be generated (along sequence length dimension).
                It should greater than max_seq_len. Defaults to 10000. \n
            rate: (float) The dropout rate for dropout layers throughout the layer.
                Defaults to 0.1. \n
            nm_attn: (bool) True if context dependant gating is to occur for the attention calculation (from nm_network);
                False otherwise. Defaults to False. \n
            nm_eol: (bool) True if context dependant gating is to occur at the end of each layer (eol) (from nm_network);
                False otherwise. Defaults to False. \n
            stop_grad_gating: (bool) True if the gradient is only allowed to backpropagate through the last layer for gating;
                False otherwise all places where gating occurs is allowed to be backpropagated, not just the last layer.
        '''
        super(Decoder, self).__init__()

        assert max_position_encoding >= max_seq_len, f"The max_position_encoding ({max_position_encoding}) should be" \
                                                     f"greater than max_seq_len ({max_seq_len})!"

        self.num_layers = num_layers
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.rel_pos_emb = rel_pos_emb

        # possible values are ["n_layers", "one"]
        self.mode = "n_layers"
        self.counter = 0

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_position_encoding, d_model)

        self.decoder_layers = [DecoderLayer(d_model, num_heads, num_layers_gating, dff, max_seq_len,
                                            rate, nm_attn, nm_eol,
                                            rel_pos_emb=rel_pos_emb) for _ in range(num_layers-1)] + \
                              [DecoderLayer(d_model, num_heads, num_layers_gating, dff, max_seq_len,
                                            rate, nm_attn, nm_eol,
                                            rel_pos_emb=rel_pos_emb)] # the gradient should never be stopped here, only potentially in the first N-1 layers.

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, nm_encoder_input=None):
        '''
        Function: call \n
        Description: Overrides the parent class' call function (i.e. run through the decoder). \n
        Input:
            x: (tf.Tensor [int]; [batch_size, max_seq_len(_target)]) Input tensor to the decoder layer. \n
            training: (bool) True if Dropout layers are to be in training mode; False otherwise. \n
            mask: (tf.Tensor) Mask for the multi-head attention layer. \n
            nm_inp_gating_attn: (tf.Tensor; [batch_size, nm_max_seq_len, nm_max_seq_len]) Context dependant gating tensor (in logit form) from the neuromodulated encoder for attn weights. \n
            nm_inp_gating_eol: (tf.Tensor; [batch_size, max_seq_len, d_model] Context dependant gating tensor (in logit form) from the neuromodulated encoder for end of layer (eol) gating.
        Return:
            x: (tf.Tensor; [batch_size, max_seq_len, d_model]) \n
            attention_weights: (dict; tf.Tensor; [batch_size, num_heads, max_seq_len, max_seq_len (can vary for encoder input)])
        '''
        assert x.shape[1] == self.max_seq_len, f"The tensor x should have a dimension 1 (python indices) size of {self.max_seq_len}." \
                                           f"Got {x.shape[1]} instead!"

        seq_len = x.shape[1]

        if self.counter == 0:
            x = self.embedding(x)
            # increase the embedding values so that position encoding doesn't completely remove the meaning of the words.
            # in practice for a dim of 512, 22.63 is what x is multiplied by.
            if not self.rel_pos_emb:
                x *= tf.math.sqrt(tf.cast(self.d_model, tf.dtypes.float32))
                x += self.pos_encoding[:, :seq_len, :]
                x /= tf.math.sqrt(tf.cast(self.d_model, tf.dtypes.float32))
            x = self.dropout(x, training=training)

        attention_weights = dict()
        if self.mode == "n_layers":
            for i in range(self.num_layers):
                x, block1 = self.decoder_layers[i](x, training, mask, nm_encoder_input)
                attention_weights[f'decoder_layer{i+1}_block1'] = block1
                #attention_weights[f'decoder_layer{i+1}_block2'] = block2
            self.reset_counter() # make sure that the counter is 0 for the next time this class is called.
        elif self.mode == "one":
            x, block1 = self.decoder_layers[i](x, training, mask, nm_encoder_input)
            attention_weights[f'decoder_layer{self.counter+1}_block1'] = block1
            #attention_weights[f'decoder_layer{self.counter+1}_block2'] = block2
            self.increment_counter() # note: when at the final layer, this function resets it.
        else: raise Exception(f"Invalid mode parameter. Got mode: {self.mode}! \n"
                              f"It should be equal to \"n_layers\" or \"one\"!")

        return x, attention_weights

    def increment_counter(self):
        self.counter += 1
        if self.counter == self.num_layers:
            self.reset_counter()

    def reset_counter(self):
        self.counter = 0

if __name__ == "__main__":
    d_model, num_heads, dff, max_seq_len, nm_attn, nm_eol = 10, 2, 20, 8, True, True
    num_layers, target_vocab_size = 4, 400
    dec = Decoder(num_layers, d_model, num_heads, dff, max_seq_len, target_vocab_size, max_position_encoding=10000,
                 rate=0.1, nm_attn=True, nm_eol=True)
    dec.mode = "n_layers"
    batch_size = 4

    x = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=400)
    #enc_output = tf.random.uniform((batch_size, max_seq_len+1, d_model)) # +1 to test different encoder max_seq_len...
    training = True
    mask = None
    nm_inp_gating_attn = tf.random.uniform((batch_size, max_seq_len+2, max_seq_len+2))
    nm_inp_gating_eol = tf.random.uniform((batch_size, max_seq_len+2, d_model))
    output, attn_weights = dec(x, training, mask, nm_inp_gating_attn, nm_inp_gating_eol)

    print(f"output: {output} \n"
          f"output.shape: {output.shape}")
    print(f"attn_weights_block1_layer1: {attn_weights['decoder_layer1_block1']}")
    print(f"attn_weights_block2_layer1: {attn_weights['decoder_layer1_block1']}")

