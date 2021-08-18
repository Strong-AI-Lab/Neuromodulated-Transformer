'''
File name: NMTransformer.py
Author: Kobe Knowles
Date created: 05/08/21
Data last modified: 12/08/21
Python version: 3.6
Tensorflow version: 2
'''

import sys
sys.path.append("..")

import tensorflow as tf
import numpy as np

from models.AttentionMasks import *
from models.Decoder import Decoder, DecoderLayer
from models.NMEncoder import *
from models.PositionEncoding import *
from models.NMMultiHeadAttention import NMMultiHeadAttention
from models.FeedForwardNetwork import *

class NMTransformer(tf.keras.Model):
    '''
    Class: NMTransformerDec \n
    Description: Implementation of the decoder only Neuromodulated Transformer. \n
    Attributes:
        num_layers: (int) The number of layers of the decoder (i.e. number of decoder layers). \n
        d_model
        nm_attn
        nm_eol
    '''

    def __init__(self, num_layers_dec, num_layers_nm, num_layers_gating, d_model, num_heads, dff, max_seq_len_dec, max_seq_len_nm,
                 target_vocab_size, nm_vocab_size, max_position_encoding_dec=10000, max_position_encoding_nm=10000,
                 rate=0.1, nm_attn=False, nm_eol=False, rel_pos_emb=True): # TODO: remove max_seq_len_dec and max_seq_len_nm
        '''
        Function: __init__ \n
        Description: Initializes the Neuromodulated Transformer (decoder version) with the passed parameters. \n
        Input:
            pass
        '''
        super(NMTransformer, self).__init__()

        self.d_model = d_model
        self.nm_attn = nm_attn
        self.nm_eol = nm_eol

        self.decoder = Decoder(num_layers_dec, num_layers_gating, d_model, num_heads, dff, max_seq_len_dec, target_vocab_size,
                               max_position_encoding_dec, rate, nm_attn, nm_eol, rel_pos_emb=rel_pos_emb)

        self.nm_encoder = NMEncoder(num_layers_nm, d_model, num_heads, dff, max_seq_len_nm, nm_vocab_size,
                                    max_position_encoding_nm, rate, rel_pos_emb=rel_pos_emb)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, dec_inp, nm_inp, training, nm_mask=None, dec_mask=None):
        '''
        Function: call \n
        Description: Description: Overrides the parent class' call function (i.e. run through the transformer). \n
        Input:
            dec_inp: (tf.Tensor [int]; [batch_size, max_seq_len(_dec)]) Input tensor to the decoder layer. \n
            nm_inp: (tf.Tensor [int]; [batch_size, max_seq_len(_nm)]) Input tensor to the neuromodulated encoder layer. \n
            training: (bool) True if Dropout layers are to be in training mode; False otherwise. \n
            padding_id: (int) The id of the <pad> token, to be used in mask creation. Defaults to 0. \n
            num_aux_tok: (int) The number of auxiliary tokens in the neuromodulated encoder's input. Defaults to 0. \n
        Return:
            final_output:
            attn_weights:
            output_dict:
        '''
        nm_encoder_input, attn_weights_nm_enc = self._run_nm_encoder(nm_inp, training, nm_mask) # (output, attn_weights, aux_pred) for each key.
        dec_output, attn_weights_dec = self._run_decoder(dec_inp, training, nm_encoder_input, dec_mask)

        final_output = self.final_layer(dec_output) # (batch_size, seq_len, vocab_size)

        final_output = tf.nn.softmax(final_output, axis=-1)

        return final_output, attn_weights_dec, attn_weights_nm_enc


    # helper function for the call method. Refer to the call function docstring for description of parameters.
    def _run_nm_encoder(self, nm_inp, training, mask):
        self.nm_encoder.mode = "n_layers"
        return self.nm_encoder(nm_inp, training, mask) # this returns a dictionary of the returned values.

    # helper function for the call method. Refer to the call function docstring for description of parameters.
    def _run_decoder(self, dec_inp, training, nm_encoder_input, mask):
        self.decoder.mode = "n_layers"
        return self.decoder(dec_inp, training, mask, nm_encoder_input=nm_encoder_input)

if __name__ == "__main__":

    num_layers_dec, num_layers_nm = 6, 6
    d_model, num_heads, dff = 100, 10, 200
    max_seq_len_dec, max_seq_len_nm = 5, 6
    target_vocab_size, nm_vocab_size = 200, 199
    batch_size=2
    parallel_layers = {}
    parallel_layers["nm_attn_gate"] = "GateLayerAttn"
    parallel_layers["nm_eol_gate"] = "NMEncoderLayerNoRC"

    transformer = NMTransformerDec(num_layers_dec, num_layers_nm, d_model, num_heads, dff, max_seq_len_dec, max_seq_len_nm,
                 target_vocab_size, nm_vocab_size, max_position_encoding_dec=10000, max_position_encoding_nm=10000,
                 rate=0.1, nm_attn=True, nm_eol=True, parallel_layers=parallel_layers)
    #dec_inp, nm_inp, training, padding_id = 0, num_aux_tok = 0
    dec_inp = tf.random.uniform((batch_size, max_seq_len_dec), minval=0, maxval=24)
    nm_inp = tf.random.uniform((batch_size, max_seq_len_nm), minval=0, maxval=24)
    output = transformer(dec_inp, nm_inp, True, 0, 1)
    print(f"output: {output}") # notice the weird look ahead mask, this is because of global attention of auxiliary tokens in the masking.