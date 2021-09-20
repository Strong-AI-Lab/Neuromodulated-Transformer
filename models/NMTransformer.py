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
                 rate=0.1, nm_attn=False, nm_eol=False, rel_pos_emb=True, parallel_layers={}):
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

        self.parallel_keys = list()
        for key in parallel_layers.keys():
            self.parallel_keys.append(str(key))

        self.metacognition_aux_loss_layers = dict()
        for key in parallel_layers.keys():
            if key == "unknown_rs":
                self.metacognition_aux_loss_layers[key] = tf.keras.Sequential([
                    tf.keras.layers.Dense(self.d_model, activation='relu', input_shape=(max_seq_len_dec, self.d_model)),
                    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(max_seq_len_dec, self.d_model))
                ]) # purpose is for an auxiliary loss performed with these layers - custom loss that measures difference in
                   # the loss of running both version (with and without reading strategy).
            else:
                self.metacognition_aux_loss_layers[key] = tf.keras.Sequential([
                    tf.keras.layers.Dense(self.d_model, activation='relu', input_shape=(1, self.d_model)),
                    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(1, self.d_model))
                ])

        self.decoder = Decoder(num_layers_dec, num_layers_gating, d_model, num_heads, dff, max_seq_len_dec, target_vocab_size,
                               max_position_encoding_dec, rate, nm_attn, nm_eol, rel_pos_emb=rel_pos_emb)

        self.nm_encoder = NMEncoder(num_layers_nm, d_model, num_heads, dff, max_seq_len_nm, nm_vocab_size,
                                    max_position_encoding_nm, rate, rel_pos_emb=rel_pos_emb, parallel_layers=parallel_layers)

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

        #TODO: take dec_inp and nm_inp and combine into one input... and modify the helper train class to account for this?
        #TODO: NO!!! keep it the same, this will be done in the DataLoader (if pytorch...) Data class that handles the input.
        #TODO: Possibly need to modify the loss function in the helper class depending on the task.

        if self.nm_eol or self.nm_attn or len(self.parallel_keys.keys()) > 0:
            nm_encoder_input, attn_weights_nm_enc, x_dict = self._run_nm_encoder(nm_inp, training, nm_mask,
                                                                                 #restrictions=[])
                                                                                 restrictions=self.parallel_keys) # (output, attn_weights, aux_pred) for each key.
        else: nm_encoder_input, attn_weights_nm_enc, x_dict = None, None, None
        # Don't run nm_encoder if both are set to False and there are no parallel_layers.
        # x_dict[key] = (out, attn_weights, aux_pred) aux_pred will not be used, keep as none for support for other alternatives.
        dec_output, attn_weights_dec = self._run_decoder(dec_inp, training, nm_encoder_input, dec_mask)

        final_output = self.final_layer(dec_output) # (batch_size, seq_len, vocab_size)

        final_output = tf.nn.softmax(final_output, axis=-1)

        x_dict_output, x_dict_output_weights = dict(), dict() # these will be dictionaries?.
        #TODO add logic for x_dict... it is going to be for metacognition... so need to run in a way without reading strategies to see
        #TODO performance increase... + additional layers on top to allow back propagation back through the network.
        #TODO the logic doesn't need to be done as long as the parameters are initialized...
        #TODO this may need to be moved or integrated with above code later...
        #for key in x_dict.keys():
        #    pass

        # x_dict_output will hold the output predictions after a softmax is applied...

        return final_output, attn_weights_dec, attn_weights_nm_enc, x_dict_output, x_dict_output_weights


    # helper function for the call method. Refer to the call function docstring for description of parameters.
    def _run_nm_encoder(self, nm_inp, training, mask, restrictions):
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