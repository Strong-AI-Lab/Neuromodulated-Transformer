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

    def generate_answer(self, dec_inp, nm_inp, training, nm_mask, dec_mask,
                        pad_tok_id=None, end_tok_id=None, gen_len_max=50):

        batch_size_ = dec_inp.shape[0]

        generation_done = set()
        generated_ids = [[] for _  in range(dec_inp.shape[0])]

        num_aux_toks = abs(nm_inp.shape[1] - dec_inp.shape[1])

        outer_dec_inp = []
        outer_nm_inp = []
        #print(f"outer_dec_inp: {outer_dec_inp}")
        for i in range(gen_len_max):
            #print(f"outer_dec_inp: {outer_dec_inp}")
            if i > 0: #i.e. not the first input which is correct.
                assert len(outer_nm_inp) > 0 and len(outer_dec_inp) > 0, f"Error: One of outer_dec_inp or outer_nm_inp is empty!"
                dec_inp = tf.cast(tf.convert_to_tensor(np.asarray(outer_dec_inp)), dtype=tf.dtypes.int64)
                nm_inp = tf.cast(tf.convert_to_tensor(np.asarray(outer_nm_inp)), dtype=tf.dtypes.int64)
                assert dec_inp.shape[0] == batch_size_, f"new decoder input batch size {dec_inp.shape[0]} doesn't match the correct batch size {batch_size_}" \
                                                        f"\n {print(dec_inp.shape)}"
                nm_mask = create_combined_mask(nm_inp, pad_tok_id, num_aux_toks)  # [batch_size, 1, seq_len, seq_len]
                dec_mask = create_combined_mask(dec_inp, pad_tok_id)

            new_dec_inp = []
            new_nm_inp = []

            for b in range(batch_size_):

                if b in generation_done:
                    new_dec_inp.append(dec_inp[b, :].numpy().tolist()) # no changes here.
                    new_nm_inp.append(nm_inp[b, :].numpy().tolist()) # no changes to be made here.
                    continue # generation is done for this batch item.

                if self.nm_eol or self.nm_attn or len(self.parallel_keys.keys()) > 0:
                    nm_encoder_input, attn_weights_nm_enc, x_dict = self._run_nm_encoder(tf.expand_dims(nm_inp[b,:], axis=0), training,
                                                                                         tf.expand_dims(nm_mask[b,:,:,:], axis=0),
                                                                                         restrictions=self.parallel_keys)  # (output, attn_weights, aux_pred) for ea
                else: nm_encoder_input, attn_weights_nm_enc, x_dict = None, None, None
                # Don't run nm_encoder if both are set to False and there are no parallel_layers.
                # x_dict[key] = (out, attn_weights, aux_pred) aux_pred will not be used, keep as none for support for other alternatives.
                dec_output, attn_weights_dec = self._run_decoder(tf.expand_dims(dec_inp[b,:], axis=0), training, nm_encoder_input,
                                                                 tf.expand_dims(dec_mask[b,:,:,:], axis=0))

                final_output = self.final_layer(dec_output) # (batch_size, seq_len, vocab_size)
                final_output = tf.nn.softmax(final_output, axis=-1) # (batch_size, seq_len, target_vocab_size) batch_size should be 1 here.

                ## get new prediction and add to first pad index... stop when end_tok_id is reached or gen_len_max is reached. (add another outside loop)
                id_prediction, pred_index, end, first_pad_element = self._get_pred_id_helper(final_output, tf.expand_dims(dec_inp[b,:], axis=0), pad_tok_id, end_tok_id) #
                #print(f"id_prediction {id_prediction}\tend_tok_id {end_tok_id}")
                if id_prediction == end_tok_id:
                    #print(f"b: {b} is finished, shouldn't see it processed any more!")
                    new_dec_inp.append(dec_inp[b, :].numpy().tolist()) # no changes made here.
                    new_nm_inp.append(nm_inp[b, :].numpy().tolist()) # no changes made here.
                    generation_done.add(b)
                    continue

                # add new prediction to dec_inp and nm_inp, if no pad token, i.e. at max seq length then remove 1st item to make room...
                # update the padding mask or generate a new one... need to handle graph generation... # can just add to the current mask...
                dec_inp_np = dec_inp[b,:].numpy().tolist() # 1D list (of integers).
                nm_inp_np = nm_inp[b, :].numpy().tolist()  # 1D list (of integers).
                dec_new_input = None
                if end: #move all to the left once and append the prediction to the end.
                    dec_new_input = dec_inp_np[1:] + [id_prediction]
                else:
                    if pred_index is not None:
                        dec_new_input = dec_inp_np[:pred_index+1] + [id_prediction] + dec_inp_np[pred_index+2:]
                    #else: # is handled below with first_pad_element.

                if first_pad_element: # also means pred_index will be None. handles one case above.
                    assert pred_index is None, f"pred_index should be None, got {pred_index}!"
                    dec_new_input = dec_inp # reset to all pad tokens, nothing happens.
                    generation_done.add(b) # all pad tokens, thus just skip. This is a set so no duplicates.

                assert dec_new_input is not None, f"dec_new_input should not be None, something went wrong in the code logic!"

                #print(f"dec_new_input: {dec_new_input}")
                new_dec_inp.append(dec_new_input)
                #print(f"new_nm_inp: {nm_inp_np[:num_aux_toks]+dec_new_input}")
                new_nm_inp.append(nm_inp_np[:num_aux_toks]+dec_new_input)

                generated_ids[b].append(id_prediction) # note: we continue above when reach end token...

            outer_dec_inp = new_dec_inp
            outer_nm_inp = new_nm_inp

            if len(generation_done) >= batch_size_: break # we are done here.
        return generated_ids

    def _get_pred_id_helper(self, prediction, dec_inp, pad_tok_id, end_tok_id):
        assert prediction.shape[0] == 1, f"The first dimension, the batch size should be equal to 1 for the prediction vector, got {prediction.shape[0]}!"
        assert dec_inp.shape[0] == 1, f"The first dimension, the batch size should be equal to 1 for the decoder input, got {dec_inp.shape[0]}!"

        pred_index = None
        for i in range(1,dec_inp.shape[1]):
            if dec_inp[0,i] == pad_tok_id:
                pred_index = i-1 # we get the prediction at the previous token.
                break
        first_pad_element = False
        if dec_inp[0,0] == pad_tok_id: first_pad_element = True


        end = None
        if pred_index is not None:
            id_prediction = tf.argmax(prediction[0,pred_index,:])
            end = False
        else: # is None.
            id_prediction = tf.argmax(prediction[0,-1,:]) # get last seq_len item to get next prediction.
            end = True

        id_prediction = id_prediction.numpy().tolist()
        if isinstance(id_prediction, int): pass
        elif isinstance(id_prediction, list): id_prediction=id_prediction[0]
        else: raise Exception(f"id_prediction is not of type int or list, got {type(id_prediction)}!")

        return id_prediction, pred_index, end, first_pad_element





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