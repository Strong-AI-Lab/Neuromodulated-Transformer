import tensorflow as tf

import sys
sys.path.append("..")

from models.miscellaneous import *

class NMDecoderLayer(tf.keras.layers.Layer):
    '''
    Class: DecoderLayer
    Description: Implementation of a single decoder layer.
    Input:
        d_model: (int) Transformer model dimension.
        num_heads: (int) Number of heads in the multi-head attention layer.
        dff: (int) Feed forward layer dimension.
        ffn_dict: (dict) Contains the parameters for initialization of the point-wise feed forward network.
        max_seq_len: (int) The max sequence length the input could be. To be used in the NMMultiHeadAttenion class.
        rate: (float) Number between 0 and 1 that is used for the dropout rate in dropout layers.
        nm_mha: (boolean) True if NMMultiHeadAttention is using a neuromodulation attention replacement; Falsse otherwise.
    '''
    def __init__(self, d_model, num_heads, dff, ffn_dict, max_seq_len, rate=0.1, nm_mha=False, enc_out=True):
        super(NMDecoderLayer, self).__init__()

        # only needed for the encoder - the decoder doesn't turn in to a NM network.
        #self.nm = neuromodulation
        self.max_seq_len = max_seq_len
        self.nm_mha = nm_mha
        self.ffn_dict = ffn_dict
        self.enc_out = enc_out

        # currently mha1 is the normal multihead attention and is unchanged.
        #self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha1 = NMMultiHeadAttention(d_model, num_heads, self.max_seq_len, nm_mha=self.nm_mha)
        self.mha2 = NMMultiHeadAttention(d_model, num_heads, self.max_seq_len, nm_mha=False) # Add restriction here, this can't be used for neuromodulation.
        #self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = NMPWFeewForwardNetwork(self.ffn_dict)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for (masked) multi-head attention layer.
        #if self.enc_out:
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for multi-head attention layer.
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for feed forward network.

        self.dropout1 = tf.keras.layers.Dropout(rate)
        #if self.enc_out:
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask, nm_attn=None, external_memory=False):
        '''
        Function: call
        Description: Implementation of a single decoder layer.
        Input:
            x: (tf.Tensor; [batch, tar_seq_len, d_model]) Input to the decoder layer.
            enc_output: (tf.Tensor; [batch, input_seq_len, d_model]) The output from the encoder (after all N layers).
            training: (boolean) Determines if we are in the training phase or evaluation (testing) phase.
            look_ahead_mask: (tf.Tensor; [batch, 1, seq_len, seq_len]) Mask that masks out future words (includes padded tokens).
            padding_mask: (tf.Tensor; [batch, 1, 1, seq_len]) Padding mask for the decoder.
            nm_attn: (dict) dictionary of tensors, each representing the output of parallel feed forward functions.
            external_memory: (boolean) Determines if external memory is to used in the system.
        Return:
            -------- out3: (tf.Tensor; [batch, target_seq_len, d_model])
            output_dict: (dict; tf.Tensor())
            attn_weights_block1: (batch_size, num_heads, tar_seq_len_q, tar_seq_len_k)
            attn_weights_block2: (batch_size, num_heads, tar_seq_len_q, tar_seq_len_k)
        '''
        #enc_output.shape == (batch_size, input_seq_len, d_model)

        assert x.shape[1] == self.max_seq_len, "The input shape for x should match the maximum sequence length parameter, " \
                                               "max_seq_len (for the decoder)!"

        seq_len = x.shape[1]
        attn_logit = None
        try:
            # to catch error if nm_output is None.
            if "attention_nm" in nm_attn.keys():
                attn_logit = nm_attn["attention_nm"] # (batch_size, seq_len_nm, seq_len_nm)
                attn_logit = attn_logit[:,-seq_len:,-seq_len:] # this avoids the auxiliary tokens and is necessary.
        except:
            pass

        # first attention block is done as usual.
        if (attn_logit is not None) and (nm_attn is not None):
            attn1, attn_weights_block1 = self.mha1(x, x, x, attn_logit, mask=look_ahead_mask) # (batch_size, target_seq_len, d_model)
        else:
            attn1, attn_weights_block1 = self.mha1(x, x, x, mask=look_ahead_mask) # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        if self.enc_out:
            attn2, attn_weights_block2 = None, None
            attn2, attn_weights_block2 = self.mha2(enc_output, enc_output,
                                                       out1, mask=padding_mask) # (batch_size, target_seq_len, d_model)

            attn2 = self.dropout2(attn2, training=training)
            out2 = self.layernorm2(out1 + attn2)
        else:
            # we don't process above, just set to first layers output.
            attn2 = attn1
            out2 = out1
            attn_weights_block2 = None

        # No need
        output_dict = dict()
        ffn_output = self.ffn(out2)["default"] # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output) # (batch_size, target_seq_len, d_model)
        output_dict["default"] = out3

        return output_dict, attn_weights_block1, attn_weights_block2

class NMDecoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, ffn_dict, max_seq_len, target_vocab_size,
                maximum_position_encoding, rate=0.1, nm_mha=False, start_layer_nm=False, enc_out=True):
        super(NMDecoder, self).__init__()

        # note, difference here is that it performs one layer at a time and then returns it.
        self.mode = "one"
        # one: one layer at a time, then increment counter.
        # n_layers: pass through all layers, then reset the counter to 0.
        self.num_layers = num_layers
        self.start_layer_nm = start_layer_nm
        self.d_model = d_model
        self.counter = 0

        self.enc_out = enc_out

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        #d_model, num_heads, dff, ffn_dict, max_seq_len, rate = 0.1, nm_mha = False
        self.decoder_layers = [NMDecoderLayer(d_model, num_heads, dff, ffn_dict, max_seq_len,
                                              rate, nm_mha, self.enc_out) for _ in range(self.num_layers)]
        #if self.start_layer_nm:
        #    self.start_layer_dense = [tf.keras.layers.Dense(d_model) for _ in range(num_layers)] # input is d_model*2
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask, nm_attn=None, external_memory=False):
        '''
        Function: call
        Description:
        Input:
            x:(tf.Tensor; [batch_size, tar_seq_len, d_model])
            enc_output: (tf.Tensor; [batch_size, inp_seq_len, d_model])
            training: (boolean) True the model is training; False otherwise.
            look_ahead_mask: (tf.Tensor; [batch, 1, seq_len, seq_len]) Mask that masks out future words.
            padding_mask: (tf.Tensor; [batch, 1, 1, seq_len]) Padding mask for the decoder.
            nm_attn: (dict) Dictionary of tensors, each representing the output of parallel feed forward functions.
            external_memory: (boolean) True if external memory is to be used; False otherwise.
        Return:
        '''
        # have the option to perform all at once or each individually.
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        if enc_output is None:
            assert self.enc_out == False, "if enc_output is None, then self.enc_output parameter should be set to False!"
        else:
            assert self.enc_out == True, "if enc_output is not None, then self.enc_output should be True!"

        if self.counter == 0:
            x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # same reason as for encoder.
            x += self.pos_encoding[:, :seq_len, :]

            x = self.dropout(x, training=training)
        # otherwise use x as is.

        if self.mode == "n_layers":
            for i in range(self.num_layers):
                if i == 0:
                    x, block1, block2 = self.decoder_layers[i](x, enc_output, training,
                                                            look_ahead_mask, padding_mask,
                                                            nm_attn, external_memory)
                else:
                    x, block1, block2 = self.decoder_layers[i](x["default"], enc_output, training,
                                                               look_ahead_mask, padding_mask,
                                                               nm_attn, external_memory)

                attention_weights[f'decoder_layer{i+1}_block1'] = block1
                attention_weights[f'decoder_layer{i+1}_block2'] = block2
            #x = {"default":x}
            self.reset_counter()
        elif self.mode == "one":
            x, block1, block2 = self.decoder_layers[self.counter](x, enc_output, training,
                                         look_ahead_mask, padding_mask,
                                         nm_attn, external_memory)
            attention_weights[f'decoder_layer{self.counter+1}_block1'] = block1
            attention_weights[f'decoder_layer{self.counter+1}_block2'] = block2
            self.increment_counter()
        else:
            raise Exception("Invalid input for {mode} variable!")

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

    def increment_counter(self):
        self.counter += 1
        if self.counter == self.num_layers:
            self.reset_counter()

    def reset_counter(self):
        self.counter = 0