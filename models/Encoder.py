import tensorflow as tf

import sys
sys.path.append("..")

from models.miscellaneous import *

class NMEncoderLayer(tf.keras.layers.Layer):
    '''
    Class: NMEncoderLayer
    Description: Neuromodulated supported implementation of the EncoderLayer class.
    Input:
        d_model: (int) Transformer model dimension.
        dff: (int) Dimension of the feed forward network.
        ffn_dict: (dict) Dictionary of dictionaries containing the parameters for each parallel feed forward layer.
        num_heads: (int) The number of heads for the multi-head attention class.
        neuromodulation: (boolean) True if the encoder is the neuromodulation variant; False if the vanilla transformer.
        max_seq_len: (int)
        rate: (float) Dropout layers dropout probability.
        nm_mha: (boolean)
    '''
    def __init__(self, d_model, dff, ffn_dict, num_heads, neuromodulation, max_seq_len, rate=0.1, nm_mha=False):
        super(NMEncoderLayer, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.dff = dff
        self.ffn_dict = ffn_dict
        self.nm = neuromodulation
        #self.use_nm = use_nm
        self.ffn = NMPWFeewForwardNetwork(self.ffn_dict)
        self.max_seq_len = max_seq_len
        # Passed as input to NMMultiHeadAttention and used to initialize additional parameters for processing
        # nm_attn calculation replacement
        self.nm_mha = nm_mha

        self.mha = NMMultiHeadAttention(self.d_model, self.num_heads, self.max_seq_len, nm_mha=self.nm_mha)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for multi-head attention layer. # axis = -1 by default.
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for feed forward network.

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, nm_output=None, external_memory=False):
        '''
        Function: call
        Description: When called it runs a single encoder layer and returns the output.
        Input:
            x: (tf.Tensor; [batch, input_seq_len, d_model]) Input to the encoder layer.
            training: (boolean) Determines if we are in the training phase or evaluation (testing) phase.
            nm_output: (dict) of tf.Tensors, each of varying dimension.
            mask: (tf.Tensor; [batch_size, ..., ..., seq_len]) Mask to apply to the attention matrix.
                Note: ... means that the exact dimension size can vary from encoder to decoder.
                Encoder: (batch_size, 1, 1, seq_len)
                Decoder 1st block: (batch_size, 1, seq_len, seq_len)
                Decoder 2nd block: (batch_size, 1, 1, seq_len)
                Python broadcasting allows this to work as intended.
            external_memory: (boolean) True if external memory is to be used; False otherwise.
        Return:
            out2: (dict of tf.Tensor; [batch_size, seq_len, d_model])
                note: the last two dimensions can vary depending on the chosen model parameters.
        '''
        attn_logit = None

        try:
            if "attention_nm" in nm_output.keys():
                attn_logit = nm_output["attention_nm"]
        except:
            # Here it would be None so catch the error and continue.
            # Later statements catch this and process accordingly.
            pass

        attn_output = None
        # If taking input from neuromodulated transformer.
        # Otherwise it is set to None.
        if (nm_output is not None) and (attn_logit is not None):
            attn_output, _ = self.mha(x, x, x, attn_logit, mask=mask)  # (batch_size, input_seq_len, d_model)
        else:
            attn_output, _ = self.mha(x, x, x, mask=mask)

        assert attn_output is not None, "Error occurred during call to MNEncoderLayer. attn_output should not be equal to None!"

        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # batch_size, input_seq_len, d_model)

        output_dict = dict()
        if self.nm:
            fnn_output = self.ffn(out1) # a dictionary is returned.
            for i, name in enumerate(self.ffn_dict.keys()):
                output_dict[name] = fnn_output[name]
            out_1_5 = self.dropout2(fnn_output["default"], training=training)
            out2 = self.layernorm2(out1 + out_1_5)
            output_dict["default"] = out2
        else:
            fnn_output = self.ffn(out1)["default"]  # (batch_size, input_seq_len, d_model)
            fnn_output = self.dropout2(fnn_output, training=training)
            out2 = self.layernorm2(out1 + fnn_output)  # (batch_size, input_seq_len, d_model)
            output_dict["default"] = out2

        #TODO in encoder (and decoder) layer put support for neuromoulation at the end of each layer.
        return output_dict

class NMEncoder(tf.keras.layers.Layer):
    '''
    Class: NMEncoder
    Description: Implementation of the Encoder in a Transformer. It has been modified to support neuromodulation.
    '''
    def __init__(self, num_layers, d_model, num_heads, dff, ffn_dict, max_seq_len, neuromodulation,
                input_vocab_size, maximum_position_encoding, rate=0.1, nm_mha=False, start_layer_nm=False):
        super(NMEncoder, self).__init__()

        self.mode = "one"
        # one: one layer at a time, then increment counter.
        # n_layers: pass through all layers, then reset the counter to 0.
        self.d_model = d_model
        self.num_layers = num_layers
        self.nm = neuromodulation
        self.start_layer_nm = start_layer_nm

        self.counter = 0

        # TODO: Note to self. consider changing the embedding to an Encoder instead and
        # modify it so that it can add unseen tokens to its repetoir.
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        # Do below for self_attention layer.
        # d_model, dff, ffn_dict, num_heads, neuromodulation, max_seq_len, rate=0.1, nm_mha=False
        self.enc_layers = [NMEncoderLayer(d_model, dff, ffn_dict, num_heads,
                                          neuromodulation, max_seq_len, rate, nm_mha) for _ in range(num_layers)]
        #if self.start_layer_nm:
        #    self.start_layer_dense = [tf.keras.layers.Dense(d_model) for _ in range(num_layers)] # input is d_model*2
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, nm_output=None, external_memory=False):
        '''
        Function:
        Description:
        Input:
            x: (tf.Tensor; [batch, inp_seq_len])
        Return:
            :
        '''
        # have the option to perform all at once or each individually.
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        #print("x shape before:", x.shape)
        # only run at the beginning layer.
        if self.counter == 0:
            x = self.embedding(x)  # (batch_size, seq_len, d_model)
            #print("x shape after:", x.shape)
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # scale embedding by the sqrt of hidden size.
            '''
            The reason we increase the embedding values before the addition is to make the 
            positional encoding relatively smaller. This means the original meaning in the 
            embedding vector won’t be lost when we add them together.
            '''
            #print(x.shape)
            #print(self.pos_encoding[:, :seq_len, :].shape)
            x += self.pos_encoding[:, :seq_len, :]

            x = self.dropout(x, training=training)
        #print("x shape test:", x.shape)
        if self.mode == "n_layers":
            # note: this mode is incompatible with neuromodulation.
            # the only element in the dictionary will be "default".
            assert not self.nm, "For the neuromodulated encoder the n_layers mode is incompatible."
            for i in range(self.num_layers):
                if i == 0:
                    x = self.enc_layers[i](x, training, mask, nm_output, external_memory)
                else:
                    x = self.enc_layers[i](x["default"], training, mask, nm_output, external_memory)
            #x = {"default":x}
            self.reset_counter() # resets counter to 0.
        elif self.mode == "one":
            x = self.enc_layers[self.counter](x, training, mask, nm_output, external_memory)
            self.increment_counter() # Increment counter by 1. If at the end (last layer) this is reset to zero.
        else:
            raise Exception("Invalid input for {mode} variable!")

        return x  # dictionary of tensors: (batch_size, input_seq_len, d_model) sample dimensions, the last two can change.

    def increment_counter(self):
        self.counter += 1
        if self.counter == self.num_layers:
            #print("\nREACH\n")
            self.reset_counter()

    def reset_counter(self):
        self.counter = 0
