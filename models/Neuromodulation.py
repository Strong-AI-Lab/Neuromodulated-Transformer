import sys
sys.path.append("..")

from models.Transformer import *
from models.attention_masks import *

os.environ["CUDA_VISIBLE_DEVICES"] = "4" #"0,1,2,3"

class NMPWFeewForwardNetwork(tf.keras.layers.Layer):
    '''
    Class: NMPWFeedForwardNetwork
    Description: Neuromodulation supported point-wise feed forward network implementation.
    Input:
        ffn_dict: (dict) Dictionary of dictionaries containing the parameters for each parallel feed forward layer.
            note: This dictionary contains d_model and dff in parent class, hence removed.
    '''
    def __init__(self, ffn_dict):
        super(NMPWFeewForwardNetwork, self).__init__()

        self.ffn_dict = ffn_dict

        # holds all of the parallel feed forward layers.
        self.ffn_networks = dict()
        for i, name in enumerate(self.ffn_dict.keys()):
            ffn_ = tf.keras.Sequential()
            num_layers = self.ffn_dict[name]["num_layers"]
            for j in range(num_layers):

                if self.ffn_dict[name]["W_"+str(j+1)+"_transpose?"]:
                    # this essentially transposes the input.
                    ffn_.add(tf.keras.layers.Permute((2,1))) # The batch dimension is left out.

                activation_function = self.ffn_dict[name]["W_"+str(j+1)+"_activation_function"]
                if activation_function != 'none':
                    ffn_.add(tf.keras.layers.Dense(self.ffn_dict[name]["W_"+str(j+1)+""], activation=activation_function))
                else:
                    ffn_.add(tf.keras.layers.Dense(self.ffn_dict[name]["W_" + str(j + 1) + ""]))

            self.ffn_networks[name] = ffn_

        #print(self.ffn_networks)

    def call(self, input):
        '''
        Function: call
        Purpose: Runs the point-wise feed forward network with the given input on all ffn_networks.
        Input:
            input: (tf.Tensor; [batch_size, seq_len, d_model])
        Return:
            output: (dict) Containing the network name and output)
        '''
        assert len(input.shape) == 3, "The number of dimensions must be equal to 3!"

        output = dict()
        for i, name in enumerate(self.ffn_networks.keys()):
            output[name] = self.ffn_networks[name](input)

        return output # dictionary of (batch_size, seq_len_nm|seq_len_q|user_defined, d_model|seq_len_k|user_defined)

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

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for multi-head attention layer.
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
    def __init__(self, d_model, num_heads, dff, ffn_dict, max_seq_len, rate=0.1, nm_mha=False):
        super(NMDecoderLayer, self).__init__()

        # only needed for the encoder - the decoder doesn't turn in to a NM network.
        #self.nm = neuromodulation
        self.max_seq_len = max_seq_len
        self.nm_mha = nm_mha
        self.ffn_dict = ffn_dict

        # currently mha1 is the normal multihead attention and is unchanged.
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = NMMultiHeadAttention(d_model, num_heads, self.max_seq_len, nm_mha=self.nm_mha)
        #self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = NMPWFeewForwardNetwork(self.ffn_dict)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for (masked) multi-head attention layer.
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for multi-head attention layer.
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for feed forward network.

        self.dropout1 = tf.keras.layers.Dropout(rate)
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
            look_ahead_mask: (tf.Tensor; [batch, 1, seq_len, seq_len]) Mask that masks out future words.
            padding_mask: (tf.Tensor; [batch, 1, 1, seq_len]) Padding mask for the decoder.
            nm_attn: (dict) dictionary of tensors, each representing the output of parallel feed forward functions.
            external_memory: (boolean) Determines if external memory is to used in the system.
        Return:
            out3: (tf.Tensor; [batch, target_seq_len, d_model])
            attn_weights_block1: (batch_size, num_heads, tar_seq_len_q, tar_seq_len_k)
            attn_weights_block2: (batch_size, num_heads, tar_seq_len_q, tar_seq_len_k)
        '''
        #enc_output.shape == (batch_size, input_seq_len, d_model)

        attn_logit = None
        try:
            # to catch error if nm_output is None.
            if "attention_nm" in nm_attn.keys():
                attn_logit = nm_attn["attention_nm"]
        except:
            pass

        # first attention block is done as usual.
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask) # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2, attn_weights_block2 = None, None
        if (attn_logit is not None) and (nm_attn is not None):
            attn2, attn_weights_block2 = self.mha2(enc_output, enc_output,
                                                   out1, attn_logit, mask=padding_mask)  # (batch_size, target_seq_len, d_model)
        else:
            attn2, attn_weights_block2 = self.mha2(enc_output, enc_output,
                                                   out1, mask=padding_mask) # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        # No nedd
        output_dict = dict()
        ffn_output = self.ffn(out2)["default"] # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output) # (batch_size, target_seq_len, d_model)
        output_dict["default"] = out3

        return output_dict, attn_weights_block1, attn_weights_block2

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



class NMDecoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, ffn_dict, max_seq_len, target_vocab_size,
                maximum_position_encoding, rate=0.1, nm_mha=False, start_layer_nm=False):
        super(NMDecoder, self).__init__()

        # note, difference here is that it performs on layer at a time and then returns it.

        self.mode = "one"
        # one: one layer at a time, then increment counter.
        # n_layers: pass through all layers, then reset the counter to 0.
        self.num_layers = num_layers
        self.start_layer_nm = start_layer_nm
        self.d_model = d_model
        self.counter = 0

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        #d_model, num_heads, dff, ffn_dict, max_seq_len, rate = 0.1, nm_mha = False
        self.decoder_layers = [NMDecoderLayer(d_model, num_heads, dff, ffn_dict, max_seq_len, rate, nm_mha) for _ in range(self.num_layers)]
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


class NMTransformer(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, ffn_dict_enc, ffn_dict_dec, max_seq_len, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate_enc=0.1, rate_dec=0.1, nm_mha_enc=False,
                 nm_mha_dec=False,
                 neuromodulation=False, nm_net_vocab_size=1000, pe_nm_net=1500, rate_nm_enc=0.1, nm_mha_net=False, ffn_dict_nm={}):
        super(NMTransformer, self).__init__()

        # nm_mha_net if neuromodulation gating in mha attetion in neuromodulation network.

        # num_layers, d_model, num_heads, dff, ffn_dict, max_seq_len, target_vocab_size,
        #                 maximum_position_encoding, rate=0.1, nm_mha=False
        #neuromodulation,  input_vocab_size, maximum_position_encoding, rate = 0.1, nm_mha = False
        self.num_layers = num_layers
        self.d_model = d_model
        self.neuromodulation = neuromodulation
        # if the nm encoder is to be applied to the encoder.
        self.enc_nm = nm_mha_enc
        # if the nm encoder is to be applied to the decoder.
        self.dec_nm = nm_mha_dec

        # Creates vectors for neuromodulation gating.
        self.encoder = NMEncoder(num_layers, d_model, num_heads, dff, ffn_dict_enc, max_seq_len, False,
                input_vocab_size, pe_input, rate=rate_enc, nm_mha=nm_mha_enc)

        # Normal encoder layer that takes neuromodulation vectors as input for gating.
        self.decoder = NMDecoder(num_layers, d_model, num_heads, dff, ffn_dict_dec, max_seq_len, target_vocab_size,
                pe_target, rate=rate_dec, nm_mha=nm_mha_dec)

        # Neuromodulation encoder.
        if self.neuromodulation:
            self.nm_encoder = NMEncoder(num_layers, d_model, num_heads, dff, ffn_dict_nm, max_seq_len, self.neuromodulation,
                    nm_net_vocab_size, pe_nm_net, rate=rate_nm_enc, nm_mha=nm_mha_net)

        # Normal decoder layer that takes neuromodulation vectors as input for gating.
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

        # TODO add ExternalMemory support.
        #TODO: Extra tests (assert) to make sure the parameters are correct/comparible - a high level check.

    def call(self, inp, tar, nm_inp_enc, nm_inp_dec, training, enc_padding_mask, look_ahead_mask, dec_padding_mask,
             nm_padding_mask_enc, nm_dec_padding_mask, external_memory=False):
        # decoder input to call function.
        # x, enc_output, training, look_ahead_mask, padding_mask, nm_attn=None, external_memory=False
        # enocder inpyt to call function.

        if self.neuromodulation:
            self.nm_encoder.mode = "one"
        self.encoder.mode = "one"
        self.decoder.mode = "one"

        # enc_output = (batch_size, max_seq_len, d_model)
        enc_output = self._run_encoder(inp, nm_inp_enc, training, enc_padding_mask, nm_padding_mask_enc, external_memory)

        # dec_output is already in tensor form, not dictionary.
        dec_output, attention_weights = self._run_decoder(tar, nm_inp_dec, enc_output, training, look_ahead_mask,
                                                          dec_padding_mask, nm_dec_padding_mask, external_memory)
        #print(type(dec_output))
        #dec_output = dec_output["default"] # (batch_size, tar_seq_len|max_seq_len, d_model)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len|max_seq_len, target_vocab_size)
        # returns logits and attention weights in the decoder.

        return final_output, attention_weights

    def _run_encoder(self, inp_enc, nm_inp_enc, training, enc_padding_mask, nm_enc_padding_mask, external_memory):

        x = None
        nm_output = None
        for i in range(self.num_layers):

            if self.enc_nm:
                if i == 0:
                    nm_output = self.nm_encoder(nm_inp_enc, training, nm_enc_padding_mask, None, False)
                else:
                    nm_output = self.nm_encoder(nm_output["default"], training, nm_enc_padding_mask, None, False)
            if i == 0:
                x = self.encoder(inp_enc, training, enc_padding_mask, nm_output, external_memory)
            else:
                x = self.encoder(x["default"], training, enc_padding_mask, nm_output, external_memory)

            try:
                if "start_layer_nm" in nm_output.keys() and nm_output is not None and self.encoder.start_layer_nm:
                    #x["default"] = tf.concat([nm_output["start_layer_nm"], x["default"]], 2)
                    #x["default"] = self.encoder.start_layer_dense[i](x["default"])
                    nm_out_prob = tf.nn.softmax(nm_output["start_layer_nm"], -1)
                    x["default"] = x["default"] * nm_out_prob
            except:
                # catches nm_output is None error. It doesn't impact the code at all, we just don't run above.
                pass

        return x["default"] # (batch_size, max_seq_len, d_model)

    def _run_decoder(self, inp_dec, nm_inp_dec, enc_output, training, look_ahead_mask, dec_padding_mask,
                     nm_dec_padding_mask, external_memory):

        x = None
        nm_output = None
        attn_weights = dict()
        for i in range(self.num_layers):

            if self.dec_nm:
                if i == 0:
                    nm_output = self.nm_encoder(nm_inp_dec, training, nm_dec_padding_mask, None, False)
                else:
                    nm_output = self.nm_encoder(nm_output["default"], training, nm_dec_padding_mask, None, False)
            if i == 0:
                # x, enc_output, training, look_ahead_mask, padding_mask, nm_attn=None, external_memory=False
                x, attn_dict = self.decoder(inp_dec, enc_output, training, look_ahead_mask, dec_padding_mask, nm_output, external_memory)
            else:
                x, attn_dict = self.decoder(x["default"], enc_output, training, look_ahead_mask, dec_padding_mask, nm_output, external_memory)

            for key in attn_dict.keys():
                attn_weights[key] = attn_dict[key]

            try:
                if "start_layer_nm" in nm_output.keys() and nm_output is not None and self.decoder.start_layer_nm:
                    #x["default"] = tf.concat([nm_output["start_layer_nm"], x["default"]], 2)
                    #assert x["default"].shape[2] == self.d_model
                    #x["default"] = self.decoder.start_layer_dense[i](x["default"])
                    nm_out_prob = tf.nn.softmax(nm_output["start_layer_nm"], -1)
                    x["default"] = x["default"] * nm_out_prob # perform gating.
            except:
                pass

        return x["default"], attn_weights  # x["default"].shape = (batch_size, max_seq_len, d_model)

class NMMultiHeadAttention(tf.keras.layers.Layer):
    '''
    Class: MultiHeadAttention
    Description: Implementation of multi-head attention to be used in by the Transformer class.
    Inputs:
        d_model: (int) The dattn_weights_block1imension  of the model.
        num_heads: (int) The number of heads in the Multi head attention layer.
        use_nm: (boolean) True if we are replacing the scaled dot product attention with gating from
            associated feed forward layer; otherwise False, calculate as per normal.
    '''
    def __init__(self, d_model, num_heads, max_seq_len=512, nm_mha=False):
        super(NMMultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.nm_mha = nm_mha
        #self.use_nm = use_nm # parameter is useless.
        self.max_seq_len = max_seq_len

        # Must be a multiple of the number of heads.
        assert d_model % self.num_heads == 0

        # the dimension of each individual head.
        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        #self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        if self.nm_mha:
            self.nm_layers_logit = [tf.keras.layers.Dense(self.max_seq_len) for _ in range(self.num_heads)]


        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        '''
        Function: split_heads
        Description: Splits the last dimension into (num_heads, depth).
            Then transpose the tensor to the following permutation, [0,2,1,3].
        Input:
            x: (tf.Tensor: [batch_size, enc_seq_len, d_model]) Input tensor.
            batch_size: (int) Batch size during training.
        Return:
            (tf.tensor; [batch_size, num_heads, enc_seq_len, depth])
        '''
        # num_heads * depth = d_model
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0,2,1,3])

    def call(self, v, k, q, gate_inp_logits=None, mask=None):
        '''
        Function: call
        Purpose: Code to be executed when the keras layer class is called.
        Input:
            v: (tf.Tensor; [batch_size, seq_len_v, word_dim|d_model]) Value input tensor.
            k: (tf.Tensor; [batch_size, seq_len_k, word_dim|d_model]) Key input tensor.
            q: (tf.Tensor; [batch_size, seq_len_q, word_dim|d_model]) Query input tensor.
            gate_inp_logits: (tf.Tensor; [batch_size])
            mask: (tf.Tensor; [batch_size, ..., ..., seq_len]) Mask to apply to the attention matrix.
                Note: ... means that the exact dimension size can vary from encoder to decoder.
                Encoder: (batch_size, 1, 1, seq_len)
                Decoder 1st block: (batch_size, 1, seq_len, seq_len)
                Decoder 2nd block: (batch_size, 1, 1, seq_len)
                Python broadcasting allows this to work as intended.
        Return:
            output: (tf.Tensor; [batch, seq_len_q, d_model])
            attention_weights: (tf.Tensor; [batch_size, num_heads, seq_len_q, seq_len_k])
        '''
        batch_size = tf.shape(q)[0]

        q = self.wq(q) # (batch_size, seq_len, d_model)
        k = self.wk(k) # (batch_size, seq_len, d_model)
        v = self.wv(v) # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size) # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size) # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

        if gate_inp_logits is not None:
            #assert gate_inp_logits is not None, "gate_inp_logits should not be None when use_nm is set to True."
            scaled_attention, attention_weights = self.neuromodulation_attention(q, k, v, gate_inp_logits, mask)
        else:
            scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0,2,1,3]) # (batch_size, seq_len_q, num_heads, v_depth)

        # combine all heads into one dimension
        concat_attention = tf.reshape(scaled_attention, [batch_size, -1, self.d_model]) # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention) # (batch_size, seq_len_q, d_model)

        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask = None):
        '''
        Function: scaled_dot_product_attention
        Purpose: Perform scaled dot-product attention with a query, key, value and a mask as input.
            Specifically calculate the attention weights and multiply them with the value, v.
        Input:
            q: (tf.Tensor; [batch_size, num_heads, seq_len_q, depth])
            k: (tf.Tensor; [batch_size, num_heads, seq_len_k, depth])
            v: (tf.Tensor; [batch_size, num_heads, seq_len_v, depth])
            mask: (tf.Tensor; [batch_size, ..., ..., seq_len]) Mask to apply to the attention matrix.
                Note: ... means that the exact dimension size can vary from encoder to decoder.
                Encoder: (batch_size, 1, 1, seq_len)
                Decoder 1st block: (batch_size, 1, seq_len, seq_len)
                Decoder 2nd block: (batch_size, 1, 1, seq_len)
                Python broadcasting allows this to work as intended.
        Return:
            output: (tf.Tensor; [batch_size, num_heads, seq_len_q, depth_v])
            attention_weights: (tf.Tensor; [batch_size, num_heads, seq_len_q, seq_len_k])
        '''

        matmul_qk = tf.matmul(q, k, transpose_b = True) # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask + -1e9)

        # softmax is normalized on the last axis (seq_len_k) so the scores add to 1.
        # Technically you could do the other axis as long as it is consistent through out training.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v) # (..., seq_len_q, depth_v)

        return output, attention_weights

    def neuromodulation_attention(self, q, k, v, gate_inp_logits, mask = None):
        '''
        Function: neuromodulated_attention
        Purpose: Perform scaled dot-product attention with a query, key, value and a mask as input.
            Specifically calculate the attention weights and multiply them with the value, v.
        Input:
            note: seq_len_q == seq_len_k for all practical purposes (only this is supported currently).
            q: (tf.Tensor; [batch_size, num_heads, seq_len_q, depth])
            k: (tf.Tensor; [batch_size, num_heads, seq_len_k, depth])
            v: (tf.Tensor; [batch_size, num_heads, seq_len_v, depth])
            gate_inp_logits: (tf.Tensor; [batch_size, seq_len_q, seq_len_k])
            mask: (tf.Tensor; [batch_size, ..., ..., seq_len]) Mask to apply to the attention matrix.
                Note: ... means that the exact dimension size can vary from encoder to decoder.
                Encoder: (batch_size, 1, 1, seq_len)
                Decoder 1st block: (batch_size, 1, seq_len, seq_len)
                Decoder 2nd block: (batch_size, 1, 1, seq_len)
                Python broadcasting allows this to work as intended.
        Return:
            output: (tf.Tensor; [batch_size, num_heads, seq_len_q, depth_v])
            attention_weights: (tf.Tensor; [batch_size, num_heads, seq_len_q, seq_len_k])
        '''

        assert q.shape[2] == k.shape[2], "The key's and queries sequence length should be the same."
        assert self.nm_mha, "nm_mha should be set to True instead of False!"

        matmul_qk = tf.matmul(q, k, transpose_b = True) # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        gate_input_head = None
        for i in range(self.num_heads):
            z = self.nm_layers_logit[i](gate_inp_logits)
            #print("z shape \n", z.shape, "\n")
            if gate_input_head is None:
                gate_input_head = tf.expand_dims(z, axis=1)
                #print("test \n", gate_input_head.shape, "\n")
            else:
                gate_input_head = tf.concat([gate_input_head, tf.expand_dims(z, axis=1)], 1)

        #print("\n", gate_input_head.shape,"\n")

        # add the mask to the scaled tensor.
        assert scaled_attention_logits.shape == gate_input_head.shape, f"Dimensions don't match!\nscaled_attention_logits:{scaled_attention_logits.shape}\ngate_input_head:{gate_input_head.shape}"
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
            gate_input_head += (mask * -1e9)

        gate_inp = tf.nn.softmax(gate_input_head, axis=-1)

        attention_weights = gate_inp * scaled_attention_logits # (batch_size, seq_len_q, seq_len_k)

        #TODO: Decide if this is to be here? Do I want a softmax here?
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)

        output = tf.matmul(attention_weights, v) # (..., seq_len_q, depth_v)

        return output, attention_weights

def create_ffn_dict(names, *list_arg):
    '''
    Function: create_ffn_dict
    Description: Initializes and returns dictionary containing parameters for feed forward network initialization.
    Input:
        names: (list of strings) Each string in the list is the name of one parallel feed forward layer.
        *list_arg: (list; varying number of lists) Each list represents parameters for a neural network.
            e.g. position 0: (int) number of layers (e.g. 3 layers in the following three positions.
                 position 1: (list) layer 1 params.
                    [dff, 'relu', (boolean; transpose)]
                 position 2: (list) layer 2 params.
                    [d_model, 'none', (booblean; transpose)]
                 position 3: (list) layer 3 params.
                    [max_seq_len, 'softmax', (boolean; transpose)]
                    note: transpose is done at the beginning of the position.
                .
                .
                .
                position n: (list) layer n params.
                    [dimension (int), activation function (string), transpose (boolean)]
    Return:
        (dict)
    '''
    counter = 0
    for val in list_arg:
        counter += 1

    assert counter == len(names), "Parameter sizes do not match!"
    assert names[0] == "default", "The first parameter in the list names, must be called 'default'."

    ffn_dict = dict()
    for name in names:
        ffn_dict[name] = None

    # keeps track of iteration through names list.
    #counter = 0
    for i, val in enumerate(list_arg):
        dict_ = dict()
        for j in range(len(val)):
            if j == 0:
                #print("\n",val,"\n")
                assert isinstance(val[0], int), "val should be an integer and is not!"
                dict_["num_layers"] = val[0] # this is an integer
                continue

            # val is a list here.
            #print("\n", val, len(val), "\n")
            assert len(val) == dict_["num_layers"]+1, "num_layers parameter doesn't match the input!"
            assert len(val[j]) == 3, "length of this list should be 3!"

            dict_["W_"+str(j)] = val[j][0]
            dict_["W_"+str(j)+"_activation_function"] = val[j][1]
            dict_["W_"+str(j)+"_transpose?"] = val[j][2]

        ffn_dict[names[i]] = dict_

    return ffn_dict

def sample_dict_creation(max_seq_len):
    # Sample initialization.
    names = ["default", "attention_nm", "start_layer_nm"]
    ffn1 = [2, [400, 'relu', False],
            [100, 'none', False]]

    ffn2 = [3, [400, 'relu', False],
            [max_seq_len, 'none', True],
            [max_seq_len, 'none', True]]

    ffn3 = [3, [400, 'relu', False],
            [max_seq_len, 'none', True],
            [100, 'none', True]]

    dct = create_ffn_dict(names, ffn1, ffn2, ffn3)
    return dct

def sample_dict_creation_dec():
    # Sample initialization.
    names = ["default"]
    ffn1 = [2, [400, 'relu', False],
            [100, 'none', False]]

    dct = create_ffn_dict(names, ffn1)
    return dct

if __name__ == "__main__":
    names = ["default", "attention_nm", "start_layer_nm"]
    ffn1 = [2, [20, 'relu', False],
               [100, 'none', False]]

    ffn2 = [3, [10, 'relu', False],
            [20, 'none', True],
            [30, 'none', True]]

    ffn3 = [3, [10, 'relu', False],
            [20, 'none', True],
            [100, 'none', True]]

    dct = create_ffn_dict(names, ffn1, ffn2, ffn3)
    #NMPWFeewForwardNetwork()
    print("\n",dct,"\n")
    ffn = NMPWFeewForwardNetwork(dct)