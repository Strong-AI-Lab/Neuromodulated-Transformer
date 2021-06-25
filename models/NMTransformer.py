import sys
sys.path.append("..")

from models.Encoder import *
from models.Decoder import *
from models.miscellaneous import *
from models.attention_masks import *

class NMTransformer(tf.keras.Model):

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

class NMTransformerDec(tf.keras.Model):

    def __init__(self, num_layers, d_model, num_heads, dff, ffn_dict_dec,
                 max_seq_len_dec, target_vocab_size, pe_target, rate_dec=0.1,
                 nm_mha_dec=False, enc_out=True, neuromodulation=False, nm_net_vocab_size=1000,
                 pe_nm_net=1500, rate_nm_enc=0.1, nm_mha_net=False, ffn_dict_nm={}, max_seq_len_nm=None):
        super(NMTransformerDec, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.neuromodulation = neuromodulation

        # if the nm encoder is to be applied to the decoder.
        self.dec_nm = nm_mha_dec
        self.enc_out = enc_out


        self.decoder = NMDecoder(num_layers, d_model, num_heads, dff, ffn_dict_dec, max_seq_len_dec, target_vocab_size,
                                 pe_target, rate=rate_dec, nm_mha=nm_mha_dec, enc_out=self.enc_out)

        self.nm_encoder = None
        if self.neuromodulation:
            if max_seq_len_nm is None:
                #TODO change this in NMTransformer class as well for the NM network, i.e give nm net its own max seq len.
                assert Exception("max_seq_len_nm shoud not be None!")

            self.nm_encoder = NMEncoder(num_layers, d_model, num_heads, dff, ffn_dict_nm, max_seq_len_nm,
                                        self.neuromodulation, nm_net_vocab_size, pe_nm_net, rate=rate_nm_enc,
                                        nm_mha=nm_mha_net)

        # Normal decoder layer that takes neuromodulation vectors as input for gating.
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, tar, nm_inp_dec, training, look_ahead_mask, dec_padding_mask,
             nm_dec_padding_mask, external_memory=False):

        if self.neuromodulation:
            self.nm_encoder.mode = "one"
        #self.encoder.mode = "one"
        self.decoder.mode = "one"

        # dec_output is already in tensor form, not dictionary.
        # enc_output is set to be None below.
        dec_output, attention_weights = self._run_decoder(tar, nm_inp_dec, None, training, look_ahead_mask,
                                                          dec_padding_mask, nm_dec_padding_mask, external_memory)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len|max_seq_len, target_vocab_size)

        return final_output, attention_weights

    def _run_decoder(self, inp_dec, nm_inp_dec, enc_output, training, look_ahead_mask, dec_padding_mask,
                     nm_dec_padding_mask, external_memory):
        # inp_dec.shape == (batch_size, seq_len_dec, d_model)
        seq_len = inp_dec.shape[1]
        x = None
        nm_output = None
        attn_dict = None
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
