import unittest

import sys
sys.path.append("..")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


from models.Neuromodulation import *


class TestTransformerShapes(unittest.TestCase):
    '''
    Class: TestTransformerShapes
    Description: Class that performs unit tests on each Transformer layer. Each test involves
        testing if the shape of the output of each module matches what is expected.

    https://www.tensorflow.org/tutorials/text/transformer is where the test cases/code is taken from.
    '''
    def test_EncoderLayer(self):

        # Test NM transformer
        batch_size = 32
        d_model = 100
        dff = 400
        max_seq_len = 30
        ffn_dict = sample_dict_creation(max_seq_len) # keys are "default", "attention_nm", "start_layer_nm".
        num_heads = 4
        nm = True
        nm_mha = False # ideally this is True for normal TF, and not for neuromodulated transformer.
        enc_layer_nm = NMEncoderLayer(d_model, dff, ffn_dict, num_heads, nm, max_seq_len, nm_mha=nm_mha)

        x = tf.random.uniform((batch_size, max_seq_len+3, d_model)) # should be able to handle larger seq_len - note
        # excessive padding is not needed in nm_transformer.
        mask_inp = tf.random.uniform((batch_size, max_seq_len+3), minval=0, maxval=100, dtype=tf.dtypes.int64)
        mask_ = create_padding_mask(mask_inp)
        output_nm = enc_layer_nm(x, True, mask_, nm_output=None, external_memory=False)

        default_ = output_nm["default"]
        attn_ = output_nm["attention_nm"]
        start_layer_ = output_nm["start_layer_nm"]
        self.assertEqual([default_.shape[0], default_.shape[1], default_.shape[2]],
                         [batch_size, max_seq_len+3, d_model])

        self.assertEqual([attn_.shape[0], attn_.shape[1], attn_.shape[2]],
                         [batch_size, max_seq_len, max_seq_len])

        self.assertEqual([start_layer_.shape[0], start_layer_.shape[1], start_layer_.shape[2]],
                         [batch_size, max_seq_len, d_model])

        # as per normal in vanilla architecture v1.
        nm=False
        nm_mha=False

        enc_layer_nm = NMEncoderLayer(d_model, dff, ffn_dict, num_heads, nm, max_seq_len, nm_mha=nm_mha)

        x = tf.random.uniform((batch_size, max_seq_len, d_model))  # should be able to handle larger seq_len - note
        mask_inp = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=100, dtype=tf.dtypes.int64)
        mask_ = create_padding_mask(mask_inp)
        output = enc_layer_nm(x, True, mask_, nm_output=None, external_memory=False)

        default_ = output["default"]

        self.assertEqual(1, len(output.keys()))
        self.assertEqual([default_.shape[0], default_.shape[1], default_.shape[2]],
                         [batch_size, max_seq_len, d_model])

        # as per normal in vanilla architecture v2.
        nm = False
        nm_mha = False

        enc_layer_nm = NMEncoderLayer(d_model, dff, ffn_dict, num_heads, nm, max_seq_len, nm_mha=nm_mha)

        x = tf.random.uniform((batch_size, max_seq_len, d_model))  # should be able to handle larger seq_len - note
        mask_inp = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=100, dtype=tf.dtypes.int64)
        mask_ = create_padding_mask(mask_inp)
        output = enc_layer_nm(x, True, mask_, nm_output=attn_, external_memory=False) # only different here is nm_output is not None.

        default_ = output["default"]

        self.assertEqual(1, len(output.keys()))
        self.assertEqual([default_.shape[0], default_.shape[1], default_.shape[2]],
                         [batch_size, max_seq_len, d_model])

        # Vanilla transformer with attention gating from neuromodulated transformer.
        nm = False
        nm_mha = True

        enc_layer_nm = NMEncoderLayer(d_model, dff, ffn_dict, num_heads, nm, max_seq_len, nm_mha=nm_mha)

        x = tf.random.uniform((batch_size, max_seq_len, d_model))  # should be able to handle larger seq_len - note
        mask_inp = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=100, dtype=tf.dtypes.int64)
        mask_ = create_padding_mask(mask_inp)
        output = enc_layer_nm(x, True, mask_, nm_output=attn_, external_memory=False) # only different here is nm_output is not None.

        default_ = output["default"]

        self.assertEqual(1, len(output.keys()))
        self.assertEqual([default_.shape[0], default_.shape[1], default_.shape[2]],
                         [batch_size, max_seq_len, d_model])

    def test_DecoderLayer(self):
        #d_model, num_heads, dff, ffn_dict, max_seq_len, rate=0.1, nm_mha=False
        d_model, num_heads, dff = 100, 4, 400
        ffn_dict = sample_dict_creation_dec()
        max_seq_len, rate, nm_mha = 34, 0.1, True

        decoder = NMDecoderLayer(d_model, num_heads, dff, ffn_dict, max_seq_len, rate, nm_mha)

        batch_size = 32

        x = tf.random.uniform((batch_size, max_seq_len, d_model))
        enc_output = tf.random.uniform((batch_size, max_seq_len, d_model))
        training=True
        inp = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=100, dtype=tf.dtypes.int64)
        tar = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=100, dtype=tf.dtypes.int64)
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar)

        nmtf_dict = dict()
        nmtf_dict["attention_nm"] = tf.random.uniform((batch_size, max_seq_len, max_seq_len))

        output, _, _ = decoder(x, enc_output, training, combined_mask, dec_padding_mask, nm_attn=nmtf_dict, external_memory=False)
        default = output["default"]
        self.assertEqual([default.shape[0], default.shape[1], default.shape[2]],
                         [batch_size, max_seq_len, d_model])

        nm_mha = False
        decoder = NMDecoderLayer(d_model, num_heads, dff, ffn_dict, max_seq_len, rate, nm_mha)

        batch_size = 32

        x = tf.random.uniform((batch_size, max_seq_len, d_model))
        enc_output = tf.random.uniform((batch_size, max_seq_len-2, d_model))
        training = True
        inp = tf.random.uniform((batch_size, max_seq_len-2), minval=0, maxval=100, dtype=tf.dtypes.int64)
        tar = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=100, dtype=tf.dtypes.int64)
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar)

        output, _, _ = decoder(x, enc_output, training, combined_mask, dec_padding_mask, nm_attn=None,
                               external_memory=False)
        default = output["default"]
        self.assertEqual([default.shape[0], default.shape[1], default.shape[2]],
                         [batch_size, max_seq_len, d_model])

        decoder = NMDecoderLayer(d_model, num_heads, dff, ffn_dict, max_seq_len, rate, nm_mha)

        batch_size = 32

        x = tf.random.uniform((batch_size, max_seq_len, d_model))
        enc_output = tf.random.uniform((batch_size, max_seq_len, d_model))
        training = True
        inp = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=100, dtype=tf.dtypes.int64)
        tar = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=100, dtype=tf.dtypes.int64)
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar)

        output, _, _ = decoder(x, enc_output, training, combined_mask, dec_padding_mask, nm_attn=None,
                               external_memory=False)
        default = output["default"]
        self.assertEqual([default.shape[0], default.shape[1], default.shape[2]],
                         [batch_size, max_seq_len, d_model])

    def test_MHA(self):
        batch_size = 32
        seq_len = 24
        d_model = 100
        num_heads = 4
        # test neuromodulation output.
        mha = NMMultiHeadAttention(d_model=100, num_heads=num_heads, max_seq_len=seq_len, nm_mha=True)

        v = tf.random.uniform((batch_size, seq_len, d_model))
        k = tf.random.uniform((batch_size, seq_len, d_model))
        q = tf.random.uniform((batch_size, seq_len, d_model))
        gate_inp_logits = tf.random.uniform((batch_size, seq_len, seq_len))
        output, attn_weights = mha(v, k, q, gate_inp_logits)

        self.assertEqual([output.shape[0], output.shape[1], output.shape[2]],
                         [batch_size, seq_len, d_model])
        self.assertEqual([attn_weights.shape[0], attn_weights.shape[1], attn_weights.shape[2], attn_weights.shape[3]],
                         [batch_size, num_heads, seq_len, seq_len])

        # test without neuromodeulation.
        mha = NMMultiHeadAttention(d_model=100, num_heads=4, max_seq_len=seq_len, nm_mha=False)

        v = tf.random.uniform((batch_size, seq_len, d_model))
        k = tf.random.uniform((batch_size, seq_len, d_model))
        q = tf.random.uniform((batch_size, seq_len, d_model))
        #gate_inp_logits = tf.random.uniform((batch_size, seq_len, seq_len))
        output, attn_weights = mha(v, k, q)

        self.assertEqual([output.shape[0], output.shape[1], output.shape[2]],
                         [batch_size, seq_len, d_model])
        self.assertEqual([attn_weights.shape[0], attn_weights.shape[1], attn_weights.shape[2], attn_weights.shape[3]],
                         [batch_size, num_heads, seq_len, seq_len])

    def test_Encoder(self):
        #num_layers, d_model, num_heads, dff, fnn_dict, max_seq_len, neuromodulation,
        #input_vocab_size, maximum_position_encoding, rate = 0.1, nm_mha = False)
        '''
        Test neuromodulation encoder.
            neuromodulation: True
            nm_mha = False
        '''
        num_layers, d_model, num_heads, dff = 3, 100, 4, 400
        max_seq_len, batch_size, neuromodulation, input_vocab_size, maximum_position_encoding = 34, 32, True, 8500, 10000
        fnn_dict = sample_dict_creation(max_seq_len)
        rate, nm_mha = 0.1, False

        encoder = NMEncoder(num_layers, d_model, num_heads, dff, fnn_dict, max_seq_len, neuromodulation,
                            input_vocab_size, maximum_position_encoding, rate, nm_mha)
        #x, training, mask, nm_output = None, external_memory = False
        #x = tf.random.uniform((batch_size, max_seq_len, d_model))
        x = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=100, dtype=tf.dtypes.int64)
        mask_ = create_padding_mask(x)
        nm_output = None # because this is the neuromodulation transformer.
        external_memory = False

        #output = None
        encoder.mode = "n_layers"
        with self.assertRaises(AssertionError):
            #print("x.shape", x.shape)
            output = encoder(x, True, mask_, nm_output, external_memory)
        #default = output["default"]
        #self.assertEqual([default.shape[0], default.shape[1], default.shape[2]],
        #                 [batch_size, max_seq_len, d_model])

        encoder.mode = "one"
        for i in range(num_layers):
            if i == 0:
                x = encoder(x, True, mask_, nm_output, external_memory)
            else:
                x = encoder(x["default"], True, mask_, nm_output, external_memory)
            default = x["default"]
            attn = x["attention_nm"]
            start_ = x["start_layer_nm"]

            self.assertEqual([default.shape[0], default.shape[1], default.shape[2]],
                             [batch_size, max_seq_len, d_model])
            self.assertEqual([attn.shape[0], attn.shape[1], attn.shape[2]],
                             [batch_size, max_seq_len, max_seq_len])
            self.assertEqual([start_.shape[0], start_.shape[1], start_.shape[2]],
                             [batch_size, max_seq_len, d_model])

        '''
        Test encoder that takes input from neuromodulated encoder.
            neuromodulation: False
            nm_mha = True
        '''
        neuromodulation = False
        nm_mha = True
        encoder = NMEncoder(num_layers, d_model, num_heads, dff, fnn_dict, max_seq_len, neuromodulation,
                            input_vocab_size, maximum_position_encoding, rate, nm_mha)

        nm_output = dict()
        nm_output["default"] = tf.random.uniform((batch_size, max_seq_len, d_model))
        nm_output["attention_nm"] = tf.random.uniform((batch_size, max_seq_len, max_seq_len))
        nm_output["start_layer_nm"] = tf.random.uniform((batch_size, max_seq_len, d_model))
        x = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=100, dtype=tf.dtypes.int64)
        mask_ = create_padding_mask(x)

        encoder.mode = "n_layers"
        output = encoder(x, True, mask_, nm_output, external_memory)
        default = output["default"]
        self.assertEqual([default.shape[0], default.shape[1], default.shape[2]],
                         [batch_size, max_seq_len, d_model])
        self.assertEqual(len(output.keys()), 1)

        #encoder = NMEncoder(num_layers, d_model, num_heads, dff, fnn_dict, max_seq_len, neuromodulation,
        #                    input_vocab_size, maximum_position_encoding, rate, nm_mha)
        x = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=100, dtype=tf.dtypes.int64)
        mask_ = create_padding_mask(x)
        encoder.mode = "one"
        for i in range(num_layers):
            if i == 0:
                x = encoder(x, True, mask_, nm_output, external_memory)
            else:
                x = encoder(x["default"], True, mask_, nm_output, external_memory)

            default = x["default"]

            self.assertEqual([default.shape[0], default.shape[1], default.shape[2]],
                             [batch_size, max_seq_len, d_model])

        '''
        Test encoder that takes input from neuromodulated encoder.
            neuromodulation: False
            nm_mha = False
        '''
        neuromodulation = False
        nm_mha = False
        encoder = NMEncoder(num_layers, d_model, num_heads, dff, fnn_dict, max_seq_len, neuromodulation,
                            input_vocab_size, maximum_position_encoding, rate, nm_mha)

        nm_output = None
        #nm_output["default"] = tf.random.uniform((batch_size, max_seq_len, d_model))
        #nm_output["attention_nm"] = tf.random.uniform((batch_size, max_seq_len, max_seq_len))
        #nm_output["start_layer_nm"] = tf.random.uniform((batch_size, max_seq_len, d_model))
        x = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=100, dtype=tf.dtypes.int64)
        mask_ = create_padding_mask(x)

        encoder.mode = "n_layers"
        output = encoder(x, True, mask_, nm_output, external_memory)
        default = output["default"]
        self.assertEqual([default.shape[0], default.shape[1], default.shape[2]],
                         [batch_size, max_seq_len, d_model])
        self.assertEqual(len(output.keys()), 1)

        # encoder = NMEncoder(num_layers, d_model, num_heads, dff, fnn_dict, max_seq_len, neuromodulation,
        #                    input_vocab_size, maximum_position_encoding, rate, nm_mha)
        x = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=100, dtype=tf.dtypes.int64)
        mask_ = create_padding_mask(x)
        encoder.mode = "one"
        for i in range(num_layers):
            if i == 0:
                x = encoder(x, True, mask_, nm_output, external_memory)
            else:
                x = encoder(x["default"], True, mask_, nm_output, external_memory)

            default = x["default"]

            self.assertEqual([default.shape[0], default.shape[1], default.shape[2]],
                             [batch_size, max_seq_len, d_model])


        #TODO: Add support for putting start_layer_nm at the begining of next layer in this unit test section.
        #TODO: This will be tested in test_Transformer function.

    def test_Decoder(self):
        #num_layers, d_model, num_heads, dff, ffn_dict, max_seq_len, target_vocab_size,
        #maximum_position_encoding, rate = 0.1, nm_mha = False

        '''
        Decoder, neuromodulated input (for attention replacement)
        '''
        num_layers, d_model, num_heads, dff, max_seq_len, batch_size = 3, 100, 4, 400, 34, 32
        ffn_dict = sample_dict_creation(max_seq_len)
        target_vocab_size, maximum_position_encoding, rate, nm_mha =  8500, 10000, 0.1, True

        decoder = NMDecoder(num_layers, d_model, num_heads, dff, ffn_dict, max_seq_len, target_vocab_size,
                maximum_position_encoding, rate, nm_mha)

        x = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=100, dtype=tf.dtypes.int64)
        inp = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=100, dtype=tf.dtypes.int64)
        tar = x
        nm_attn = tf.random.uniform((batch_size, max_seq_len, max_seq_len))
        external_memory = False
        enc_output = tf.random.uniform((batch_size, max_seq_len, d_model))
        training = True

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar)
        #x, enc_output, training, look_ahead_mask, padding_mask, nm_attn = None, external_memory = False
        decoder.mode = "n_layers"
        x_dict, attn_dict = decoder(x, enc_output, training, combined_mask, dec_padding_mask, nm_attn, False)
        default = x_dict["default"]
        self.assertEqual([default.shape[0], default.shape[1], default.shape[2]],
                         [batch_size, max_seq_len, d_model])
        #Note: assume attn_dict works correctly, possibly add unit test here later.

        x = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=100, dtype=tf.dtypes.int64)
        inp = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=100, dtype=tf.dtypes.int64)
        tar = x
        nm_attn = tf.random.uniform((batch_size, max_seq_len, max_seq_len))
        external_memory = False
        enc_output = tf.random.uniform((batch_size, max_seq_len, d_model))
        training = True
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar)
        attn_weights = dict()
        decoder.mode = "one"
        for i in range(num_layers):
            if i == 0:
                x, attn_dict = decoder(x, enc_output, training, combined_mask, dec_padding_mask, nm_attn, False)
                for key in attn_dict.keys():
                    attn_weights[key] = attn_dict[key]
            else:
                x, attn_dict = decoder(x["default"], enc_output, training, combined_mask, dec_padding_mask, nm_attn, False)
                for key in attn_dict.keys():
                    attn_weights[key] = attn_dict[key]
            self.assertEqual([x["default"].shape[0], x["default"].shape[1], x["default"].shape[2]],
                             [batch_size, max_seq_len, d_model])
            self.assertTrue(len(attn_weights.keys()) == (i+1)*2)

        '''
        Decoder, no neuromodulated input (for attention replacement)
        '''
        num_layers, d_model, num_heads, dff, max_seq_len, batch_size = 3, 100, 4, 400, 34, 32
        ffn_dict = sample_dict_creation(max_seq_len)
        target_vocab_size, maximum_position_encoding, rate, nm_mha = 8500, 10000, 0.1, False

        decoder = NMDecoder(num_layers, d_model, num_heads, dff, ffn_dict, max_seq_len, target_vocab_size,
                            maximum_position_encoding, rate, nm_mha)

        x = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=100, dtype=tf.dtypes.int64)
        inp = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=100, dtype=tf.dtypes.int64)
        tar = x
        nm_attn = None  # because this is the neuromodulation transformer.
        external_memory = False
        enc_output = tf.random.uniform((batch_size, max_seq_len, d_model))
        training = True

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar)
        # x, enc_output, training, look_ahead_mask, padding_mask, nm_attn = None, external_memory = False
        decoder.mode = "n_layers"
        x_dict, attn_dict = decoder(x, enc_output, training, combined_mask, dec_padding_mask, nm_attn, False)
        default = x_dict["default"]
        self.assertEqual([default.shape[0], default.shape[1], default.shape[2]],
                         [batch_size, max_seq_len, d_model])
        # Note: assume attn_dict works correctly, possibly add unit test here later.

        x = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=100, dtype=tf.dtypes.int64)
        inp = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=100, dtype=tf.dtypes.int64)
        tar = x
        nm_attn = None  # because this is the neuromodulation transformer.
        external_memory = False
        enc_output = tf.random.uniform((batch_size, max_seq_len, d_model))
        training = True
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar)
        attn_weights = dict()
        decoder.mode = "one"
        for i in range(num_layers):
            if i == 0:
                x, attn_dict = decoder(x, enc_output, training, combined_mask, dec_padding_mask, nm_attn, False)
                for key in attn_dict.keys():
                    attn_weights[key] = attn_dict[key]
            else:
                x, attn_dict = decoder(x["default"], enc_output, training, combined_mask, dec_padding_mask, nm_attn,
                                       False)
                for key in attn_dict.keys():
                    attn_weights[key] = attn_dict[key]
            self.assertEqual([x["default"].shape[0], x["default"].shape[1], x["default"].shape[2]],
                             [batch_size, max_seq_len, d_model])
            self.assertTrue(len(attn_weights.keys()) == (i + 1) * 2)

    def test_Transformer_nm(self):
        #num_layers, d_model, num_heads, dff, ffn_dict_enc, ffn_dict_dec, max_seq_len, input_vocab_size,
        #         target_vocab_size, pe_input, pe_target, rate_enc=0.1, rate_dec=0.1, nm_mha_enc=False,
        #         nm_mha_dec=False,
        #         neuromodulation=False, nm_net_vocab_size=1000, pe_nm_net=1500, rate_nm_enc=0.1, nm_mha_net=False, ffn_dict_nm={}
        num_layers, d_model, num_heads, dff, max_seq_len = 3, 100, 4, 400, 34
        names = ["default"]
        ffn1 = [2, [400, 'relu', False],
                [100, 'none', False]]
        ffn_dict_enc = create_ffn_dict(names, ffn1)
        ffn_dict_dec = create_ffn_dict(names, ffn1)
        names = ["default", "attention_nm", "start_layer_nm"]
        ffn1 = [2, [400, 'relu', False],
                [100, 'none', False]]

        ffn2 = [3, [400, 'relu', False],
                [max_seq_len, 'none', True],
                [max_seq_len, 'none', True]]

        ffn3 = [3, [400, 'relu', False],
                [max_seq_len, 'none', True],
                [100, 'none', True]]
        ffn_dict_nm = create_ffn_dict(names, ffn1, ffn2, ffn3)
        input_vocab_size, target_vocab_size, pe_input, pe_target = 8000, 8500, 10000, 11000
        rate_enc, rate_dec, rate_nm_enc = 0.1, 0.1, 0.1
        nm_mha_enc, nm_mha_dec = True, True # test out mismatches later and see if system still works.
        neuromodulation = True # if this is False, the above two must be False as well.
        nm_net_vocab_size, pe_nm_net, nm_mha_net = 8000, 10000, False

        transformer = NMTransformer(num_layers, d_model, num_heads, dff, ffn_dict_enc, ffn_dict_dec, max_seq_len, input_vocab_size,
                      target_vocab_size, pe_input, pe_target, rate_enc, rate_dec, nm_mha_enc, nm_mha_dec,
                      neuromodulation, nm_net_vocab_size, pe_nm_net, rate_nm_enc, nm_mha_net, ffn_dict_nm)

        # inp, tar, nm_inp_enc, nm_inp_dec, training, enc_padding_mask, look_ahead_mask, dec_padding_mask,
        #              nm_padding_mask_enc, nm_dec_padding_mask, external_memory=False
        batch_size = 32
        inp = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=input_vocab_size, dtype=tf.dtypes.int64)
        tar = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=target_vocab_size, dtype=tf.dtypes.int64)
        # input for the encoder run through neuromodulation encoder.
        nm_inp_enc = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=input_vocab_size, dtype=tf.dtypes.int64)
        # input for the decoder run through neuromodulated encoder.
        nm_inp_dec = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=input_vocab_size, dtype=tf.dtypes.int64)
        training=True
        #enc_padding_mask = create_padding_mask(inp)
        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp, tar)
        nm_padding_mask_enc = create_padding_mask(nm_inp_enc)
        nm_dec_padding_mask = create_padding_mask(nm_inp_dec)
        output, attn_weights = transformer(inp, tar, nm_inp_enc, nm_inp_dec, training, enc_padding_mask,
                                           look_ahead_mask, dec_padding_mask, nm_padding_mask_enc, nm_dec_padding_mask,
                                           external_memory=False)

        self.assertEqual([output.shape[0], output.shape[1], output.shape[2]],
                         [batch_size, max_seq_len, target_vocab_size])
        self.assertTrue(len(attn_weights.keys()) == (num_layers*2))

    def test_Transformer_nm_enc_only(self):
        #num_layers, d_model, num_heads, dff, ffn_dict_enc, ffn_dict_dec, max_seq_len, input_vocab_size,
        #         target_vocab_size, pe_input, pe_target, rate_enc=0.1, rate_dec=0.1, nm_mha_enc=False,
        #         nm_mha_dec=False,
        #         neuromodulation=False, nm_net_vocab_size=1000, pe_nm_net=1500, rate_nm_enc=0.1, nm_mha_net=False, ffn_dict_nm={}
        num_layers, d_model, num_heads, dff, max_seq_len = 3, 100, 4, 400, 34
        names = ["default"]
        ffn1 = [2, [400, 'relu', False],
                [100, 'none', False]]
        ffn_dict_enc = create_ffn_dict(names, ffn1)
        ffn_dict_dec = create_ffn_dict(names, ffn1)
        names = ["default", "attention_nm", "start_layer_nm"]
        ffn1 = [2, [400, 'relu', False],
                [100, 'none', False]]

        ffn2 = [3, [400, 'relu', False],
                [max_seq_len, 'none', True],
                [max_seq_len, 'none', True]]

        ffn3 = [3, [400, 'relu', False],
                [max_seq_len, 'none', True],
                [100, 'none', True]]
        ffn_dict_nm = create_ffn_dict(names, ffn1, ffn2, ffn3)
        input_vocab_size, target_vocab_size, pe_input, pe_target = 8000, 8500, 10000, 11000
        rate_enc, rate_dec, rate_nm = 0.1, 0.1, 0.1
        nm_mha_enc, nm_mha_dec = True, False # test out mismatches later and see if system still works.
        neuromodulation = True # if this is False, the above two must be False as well.
        nm_net_vocab_size, pe_nm_net, nm_mha_net = 8000, 10000, False

        transformer = NMTransformer(num_layers, d_model, num_heads, dff, ffn_dict_enc, ffn_dict_dec, max_seq_len, input_vocab_size,
                      target_vocab_size, pe_input, pe_target, rate_enc, rate_dec, nm_mha_enc, nm_mha_dec,
                      neuromodulation, nm_net_vocab_size, pe_nm_net, rate_nm, nm_mha_net, ffn_dict_nm)

        # inp, tar, nm_inp_enc, nm_inp_dec, training, enc_padding_mask, look_ahead_mask, dec_padding_mask,
        #              nm_padding_mask_enc, nm_dec_padding_mask, external_memory=False
        batch_size = 32
        inp = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=input_vocab_size, dtype=tf.dtypes.int64)
        tar = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=target_vocab_size, dtype=tf.dtypes.int64)
        # input for the encoder run through neuromodulation encoder.
        nm_inp_enc = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=input_vocab_size, dtype=tf.dtypes.int64)
        # input for the decoder run through neuromodulated encoder.
        #nm_inp_dec = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=input_vocab_size, dtype=tf.dtypes.int64)
        nm_inp_dec = None
        training=True
        #enc_padding_mask = create_padding_mask(inp)
        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp, tar)
        nm_padding_mask_enc = create_padding_mask(nm_inp_enc)
        #nm_dec_padding_mask = create_padding_mask(nm_inp_dec)
        nm_dec_padding_mask = None
        output, attn_weights = transformer(inp, tar, nm_inp_enc, nm_inp_dec, training, enc_padding_mask,
                                           look_ahead_mask, dec_padding_mask, nm_padding_mask_enc, nm_dec_padding_mask,
                                           external_memory=False)

        self.assertEqual([output.shape[0], output.shape[1], output.shape[2]],
                         [batch_size, max_seq_len, target_vocab_size])
        self.assertTrue(len(attn_weights.keys()) == (num_layers*2))

    def test_Transformer_nm_dec_only(self):
        #num_layers, d_model, num_heads, dff, ffn_dict_enc, ffn_dict_dec, max_seq_len, input_vocab_size,
        #         target_vocab_size, pe_input, pe_target, rate_enc=0.1, rate_dec=0.1, nm_mha_enc=False,
        #         nm_mha_dec=False,
        #         neuromodulation=False, nm_net_vocab_size=1000, pe_nm_net=1500, rate_nm_enc=0.1, nm_mha_net=False, ffn_dict_nm={}
        num_layers, d_model, num_heads, dff, max_seq_len = 3, 100, 4, 400, 34
        names = ["default"]
        ffn1 = [2, [400, 'relu', False],
                [100, 'none', False]]
        ffn_dict_enc = create_ffn_dict(names, ffn1)
        ffn_dict_dec = create_ffn_dict(names, ffn1)
        names = ["default", "attention_nm", "start_layer_nm"]
        ffn1 = [2, [400, 'relu', False],
                [100, 'none', False]]

        ffn2 = [3, [400, 'relu', False],
                [max_seq_len, 'none', True],
                [max_seq_len, 'none', True]]

        ffn3 = [3, [400, 'relu', False],
                [max_seq_len, 'none', True],
                [100, 'none', True]]
        ffn_dict_nm = create_ffn_dict(names, ffn1, ffn2, ffn3)
        input_vocab_size, target_vocab_size, pe_input, pe_target = 8000, 8500, 10000, 11000
        rate_enc, rate_dec, rate_nm = 0.1, 0.1, 0.1
        nm_mha_enc, nm_mha_dec = False, True # test out mismatches later and see if system still works.
        neuromodulation = True # if this is False, the above two must be False as well.
        nm_net_vocab_size, pe_nm_net, nm_mha_net = 8000, 10000, False

        transformer = NMTransformer(num_layers, d_model, num_heads, dff, ffn_dict_enc, ffn_dict_dec, max_seq_len, input_vocab_size,
                      target_vocab_size, pe_input, pe_target, rate_enc, rate_dec, nm_mha_enc, nm_mha_dec,
                      neuromodulation, nm_net_vocab_size, pe_nm_net, rate_nm, nm_mha_net, ffn_dict_nm)

        # inp, tar, nm_inp_enc, nm_inp_dec, training, enc_padding_mask, look_ahead_mask, dec_padding_mask,
        #              nm_padding_mask_enc, nm_dec_padding_mask, external_memory=False
        batch_size = 32
        inp = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=input_vocab_size, dtype=tf.dtypes.int64)
        tar = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=target_vocab_size, dtype=tf.dtypes.int64)
        # input for the encoder run through neuromodulation encoder.
        #nm_inp_enc = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=input_vocab_size, dtype=tf.dtypes.int64)
        # input for the decoder run through neuromodulated encoder.
        nm_inp_dec = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=input_vocab_size, dtype=tf.dtypes.int64)
        nm_inp_enc = None
        training=True
        #enc_padding_mask = create_padding_mask(inp)
        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp, tar)
        #nm_padding_mask_enc = create_padding_mask(nm_inp_enc)
        nm_dec_padding_mask = create_padding_mask(nm_inp_dec)
        nm_padding_mask_enc = None
        output, attn_weights = transformer(inp, tar, nm_inp_enc, nm_inp_dec, training, enc_padding_mask,
                                           look_ahead_mask, dec_padding_mask, nm_padding_mask_enc, nm_dec_padding_mask,
                                           external_memory=False)

        self.assertEqual([output.shape[0], output.shape[1], output.shape[2]],
                         [batch_size, max_seq_len, target_vocab_size])
        self.assertTrue(len(attn_weights.keys()) == (num_layers*2))

    def test_Transformer_no_nm(self):
        #num_layers, d_model, num_heads, dff, ffn_dict_enc, ffn_dict_dec, max_seq_len, input_vocab_size,
        #         target_vocab_size, pe_input, pe_target, rate_enc=0.1, rate_dec=0.1, nm_mha_enc=False,
        #         nm_mha_dec=False,
        #         neuromodulation=False, nm_net_vocab_size=1000, pe_nm_net=1500, rate_nm_enc=0.1, nm_mha_net=False, ffn_dict_nm={}
        num_layers, d_model, num_heads, dff, max_seq_len = 3, 100, 4, 400, 34
        names = ["default"]
        ffn1 = [2, [400, 'relu', False],
                [100, 'none', False]]
        ffn_dict_enc = create_ffn_dict(names, ffn1)
        ffn_dict_dec = create_ffn_dict(names, ffn1)
        names = ["default", "attention_nm", "start_layer_nm"]
        ffn1 = [2, [400, 'relu', False],
                [100, 'none', False]]

        ffn2 = [3, [400, 'relu', False],
                [max_seq_len, 'none', True],
                [max_seq_len, 'none', True]]

        ffn3 = [3, [400, 'relu', False],
                [max_seq_len, 'none', True],
                [100, 'none', True]]
        ffn_dict_nm = create_ffn_dict(names, ffn1, ffn2, ffn3)
        input_vocab_size, target_vocab_size, pe_input, pe_target = 8000, 8500, 10000, 11000
        rate_enc, rate_dec, rate_nm = 0.1, 0.1, 0.1
        nm_mha_enc, nm_mha_dec = False, False # test out mismatches later and see if system still works.
        neuromodulation = False # if this is False, the above two must be False as well.
        nm_net_vocab_size, pe_nm_net, nm_mha_net = 8000, 10000, False

        transformer = NMTransformer(num_layers, d_model, num_heads, dff, ffn_dict_enc, ffn_dict_dec, max_seq_len, input_vocab_size,
                      target_vocab_size, pe_input, pe_target, rate_enc, rate_dec, nm_mha_enc, nm_mha_dec,
                      neuromodulation, nm_net_vocab_size, pe_nm_net, rate_nm, nm_mha_net, ffn_dict_nm)

        # inp, tar, nm_inp_enc, nm_inp_dec, training, enc_padding_mask, look_ahead_mask, dec_padding_mask,
        #              nm_padding_mask_enc, nm_dec_padding_mask, external_memory=False
        batch_size = 32
        inp = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=input_vocab_size, dtype=tf.dtypes.int64)
        tar = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=target_vocab_size, dtype=tf.dtypes.int64)
        # input for the encoder run through neuromodulation encoder.
        #nm_inp_enc = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=input_vocab_size, dtype=tf.dtypes.int64)
        # input for the decoder run through neuromodulated encoder.
        #nm_inp_dec = tf.random.uniform((batch_size, max_seq_len), minval=0, maxval=input_vocab_size, dtype=tf.dtypes.int64)
        nm_inp_enc = None
        nm_inp_dec = None
        training=True
        #enc_padding_mask = create_padding_mask(inp)
        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp, tar)
        #nm_padding_mask_enc = create_padding_mask(nm_inp_enc)
        #nm_dec_padding_mask = create_padding_mask(nm_inp_dec)
        nm_dec_padding_mask = None
        nm_padding_mask_enc = None
        output, attn_weights = transformer(inp, tar, nm_inp_enc, nm_inp_dec, training, enc_padding_mask,
                                           look_ahead_mask, dec_padding_mask, nm_padding_mask_enc, nm_dec_padding_mask,
                                           external_memory=False)

        self.assertEqual([output.shape[0], output.shape[1], output.shape[2]],
                         [batch_size, max_seq_len, target_vocab_size])
        self.assertTrue(len(attn_weights.keys()) == (num_layers*2))

    def test_PWFFN_shape(self):
        names = ["default", "start_layer_nm", "attention_nm"]
        dff = 400
        d_model = 100
        tf_seq_len = 24
        ffn1 = [2, [dff, 'relu', False],
                [d_model, 'none', False]]

        ffn2 = [3, [dff, 'relu', False],
                [tf_seq_len, 'none', True],
                [d_model, 'softmax', True]]

        ffn3 = [3, [dff, 'relu', False],
                [tf_seq_len, 'none', True],
                [tf_seq_len, 'softmax', True]]

        dct = create_ffn_dict(names, ffn1, ffn2, ffn3)
        ffn = NMPWFeewForwardNetwork(dct)

        batch_size = 32
        seq_len = 20
        input = tf.ones((batch_size, seq_len, d_model))
        output_dict = ffn(input)

        default_layer = output_dict["default"]
        attention_nm_layer = output_dict["attention_nm"]
        start_layer_nm = output_dict["start_layer_nm"]

        self.assertEqual([default_layer.shape[0], default_layer.shape[1], default_layer.shape[2]],
                         [batch_size, seq_len, d_model])
        self.assertEqual([start_layer_nm.shape[0], start_layer_nm.shape[1], start_layer_nm.shape[2]],
                         [batch_size, tf_seq_len, d_model])
        self.assertEqual([attention_nm_layer.shape[0], attention_nm_layer.shape[1], attention_nm_layer.shape[2]],
                         [batch_size, tf_seq_len, tf_seq_len])
        self.assertEqual(len(default_layer.shape), 3)
        self.assertEqual(len(attention_nm_layer.shape), 3)
        self.assertEqual(len(start_layer_nm.shape), 3)

if __name__ == "__main__":
    unittest.main()