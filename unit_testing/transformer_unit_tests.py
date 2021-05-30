import unittest

import sys
sys.path.append("..")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


from models.Transformer import *


class TestTransformerShapes(unittest.TestCase):
    '''
    Class: TestTransformerShapes
    Description: Class that performs unit tests on each Transformer layer. Each test involves
        testing if the shape of the output of each module matches what is expected.

    https://www.tensorflow.org/tutorials/text/transformer is where the test cases/code is taken from.
    '''
    def test_EncoderLayer(self):
        enc_layer = EncoderLayer(100, 10, 400)
        x = tf.random.uniform((64, 43, 100))
        out = enc_layer(x, False, None)

        self.assertEqual([out.shape[0], out.shape[1], out.shape[2]], [64, 43, 100])

    def test_DecoderLayer(self):
        dec_layer = DecoderLayer(100, 10, 400)

        # Encoder output is needed here.
        enc_layer = EncoderLayer(100, 10, 400)
        x = tf.random.uniform((64, 43, 100))
        out_enc = enc_layer(x, False, None)

        out, _, _ = dec_layer(tf.random.uniform((64, 50, 100)), out_enc, False, None, None)

        self.assertEqual([out.shape[0], out.shape[1], out.shape[2]], [64, 50, 100])

    def test_MHA(self):
        temp_mha = MultiHeadAttention(d_model=100, num_heads=10)
        y = tf.random.uniform((1, 60, 100))  # (batch_size, encoder_sequence, d_model)
        out, attn = temp_mha(y, k=y, q=y, mask=None)
        self.assertEqual([out.shape[0], out.shape[1], out.shape[2],
                          attn.shape[0], attn.shape[1], attn.shape[2], attn.shape[3]],
                         [1, 60, 100, 1, 10, 60, 60])

    def test_Encoder(self):
        encoder = Encoder(num_layers=2, d_model=100, num_heads=10,
                                 dff=400, input_vocab_size=8500,
                                 maximum_position_encoding=10000)
        temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

        out = encoder(temp_input, training=False, mask=None)

        self.assertEqual([out.shape[0], out.shape[1], out.shape[2]], [64, 62, 100])
        # (batch_size, input_seq_len, d_model)

    def test_Decoder(self):
        decoder = Decoder(num_layers=2, d_model=100, num_heads=10,
                                 dff=400, target_vocab_size=8000,
                                 maximum_position_encoding=5000)

        encoder = Encoder(num_layers=2, d_model=100, num_heads=10,
                                 dff=400, input_vocab_size=8500,
                                 maximum_position_encoding=10000)

        temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

        sample_encoder_output = encoder(temp_input, training=False, mask=None)

        temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

        out, attn = decoder(temp_input,
                                      enc_output=sample_encoder_output,
                                      training=False,
                                      look_ahead_mask=None,
                                      padding_mask=None)

        attn = attn['decoder_layer2_block2'] # to simplify code for testing below.
        #output.shape, attn['decoder_layer2_block2'].shape
        self.assertEqual([out.shape[0], out.shape[1], out.shape[2],
                          attn.shape[0], attn.shape[1], attn.shape[2], attn.shape[3]],
                         [64, 26, 100, 64, 10, 26, 62])

    def test_Transformer(self):
        transformer = Transformer(
            num_layers=2, d_model=100, num_heads=10, dff=400,
            input_vocab_size=8500, target_vocab_size=8000,
            pe_input=10000, pe_target=6000)

        temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
        temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

        fn_out, _ = transformer(temp_input, temp_target, training=False,
                                       enc_padding_mask=None,
                                       look_ahead_mask=None,
                                       dec_padding_mask=None)

        #fn_out.shape  # (batch_size, tar_seq_len, target_vocab_size)
        self.assertEqual([fn_out.shape[0], fn_out.shape[1], fn_out.shape[2]], [64, 36, 8000])

    def test_PWFFN(self):
        sample_ffn = PWFeedForwardNetwork(100, 400)
        out = sample_ffn(tf.random.uniform((64, 50, 100)))
        self.assertEqual([out.shape[0], out.shape[1], out.shape[2]],[64,50,100])

if __name__ == "__main__":
    unittest.main()