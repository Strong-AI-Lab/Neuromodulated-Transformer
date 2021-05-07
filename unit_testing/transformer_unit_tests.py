import unittest

import sys
sys.path.append("..")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from models.Transformer import *



class TestTransformerShapes(unittest.TestCase):
    '''
    Class:
    Description:

    https://www.tensorflow.org/tutorials/text/transformer is where the test code is taken from.
    '''
    def test_EncoderLayer(self):
        sample_encoder_layer = EncoderLayer(512, 8, 2048)
        z = tf.random.uniform((64, 43, 512))
        out = sample_encoder_layer(z, False, None)
        #print("Output shape:", out.shape)  # (batch_size, input_seq_len, d_model)

        self.assertEqual([out.shape[0], out.shape[1], out.shape[2]], [64, 43, 512])

    def test_DecoderLayer(self):
        sample_decoder_layer = DecoderLayer(512, 8, 2048)

        # Encoder output is needed here.
        sample_encoder_layer = EncoderLayer(512, 8, 2048)
        z = tf.random.uniform((64, 43, 512))
        sample_encoder_layer_output = sample_encoder_layer(
            z, False, None)

        out, _, _ = sample_decoder_layer(
            tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
            False, None, None)

        self.assertEqual([out.shape[0], out.shape[1], out.shape[2]], [64, 50, 512])
        # (batch_size, target_seq_len, d_model)

    def test_MHA(self):
        temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
        y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
        out, attn = temp_mha(y, k=y, q=y, mask=None)
        self.assertEqual([out.shape[0], out.shape[1], out.shape[2],
                          attn.shape[0], attn.shape[1], attn.shape[2], attn.shape[3]],
                         [1, 60, 512, 1, 8, 60, 60])

    def test_Encoder(self):
        sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                                 dff=2048, input_vocab_size=8500,
                                 maximum_position_encoding=10000)
        temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

        out = sample_encoder(temp_input, training=False, mask=None)

        self.assertEqual([out.shape[0], out.shape[1], out.shape[2]], [64, 62, 512])
        # (batch_size, input_seq_len, d_model)

    def test_Decoder(self):
        sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
                                 dff=2048, target_vocab_size=8000,
                                 maximum_position_encoding=5000)

        sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                                 dff=2048, input_vocab_size=8500,
                                 maximum_position_encoding=10000)

        temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

        sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

        temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

        out, attn = sample_decoder(temp_input,
                                      enc_output=sample_encoder_output,
                                      training=False,
                                      look_ahead_mask=None,
                                      padding_mask=None)

        attn = attn['decoder_layer2_block2'] # to simplify code for testing below.
        #output.shape, attn['decoder_layer2_block2'].shape
        self.assertEqual([out.shape[0], out.shape[1], out.shape[2],
                          attn.shape[0], attn.shape[1], attn.shape[2], attn.shape[3]],
                         [64, 26, 512, 64, 8, 26, 62])

    def test_Transformer(self):
        sample_transformer = Transformer(
            num_layers=2, d_model=512, num_heads=8, dff=2048,
            input_vocab_size=8500, target_vocab_size=8000,
            pe_input=10000, pe_target=6000)

        temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
        temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

        fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
                                       enc_padding_mask=None,
                                       look_ahead_mask=None,
                                       dec_padding_mask=None)

        #fn_out.shape  # (batch_size, tar_seq_len, target_vocab_size)
        self.assertEqual([fn_out.shape[0], fn_out.shape[1], fn_out.shape[2]], [64, 36, 8000])

    def test_PWFFN(self):
        sample_ffn = PWFeedForwardNetwork(512, 2048)
        out = sample_ffn(tf.random.uniform((64, 50, 512)))
        self.assertEqual([out.shape[0], out.shape[1], out.shape[2]],[64,50,512])

if __name__ == "__main__":
    unittest.main()