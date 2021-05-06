import collections
import logging
import os
import pathlib
import re
import string
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"#"0,1,2,3"

class MultiHeadAttention(tf.keras.layers.Layer):
    '''
    Class: MultiHeadAttention
    Purpose: Implementation of multi-head attention to be used in a Transformer.
    Inputs:
        d_model: (int) The dimension  of the model.
        num_heads: (int) The number of heads in the Multi head attention layer.
    '''
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model

        # Must be a multiple of the number of heads.
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        '''
        Function: split_heads
        Purpose: To split the last dimension into (num_heads, depth).
            Then transpose the tensor to the following permutation, [0,2,1,3].
        Input:
            x: (tf.Tensor).
            batch_size: (int) Batch size during training.
        Return:
        '''
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0,2,1,3])

    def call(self, v, k, q, mask):
        '''
        Function: call
        Purpose: Code to be executed when the keras layer class is called.
        Input:
            v:
            k:
            q:
            mask:
        Return:
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
            q:
            k:
            v:
            mask: (must be broadcastable)
        Return:
        '''

        matmul_qk = tf.matmul(q, k, transpose_b = True) # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so the scores add to 1.
        # Technically you could do the other axis as long as it is consistent throough training.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v) # (..., seq_len_q, depth_v)

        return output, attention_weights


class PWFeedForwardNetwork(tf.keras.layers.Layer):
    '''
    Class: PWFeedForwardNetwork
    Purpose: Implementation of the point-wise feed forward network
    Inputs:
        d_model: (int) The dimension of the transformer model.
        dff: (int) The dimension of the feed forward layer.

        TODO. num_ffn: (int) The number of feed forward layers to run in parallel.
    '''
    def __init__(self, d_model, dff, num_ffn=1):
        super(PWFeedForwardNetwork, self).__init__()

        self.d_model = d_model
        self.dff = dff

        # TODO: later incorporate this many feed forward layers.
        self.num_ffn = num_ffn

        # TODO: possibly put all into a list and loop over to create them.
        self.ffnetwork1 = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dff, activation='relu'), # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(self.d_model) # (batch_size, seq_len, d_model)
        ])

    def call(self, input):
        '''
        Function: call
        Purpose: Runs the point-wise feed forward network with the given input.
        Input:
            input: (tf.Tensor) -- later put dimensions here. (batch_size, seq_len, d_model)
        '''
        # TODO: add support for multiple feed forward layers in parallel.
        output = self.ffnetwork1(input)
        return output

class EncoderLayer(tf.keras.layers.Layer):
    '''
    Class: EncoderLayer
    Description: Implementation of a single encoder layer.
    Input:
        d_model: (int) Transformer model dimension.
        num_heads: (int) Number of multi-head attention heads.
        dff: (int) Dimension of the feed forward network.
        num_ffn: (int) Number of feed forward networks to run in parallel in an encoder layer.
        rate: (float) The dropout rate for the keras dropout layers.
    '''
    def __init__(self, d_model, num_heads, dff, num_ffn=1, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PWFeedForwardNetwork(d_model, dff, num_ffn)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6) # for multi-head attention layer.
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6) # for feed forward network.

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        '''
        Function: call
        Description: When called it runs a single encoder layer.
        Input:
            x:
            training: (boolean) Determines if we are in the training phase or evaluation (testing) phase.
            mask:
        '''

        attn_output, _ = self.mha(x, x, x, mask) # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output) # batch_size, input_seq_len, d_model)

        fnn_output = self.ffn(out1) #( batch_size, input_seq_len, d_model)
        fnn_output = self.dropout2(fnn_output, training=training)
        out2 = self.layernorm2(out1 + fnn_output) # (batch_size, input_seq_len, d_model)

        return out2

class DecoderLayer(tf.keras.layers.Layer):
    '''
    Class: DecoderLayer
    Description: Implementation of a single decoder layer.
    Input:
        d_model: (int) Transformer model dimension.
        num_heads: (int) Number of heads in the multi-head attention layer.
        dff: (int) Feed forward layer dimension.
    '''
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = PWFeedForwardNetwork(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for (masked) multi-head attention layer.
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for multi-head attention layer.
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for feed forward network.

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        '''
        Function: call
        Description: Implementation of a single decoder layer.
        Input:
            x:
            enc_output:
            training: (boolean)
            look_ahead_mask:
            padding_mask:
        '''
        #enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask) # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output,
                                               out1, padding_mask) # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2) # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output) # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

class Encoder(tf.keras.layers.Layer):
    '''
    Class: Encoder
    Description: Implementaion of N stacked Encoder layers.
    Input:
        num_layers:
        d_model:
        num_heads:
        dff:
        input_vocab_size:
        maximum_pposition_encoding:
        rate: (float)
    '''
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(maximum_position_encoding, self.d_model)

        # Do below for self_attention layer.
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        '''
        pass
        '''

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x) # (batch_size, seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # scale embedding by the sqrt of hidden size.
        '''
        The reason we increase the embedding values before the addition is to make the 
        positional encoding relatively smaller. This means the original meaning in the 
        embedding vector won’t be lost when we add them together.
        '''
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x # (batch_size, input_seq_len, d_model)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :],
                                d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

class Decoder(tf.keras.layers.Layer):
    '''
    pass
    '''
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        '''
        pass
        '''
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # same reason as for encoder.
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :],
                                d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)


class Transformer(tf.keras.Model):
    '''
    pass
    '''
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.tokenizer = Encoder(num_layers, d_model, num_heads, dff,
                                 input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        '''
        pass.
        '''

        enc_output = self.tokenizer(inp, training, enc_padding_mask) # (batch_size, inp_seq_len, d_model)

        #dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output) # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


def create_check_point_objects(path, transformer, optimizer):
    pass

if __name__ == '__main__':

    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    #print(tf.test.gpu_device_name())

    # Note: Make Unit tests for these later.

    '''
    # Test Multi-head Attention.
    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    print(out.shape, attn.shape)
    '''

    '''
    # Test Encoder layer.
    sample_encoder_layer = EncoderLayer(512, 8, 2048)
    z = tf.random.uniform((64, 43, 512))
    sample_encoder_layer_output = sample_encoder_layer(
        z, False, None)
    print("Input", z)
    print("Output", sample_encoder_layer_output)
    print("Output shape:", sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)
    '''

    '''
    # Decoder Layer test.
    sample_decoder_layer = DecoderLayer(512, 8, 2048)

    # Encoder output is needed here.
    sample_encoder_layer = EncoderLayer(512, 8, 2048)
    z = tf.random.uniform((64, 43, 512))
    sample_encoder_layer_output = sample_encoder_layer(
        z, False, None)

    sample_decoder_layer_output, _, _ = sample_decoder_layer(
        tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
        False, None, None)

    print("Output dimension", sample_decoder_layer_output.shape)  # (batch_size, target_seq_len, d_model)
    '''

    '''
    # Full encoder layer (n layers)
    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048, input_vocab_size=8500,
                             maximum_position_encoding=10000)
    temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

    sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

    print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)
    '''

    '''
    # Decoder Test
    sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048, target_vocab_size=8000,
                             maximum_position_encoding=5000)

    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048, input_vocab_size=8500,
                             maximum_position_encoding=10000)
    temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

    sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

    temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

    output, attn = sample_decoder(temp_input,
                                  enc_output=sample_encoder_output,
                                  training=False,
                                  look_ahead_mask=None,
                                  padding_mask=None)

    print(output.shape, attn['decoder_layer2_block2'].shape)
    '''

    '''
    # Testing the entire transformer.
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

    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)
    '''