import tensorflow as tf
import numpy as np

import sys
sys.path.append("..")

from models.MultiHeadAttention import MultiHeadAttention
from models.FeedForwardNetwork import FeedForwardNetwork
from models.PositionEncoding import get_angles, positional_encoding

class DecoderLayer(tf.keras.layers.Layer):
    '''
    Description: Implementation of a transformer decoder layer.
    '''

    def __init__(self, d_model, num_heads, dff, mask_strategy, rate=0.1, name=None):
        '''
        :param d_model: (int) An integer that represents the dimension of the decoder layer (or the transformer as a whole).
        :param num_heads: (int) An integer that represents the number of heads in the multi-head attention component.
        :param dff: (int) An integer that represents the dimension of the feed forward layer.
        :param mask_strategy: (str) A string that represents what mask is to be used in this layer.
        :param rate: (float) A floating point number that represents the dropout rate of dropout layers.
        :param name: (None | str) A NoneType object or string if the name of this layer is needed to be specified.
        '''
        if name is not None: super(DecoderLayer, self).__init__(name=name)
        else: super(DecoderLayer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.mask_strategy = mask_strategy
        self.rate = rate

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, dff, "vanilla")

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, input_shape=(d_model,))
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, input_shape=(d_model,))

        self.dropout1 = tf.keras.layers.Dropout(rate=rate, input_shape=(d_model,))
        self.dropout2 = tf.keras.layers.Dropout(rate=rate, input_shape=(d_model,))

    def call(self, x, training, mask):
        '''
        :param x: (tf.Tensor; [batch_size, seq_len, d_model]) Input to the model in tensor form.
        :param training: (bool) Boolean value representing if the dropout layers are to be in training mode or not.
        :param mask: (tf.Tensor; [batch_size, ..., ..., seq_len]) A tensor representing the mask to be used in multi-head attention.
        :return: (tf.Tensor; [batch_size, seq_len, d_model])
        '''
        # note: that if aux tokens are meant to be removed from the beginning of the sequence, do it before passing as input to this function.

        # block 1
        x_ = self.layernorm1(x)
        attn1, attention_weights = self.mha(x_, x_, x_, mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = x + attn1 # (batch_size, seq_len, d_model)

        # block 2
        out1_ = self.layernorm2(out1)
        out2 = self.ffn(out1_)
        out2 = self.dropout2(out2, training=training)
        out2 = out1 + out2

        return out2, attention_weights


class Decoder(tf.keras.layers.Layer):
    '''
    Description: Implementation of the full Decoder of a transformer (all N layers).
    '''

    def __init__(self, num_layers, d_model, num_heads, dff, max_seq_len=768,
                 mask_strategy='default', rate=0.1):
        '''
        :param num_layers: (int) An integer specifying the number of decoder layers.
        :param d_model: (int) An integer specifying the dimension of the decoder layers (and the transformer as a whole).
        :param num_heads: (int) An integer specifying the number of heads in the multi-head attention module.
        :param dff: (int) An integer specifying the dimension of the feed forward layer.
        :param max_seq_len: (int) An integer specifying the maximum sequence length of the input tensors.
        :param mask_strategy: (str) A string specifying the masking strategy.
        :param rate: (float) A floating point number that represents the dropout rate of dropout layers.
        '''
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.mask_strategy = mask_strategy
        self.max_seq_len = max_seq_len
        self.rate = rate

        # note that the embedding is moving from here to the parent NMTransformer class... as well as the absolute pos embeddings...

        self.W1 = tf.keras.layers.Dense(d_model, input_shape=(d_model,)) # linear projection.
        self.W2 = tf.keras.layers.Dense(d_model, input_shape=(d_model,)) # linear projection.
        self.W3 = tf.keras.layers.Dense(d_model, input_shape=(d_model,)) # linear projection.
        self.W4 = tf.keras.layers.Dense(d_model, input_shape=(d_model*2,)) # input is d_model*2, just noting.

        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, mask_strategy, rate=rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate, input_shape=(d_model,))

    def call(self, x, training, mask, do_aoint_bool=False, do_aoint_indices=[0, 0]):
        '''
        :param x: (tf.Tensor; [batch_size, seq_len, d_model]) A tensor representing the input to the decoder component.
        :param training: (bool) A boolean value specifying if we are in training mode for layers which have differnt modes
            for training an non-training.
        :param mask: (tf.Tensor; [batch_size, ..., ..., seq_len]) A tensor representing the mask to be used in multi-head attention.
        :param do_aoint_bool: (bool) A boolean value indicating if the answer aption reading strategy is to be utilized.
        :param do_aoint_indices: (list; [start_int, end_int]) A list of two integers where the first specifies the first
            specifies the start index and the second specifies the second index.
        :return: (tf.Tensor; [batch_size, max_seq_len, d_model]) |
            (dict of tf.Tensor; [batch_size, num_heads, seq_len, seq_len (can vary)])
        '''

        seq_len = x.shape[1]
        if do_aoint[0]:
            x = self.aoint_helper(x, do_aoint[1], do_aoint[2])

        attention_weights = dict()
        for i, layer in enumerate(self.decoder_layers):
            x, block1 = layer(x=x, training=training, mask=mask)
            attention_weights[f"decoder_layer_{i+1}_block1"] = block1

        x = self.dropout(x)
        return x, attention_weights

    def aoint_helper(self, x, start, end):
        '''
        Desription: Helper function that runs the answer option interaction reading strategy. \n
        :param x: (tf.Tensor; [batch_size, seq_len, d_model]) A tensor representing the input to the decoder.
        :param start: (int) An integer specifying the start of the question in the input (inlcusive).
        :param end: (int) An integer specifying the end of the questions in the input (non-inclusive).
        :return: (tf.Tensor; [batch_size, seq_len, d_model])
        '''
        seq_len = x.shape[1]
        x_ans = x[:, start:end, :]

        H1 = self.W1(x_ans) # (batch_size, seq_len_ans, d_model)
        H2 = self.W2(x_ans) # (batch_size, seq_len_ans, d_model)

        G = tf.nn.softmax(tf.matmul(self.W3(H1), H2, transpose_b=True)) # (batch_size, seq_len_ans, seq_len_ans)
        H_int = tf.maximum(tf.matmul(G, H2), 0) # (batch_size, seq_len_ans, d_model)

        # concat along the dimension (d_model) axis. (batch_size, seq_len_ans, d_model*2)
        g = self.W4(tf.concat([H_int, H1], axis=-1)) # (batch_sise, seq_len_ans, d_model)

        x_new = (g * H1) + ((1-g) * H_int) # (batch_size, seq_len_ans, d_model)

        x = tf.concat([x[:, :start, :],
                       x_new,
                       x[:, end:, :]], axis=1)
        assert x.shape[1] == seq_len, f"The number of tokens has changed. Input: {seq_len} Output: {x.shape[1]}"
        return x


if __name__ == "__main__":
    #num_layers, d_model, num_heads, dff, max_seq_len=768,
                 #mask_strategy='default', rate=0.1
    d_model, num_heads, dff, max_seq_len = 100, 10, 20, 24
    num_layers = 4
    dec = Decoder(num_layers, d_model, num_heads, dff, max_seq_len=max_seq_len,
                 mask_strategy='default', rate=0.1)
    batch_size = 4

    x = tf.random.uniform((batch_size, max_seq_len, d_model))
    #enc_output = tf.random.uniform((batch_size, max_seq_len+1, d_model)) # +1 to test different encoder max_seq_len...
    training = True
    mask = None

    output, attn_weights = dec(x, training, mask, do_aoint=[True, 5, 14])

    print(f"output: {output} \n"
          f"output.shape: {output.shape}")
    #print(f"attn_weights_block1_layer1: {attn_weights['decoder_layer_1_block1']}")
    #print(f"attn_weights_block2_layer1: {attn_weights['decoder_layer_2_block1']}")

