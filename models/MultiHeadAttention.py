import tensorflow as tf
import numpy as np

import sys
sys.path.append("..")

class MultiHeadAttention(tf.keras.layers.Layer):
    '''
    Description: An implementation of Multi-head attention in a transformer.
    '''

    def __init__(self, d_model: int, num_heads: int):
        '''
        :param d_model: (int) An integer specifying the dimension of the transformer.
        :param num_heads: (int) An integer specifying the number of heads.
        '''
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, f"The number of heads is incompatible with the dimension of the model!"
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model, input_shape=(d_model,))
        self.wk = tf.keras.layers.Dense(d_model, input_shape=(d_model,))
        self.wv = tf.keras.layers.Dense(d_model, input_shape=(d_model,))

        self.dense = tf.keras.layers.Dense(d_model, input_shape=(d_model,))

    def split_heads(self, x):
        '''
        :param x: (tf.Tensor; [batch_size, (max_)seq_len, d_model]
        :return: (tf.Tensor; [batch_size, num_heads, seq_len, depth])
        '''
        assert len(x.shape) == 3, f"The number of dimensions of the input x should be 3, got {x.shape}!"
        batch_size = x.shape[0]

        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        '''
        Description: One run through the multi-head attention layer. \n
        :param v: (tf.Tensor; [batch_size, seq_len_v, d_model]) Value tensor.
        :param k: (tf.Tensor; [batch_size, seq_len_k, d_model]) Key tensor.
        :param q: (tf.Tensor; [batch_size, seq_len_q, d_model]) Query tensor.
        :param mask: (tf.Tensor; [batch_size, ..., ..., seq_len]) Tensor representing the mask to be applied.
        :return: (tf.Tensor; [batch_size, seq_len_q, d_model]) | (dict of tf.Tensor; [batch_size, num_heads, seq_len, seq_len])
        '''
        batch_size = q.shape[0]

        q = self.wq(q) 
        k = self.wk(k)
        v = self.wk(v) # (batch_size, seq_len, d_model)

        q = self.split_heads(q) # (batch_size, num_heads, seq_len, depth) # note for my purposes all seq lengths will be the same.
        k = self.split_heads(k) # (batch_size, num_heads, seq_len, depth)
        v = self.split_heads(v) # (batch_size, num_heads, seq_len, depth)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0,2,1,3]) # (batch_size, seq_len, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model)) # (batch_size, seq_len, d_model)

        output = self.dense(concat_attention) # (batch_size, seq_len, d_model)

        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        '''
        Description: Performs scaled dot-product attention with a query, key, value and mask. \n
        :param q: (tf.Tensor; [batch_size, num_heads, seq_len, depth]) Tensor representing the linearly projected query.
        :param k: (tf.Tensor; [batch_size, num_heads, seq_len, depth]) Tensor representing the linearly projected key.
        :param v: (tf.Tensor; [batch_size, num_heads, seq_len, depth]) Tensor representing the linearly projected value.
        :param mask: None | (tf.Tensor; [batch_size, .., ..., seq_len]) Tensor representing the mask to be applied.
        :return: (tf.Tensor; [batch_size, num_heads, seq_len, depth]) and (dict of tf.Tensor; [batch_size, num_heads, seq_len, seq_len])
        '''
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)  # i.e. get the depth.
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so the scores add to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, depth_v)

        return output, attention_weights

if __name__ == "__main__":
    d_model, num_heads, max_seq_len, nm_gating = 10, 2, 3, True
    batch_size = 4
    q = tf.random.uniform((batch_size, max_seq_len, d_model))
    mask = None
    nm_inp_gating = tf.random.uniform((batch_size, max_seq_len, max_seq_len))
    q, k, v = q, q, q
    mha = MultiHeadAttention(d_model, num_heads)
    output, attention = mha(q,k,v,mask)
    print(f"Output.shape: {output.shape} \n"
          f"attention: {attention}")