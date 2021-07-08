'''
File name: MultiHeadAttention.py
Author: Kobe Knowles
Date created: 05/07/21
Data last modified: 05/07/21
Python Version: 3.6
Tensorflow version: 2
'''

import tensorflow as tf
import numpy as np

class MultiHeadAttention(tf.keras.layers.Layer):
    '''
    Class: MultiHeadAttention \n
    Description: Implementation of the multi-head attention for a transformer layer. \n
    Attributes:
        d_model: (int) The dimension of the transformer. \n
        num_heads: (int) The number of heads in the multi-head attention layer. \n
        max_seq_len: (int) The maximum sequence length to be passed in as input. \n
        nm_gating: (bool) True if neuromodulated gating is to occur; False otherwise. \n
        depth: (int) The dimension of each individual head. \n
        wq: (tf.keras.layers.Dense) Dense layer to convert the input to the Query matrix. \n
        wk: (tf.keras.layers.Dense) Dense layer to convert the input to the Key matrix. \n
        wv: (tf.keras.layers.Dense) Dense layer to convert the input to the Value matrix. \n
        nm_gate_logits: (list; tf.keras.layers.Dense) List of dense layers, one for each head.
            (list of Dense layers if nm_gating is True; None otherwise) \n
        dense: (tf.keras.layers.Dense) Dense layer at the end of the multi-head attention layer, after all
            heads' tensors are concatenated together.
    '''

    def __init__(self, d_model, num_heads, max_seq_len, nm_gating=False):
        '''
        Function: __init__ \n
        Description: Initializes the multi-head attention layer with the passed parameters. \n
        Input:
            d_model: (int) The dimension of the transformer. \n
            num_heads: (int) The number of heads in this multi-head attention layer. \n
            max_seq_len: (int) The maximum sequence length to be passed as input. \n
            nm_gataing: (bool) True if context dependant gating is to occur; False otherwise. Defaults to False.
        '''
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, f"The number of heads is incompatible with the dimension of the model \n" \
                                         f"d_model: {d_model} \t num_heads: {num_heads}"

        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.nm_gating = nm_gating
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.nm_gate_logits = None
        if self.nm_gating:
            self.nm_gate_logits = [tf.keras.layers.Dense(self.max_seq_len) for _ in range(num_heads)]

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x):
        '''
        Function: split_heads \n
        Description: Adjusts the dimensions (i.e. splits them) to multiple head format.\n
        Input:
            x: (tf.Tensor; [batch_size, (max_)seq_len, d_model]
        Return:
            (tf.Tensor; [batch_size, num_heads, (max_)seq_len, depth])
        '''
        assert len(x.shape) == 3, f"The number of dimensions of the input x should be 3, got {x.shape}!"
        batch_size = x.shape[0]

        x = tf.reshape(x, (batch_size, self.num_heads, -1, self.depth)) # -1 represents max_seq_len dimension.
        return x

    def call(self, v, k, q, nm_inp_gating=None, mask=None):
        '''
        Function: call \n
        Description: Overrides parents class's call function (i.e. run through the multi-head attention layer once). \n
        Input:
            v: (tf.Tensor; [batch_size, seq_len_v, word_dim|d_model]) Value input tensor. \n
            k: (tf.Tensor; [batch_size, seq_len_k, word_dim|d_model]) Key input tensor. \n
            q: (tf.Tensor; [batch_size, seq_len_q, word_dim|d_model]) Query input tensor. \n
            nm_inp_gating: (tf.Tensor; [batch_size, (max_)seq_len, (max_)seq_len])
                Gating input from the neuromodulation network. \n
            mask: (tf.Tensor; [batch_size, ..., ..., seq_len]) Mask to apply to the attention matrix.
                Note: ... means that the exact dimension size can vary from encoder to decoder.
                Encoder: (batch_size, 1, 1, seq_len)
                Decoder 1st block: (batch_size, 1, seq_len, seq_len)
                Decoder 2nd block: (batch_size, 1, 1, seq_len)
                Python broadcasting allows this to work as intended. \n
        Return:
            output: (tf.Tensor; [batch_size, (max_)seq_len, d_model]) \n
            attention_weights: (tf.Tensor; [batch_size, num_heads, (max_)seq_len(_q), (max_)seq_len(_k)])
        '''
        batch_size = q.shape[0]

        q = self.wq(q)  # (batch_size, (max_)seq_len, d_model)
        k = self.wk(k)  # (batch_size, (max_)seq_len, d_model)
        v = self.wv(v)  # (batch_size, (max_)seq_len, d_model)

        q = self.split_heads(q)  # (batch_size, num_heads, (max_)seq_len, depth)
        k = self.split_heads(k)  # (batch_size, num_heads, (max_)seq_len, depth)
        v = self.split_heads(v)  # (batch_size, num_heads, (max_)seq_len, depth)

        scaled_attention, attention_weights = None, None
        if nm_inp_gating is not None:
            if not self.nm_gating:
                raise Exception(f"nm_inp_gating ({type(nm_inp_gating)}) should be None if nm_gating ({self.nm_gating}) is set to False")
            scaled_attention, attention_weights = self.neuromodulation_attention(q, k, v, nm_inp_gating, mask)
        else:
            scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0,2,1,3]) # # (batch_size, (max_)seq_len(_q), num_heads, v_depth)

        # combine all heads into one dimension
        scaled_attention = tf.reshape(scaled_attention, [batch_size, -1, self.d_model])

        output = self.dense(scaled_attention) # (batch_size, (max_)seq_len, d_model)

        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        '''
        Function: scaled_dot_product_attention \n
        Purpose: Perform scaled dot-product attention with a query, key, value and a mask as input.
            Specifically calculate the attention weights and multiply them with the value, v. \n
        Input:
            q: (tf.Tensor; [batch_size, num_heads, seq_len_q, depth]) Query input tensor. \n
            k: (tf.Tensor; [batch_size, num_heads, seq_len_k, depth]) Key input tensor. \n
            v: (tf.Tensor; [batch_size, num_heads, seq_len_v, depth]) Value input tensor. \n
            mask: (tf.Tensor; [batch_size, ..., ..., seq_len]) Mask to apply to the attention matrix.
                Note: ... means that the exact dimension size can vary from encoder to decoder.
                Encoder: (batch_size, 1, 1, seq_len)
                Decoder 1st block: (batch_size, 1, seq_len, seq_len)
                Decoder 2nd block: (batch_size, 1, 1, seq_len)
                Python broadcasting allows this to work as intended. \n
        Return:
            output: (tf.Tensor; [batch_size, num_heads, seq_len_q, depth_v]) \n
            attention_weights: (tf.Tensor; [batch_size, num_heads, seq_len_q, seq_len_k])
        '''

        matmul_qk = tf.matmul(q, k, transpose_b = True) # (batch_size, num_heads, seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32) # i.e. get the depth.
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so the scores add to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v) # (batch_size, num_heads, seq_len_q, depth_v)

        return output, attention_weights

    def neuromodulation_attention(self, q, k, v, nm_inp_gating, mask=None):
        '''
        Function: neuromodulation_attention
        Description: Performs neuromodulation gated variant of scaled dot-product attention.
        Input:
            v: (tf.Tensor; [batch_size, seq_len_v, word_dim|d_model]) Value input tensor. \n
            k: (tf.Tensor; [batch_size, seq_len_k, word_dim|d_model]) Key input tensor. \n
            q: (tf.Tensor; [batch_size, seq_len_q, word_dim|d_model]) Query input tensor. \n
            nm_inp_gating: (tf.Tensor; [batch_size, (nm_)(max_)seq_len, (nm_)(max_)seq_len])
                Gating input from the neuromodulation network. \n
            mask: (tf.Tensor; [batch_size, ..., ..., seq_len]) Mask to apply to the attention matrix.
                Note: ... means that the exact dimension size can vary from encoder to decoder.
                Encoder: (batch_size, 1, 1, seq_len)
                Decoder 1st block: (batch_size, 1, seq_len, seq_len)
                Decoder 2nd block: (batch_size, 1, 1, seq_len)
                Python broadcasting allows this to work as intended. \n
        Return:
            output: (tf.Tensor; [batch_size, num_heads, seq_len_q, depth_v])
            attention_weights: (tf.Tensor; [batch_size, num_heads, seq_len_q, seq_len_k])
        '''
        assert q.shape[2] == k.shape[2], f"The key's and queries' sequence length should be the same \n" \
                                         f"Got q.shape[2]: {q.shape[2]} \t k.shape[2]: {k.shape[2]}"
        assert self.nm_gating, f"The nm_gating variable should be set to True. nm_gating: {self.nm_gating}"

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)  # i.e. get the depth.
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        gated_attn_logits = None
        for i in range(self.num_heads):
            # for each head multiply the head by it's associated dense layer.
            z = self.nm_gate_logits[i](nm_inp_gating)

            # expand along dim 1 to match the scaled_attention_logits' num_heads dimension.
            if gated_attn_logits is None:
                gated_attn_logits = tf.expand_dims(z, axis=1)
            else:
                gated_attn_logits = tf.concat([gated_attn_logits, tf.expand_dims(z, axis=1)], 1)

        # gated_attn_logits.shape == (batch_size, num_heads, max_seq_len, max_seq_len)

        assert scaled_attention_logits.shape == gated_attn_logits.shape, f"The dimensions between scaled_attention_logits ({scaled_attention_logits.shape})" \
                                                                                f"and gated_attn_logits ({gated_attn_logits.shape}) doesn't match"

        gated_attn_logits = tf.math.sigmoid(gated_attn_logits)

        # perform context dependant gating.
        attention_weights = gated_attn_logits * scaled_attention_logits # (batch_size, num_heads, max_seq_len, max_seq_len)

        if mask is not None:
            # mask as per the vanilla scaled dot-product attention.
            attention_weights += (mask * -1e9)
            # uncomment below if softmax below is not used.
            #attention_weights *= tf.cast(tf.math.equal(mask, 0), tf.float32)  # flip 0's and 1's and multiply. Has the same effect for when softmax isn't used.

        # comment/uncomment below if it matches the above masking.
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

if __name__ == "__main__":
    d_model, num_heads, max_seq_len, nm_gating = 10, 2, 3, True
    batch_size = 4
    q = tf.random.uniform((batch_size, max_seq_len, d_model))
    mask = None
    nm_inp_gating = tf.random.uniform((batch_size, max_seq_len, max_seq_len))
    q, k, v = q, q, q
    mha = MultiHeadAttention(d_model, num_heads, max_seq_len, nm_gating)
    output, attention = mha(q,k,v,nm_inp_gating,mask)
    print(f"Output.shape: {output.shape} \n"
          f"attention: {attention}")

