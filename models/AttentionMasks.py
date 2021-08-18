'''
File name: AttentionMasks.py
Author: Kobe Knowles
Date created: 05/07/21
Data last modified: 18/08/21
Python Version: 3.6
Tensorflow version: 2
'''

import tensorflow as tf
import numpy as np

'''
Note: In the transformer architecture a 1 is multiplied by a large negative value, so a 1 in mask 
represents what we are masking out. 
'''

def create_padding_mask(x, padding_id=0):
    '''
    Function: create_padding_mask \n
    Description: Function that creates and returns a padding mask from the input sequence. \n
    Input:
        x: (torch.Tensor; [batch, seq_len]) The input sequence with 0 (padding_id) representing the tokens to pad.
    Return:
        (torch.Tensor; [batch, 1, 1, seq_len])
    Modified from https://www.tensorflow.org/tutorials/text/transformer
    '''
    seq = tf.cast(tf.math.equal(x, padding_id), tf.float32)

    return seq[:, tf.newaxis, tf.newaxis, :] # add additional dimensions to the padded attention logits.

def create_look_ahead_mask(size, num_aux_tok):
    '''
    Function: create_look_ahead_mask \n
    Description: Function to mask the future tokens in a sequence. Note that it is only applied to self-attention. \n
    Input:
        size: (int) Parameter indicating the size. (e.g. the sequence length) \n
        num_aux_tok: (int) The number of auxiliary tokens at the beginning where look_ahead padding is not to be applied. \n
    Return:
        mask: (tf.Tensor; [seq_len, seq_len]|[size, size]
    Modified from https://www.tensorflow.org/tutorials/text/transformer
    '''
    # Keep lower trianglar part only (i.e. set to a value of 0 and upper triangle part to 1's). # a 1 means that we are to pad it.
    # Note: padding id variation is irrelevant here as ones represent what we are to keep, and 0's what to pad out.

    mask = 1 - tf.linalg.band_part(tf.ones((size-num_aux_tok, size-num_aux_tok)), -1, 0)
    if num_aux_tok == 0: return mask # skip the extra processing as it isn't needed.

    #y = tf.zeros((size-num_aux_tok, num_aux_tok))
    #z = tf.zeros((num_aux_tok, size))

    look_ahead_top = tf.ones((num_aux_tok, size-num_aux_tok))
    global_attn_left = tf.zeros((size, num_aux_tok))

    mask = tf.concat([look_ahead_top, mask], axis=0) # (size-num_aux_tok, size)
    mask = tf.concat([global_attn_left, mask], axis=1) # (size, size)

    #############################
    ##### sample mask shape #####
    #############################
    # num_aux_tok = 3 -- Global attention to the auxiliary tokens only (e.g. first 3 rows and columns)
    # size = 10 -- look ahead padding is applied to these tokens as per normal.
    #[[0., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
    # [0., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
    # [0., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
    # [0., 0., 0., 0., 1., 1., 1., 1., 1., 1.],
    # [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
    # [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
    # [0., 0., 0., 0., 0., 0., 0., 1., 1., 1.],
    # [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
    # [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    # note: that if an aux token id is the same as what is set as the padding id, it will be overriden with a 1,
    # representing to pad at a global aux tok position.
    # just a note to be aware...

    return mask # (seq_len, seq_len)

def create_combined_mask(x, padding_id=0, num_aux_tok=0):
    '''
    Function: create_combined_mask \n
    Description: Creates a mask with both padded tokens and look ahead tokens. \n
    Input:
        x: (tf.Tensor; [batch_size, seq_len]) Input to create a mask for. \n
        padding_id: (int) The padding id that is to be padded in the input x.
            Defaults to 0. \n
        num_aux_tok: (int) The number of auxiliary tokens at the beginning where look_ahead padding is not to be applied. \n
    Return:
        combined_mask: (tf.Tensor; [batch_size, 1, seq_len, seq_len]) Mask to apply to the attention matrix.
    '''
    look_ahead_mask = create_look_ahead_mask(tf.shape(x)[1], num_aux_tok)
    dec_target_padding_mask = create_padding_mask(x, padding_id=padding_id)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return combined_mask

if __name__ == "__main__":
    size = 8
    num_aux_tok = 5
    batch = 1
    x = tf.random.uniform((batch, size), minval=1, maxval=5, dtype=tf.dtypes.int32)
    print(f"x tensor (input): {x}")
    mask = create_combined_mask(x, 0, num_aux_tok)
    print(f"mask.shape: {tf.squeeze(mask).shape} \n" \
          f"mask: {tf.squeeze(mask)}")
    padding_mask = create_padding_mask(x, padding_id=0)
    print(f"padding_mask.shape: {padding_mask.shape} \n" \
          f"padding_mask: {tf.squeeze(padding_mask)}")