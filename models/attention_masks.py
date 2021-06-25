import tensorflow as tf

'''
Note: In the transformer architecture a 1 is multiplied by a large negative value, so a 1 in mask 
represents what we are masking out. 
'''

def create_padding_mask(x, padding_id=0):
    '''
    Function: create_padding_mask
    Description: Function that creates and returns a padding mask from the input sequence.
    Input:
        x: (torch.Tensor; [batch, seq_len]) The input sequence with 0 (padding_id) representing the tokens to pad.
    Return:
        (torch.Tensor; [batch, 1, 1, seq_len])

    From https://www.tensorflow.org/tutorials/text/transformer
    '''
    seq = tf.cast(tf.math.equal(x, padding_id), tf.float32)

    return seq[:, tf.newaxis, tf.newaxis, :] # add additional dimensions to the padded attention logits.

def create_look_ahead_mask(size):
    '''
    Function: create_look_ahead_mask
    Description: Function to mask the future tokens in a sequence.
    Input:
        size: (int) Parameter indicating the size. (e.g. the sequence length)
    Return:
        mask: (tf.Tensor; [seq_len, seq_len]|[size, size]

    From https://www.tensorflow.org/tutorials/text/transformer
    '''
    # Keep lower trianglar part only (i.e. set to a value of 0 and upper triangle part to 1's). # a 1 means that we are to pad it.
    # Note: padding id variation is irrelevant here as ones represent what we are to keep, and 0's what to pad out.
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask # (seq_len, seq_len)

def create_masks(inp, tar, padding_id=0):
    '''
    Function: create_masks
    Description: Create masks for the encoder and decoder.
    Input:
        inp: (tf.Tensor; [batch_size, inp_seq_len])
        tar: (tf.Tensor; [batch_size, tar_seq_len])
    Return:
        enc_padding_mask:
        dec_padding_mask:
        combined_mask:

    From https://www.tensorflow.org/tutorials/text/transformer
    '''
    enc_padding_mask = create_padding_mask(inp, padding_id=padding_id) # (batch, 1, 1, inp_seq_len)

    # Remember below is for the two block in one individual layer.
    # Used to mask the encoder outputs in the second attention block in the decoder.
    dec_padding_mask = create_padding_mask(inp, padding_id=padding_id) # (batch, 1, 1, inp_seq_len)

    # Used in first attention block in the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar, padding_id=padding_id)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

def create_combined_mask(x, padding_id=0):
    look_ahead_mask = create_look_ahead_mask(tf.shape(x)[1])
    dec_target_padding_mask = create_padding_mask(x, padding_id=padding_id)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return combined_mask

def relative_distance_penalty_mask(size):
    '''
    Implement this later. Aim is to replace the position encoding with this.
    '''
    pass

if __name__ == "__main__":
    batch, inp_seq_len, tar_seq_len = 4, 10, 12
    x = tf.ones((batch, inp_seq_len))
    y = tf.ones((batch, tar_seq_len))
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(x, y)

    print("enc_padding_mask:", enc_padding_mask.shape)
    print("combined_mask:", combined_mask.shape)
    print("dec_padding_mask:", dec_padding_mask.shape)
