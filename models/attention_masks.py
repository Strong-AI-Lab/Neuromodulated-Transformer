import tensorflow as tf

def create_padding_mask(x):
    '''
    Function: create_padding_mask
    Description: Function that creates and returns a padding mask from the input sequence.
    Input:
        x: (torch.Tensor; [batch, seq_len]) The input sequence with 0 representing the tokens to pad.
    Return:
        (torch.Tensor; [batch, 1, 1, seq_len])

    From https://www.tensorflow.org/tutorials/text/transformer
    '''
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)

    return seq[:, tf.newaxis, tf.newaxis, :] # add additional dimensions to the padded attention logits.

def create_look_ahead_mask(size):
    '''
    Function: create_look_ahead_mask
    Description: Function to mask the future tokens in a sequence.
    Input:
        size: (int) Parameter indicating the size.
    Return:
        mask: (tf.Tensor; [seq_len, seq_len]|[size, size]

    From https://www.tensorflow.org/tutorials/text/transformer
    '''
    # Keep lower trianglar part only (i.e. set to a value of 1 and upper triangle part to 0's).
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask # (seq_len, seq_len)

def create_masks(inp, tar):
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
    enc_padding_mask = create_padding_mask(inp) # (batch, 1, 1, inp_seq_len)

    # Remember below is for the two block in one individual layer.
    # Used to mask the encoder outputs in the second attention block in the decoder.
    dec_padding_mask = create_padding_mask(inp) # (batch, 1, 1, inp_seq_len)

    # Used in first attention block in the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

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
