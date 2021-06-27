import tensorflow as tf
import numpy as np

class NMPWFeewForwardNetwork(tf.keras.layers.Layer):
    '''
    Class: NMPWFeedForwardNetwork
    Description: Neuromodulation supported point-wise feed forward network implementation.
    Input:
        ffn_dict: (dict) Dictionary of dictionaries containing the parameters for each parallel feed forward layer.
            note: This dictionary contains d_model and dff in parent class, hence removed.
    '''
    def __init__(self, ffn_dict):
        super(NMPWFeewForwardNetwork, self).__init__()

        self.ffn_dict = ffn_dict

        # holds all of the parallel feed forward layers.
        self.ffn_networks = dict()
        for i, name in enumerate(self.ffn_dict.keys()):
            ffn_ = tf.keras.Sequential()
            num_layers = self.ffn_dict[name]["num_layers"]
            for j in range(num_layers):  # Represents each layer in an individual feed forward layer.

                if self.ffn_dict[name]["W_"+str(j+1)+"_transpose?"]: # check here first if we are to transpose the input.
                    # this essentially transposes the input.
                    ffn_.add(tf.keras.layers.Permute((2,1))) # The batch dimension is left out.

                activation_function = self.ffn_dict[name]["W_"+str(j+1)+"_activation_function"]
                if activation_function != 'none':
                    ffn_.add(tf.keras.layers.Dense(self.ffn_dict[name]["W_"+str(j+1)+""], activation=activation_function))
                else:
                    ffn_.add(tf.keras.layers.Dense(self.ffn_dict[name]["W_" + str(j + 1) + ""]))
                #TODO: just marking this here.
                if self.ffn_dict[name]["W_"+str(j+1)+"_layernorm?"]:
                    #print("REACH LN ffn_dict initialization!")
                    ffn_.add(tf.keras.layers.LayerNormalization(epsilon=1e-6))

            self.ffn_networks[name] = ffn_

        #print(self.ffn_networks)

    def call(self, input):
        '''
        Function: call
        Purpose: Runs the point-wise feed forward network with the given input on all ffn_networks.
        Input:
            input: (tf.Tensor; [batch_size, seq_len, d_model])
        Return:
            output: (dict) Containing the network name and output)
        '''
        assert len(input.shape) == 3, "The number of dimensions must be equal to 3!"

        output = dict()
        for i, name in enumerate(self.ffn_networks.keys()):
            output[name] = self.ffn_networks[name](input)

        return output # dictionary of (batch_size, seq_len_nm|seq_len_q|user_defined, d_model|seq_len_k|user_defined)

class NMMultiHeadAttention(tf.keras.layers.Layer):
    '''
    Class: MultiHeadAttention
    Description: Implementation of multi-head attention to be used in by the Transformer class.
    Inputs:
        d_model: (int) The dattn_weights_block1imension  of the model.
        num_heads: (int) The number of heads in the Multi head attention layer.
        max_seq_len:
        use_nm: (boolean) True if we are replacing the scaled dot product attention with gating from
            associated feed forward layer; otherwise False, calculate as per normal.
    '''
    def __init__(self, d_model, num_heads, max_seq_len=512, nm_mha=False):
        super(NMMultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.nm_mha = nm_mha
        #self.use_nm = use_nm # parameter is useless.
        self.max_seq_len = max_seq_len

        # Must be a multiple of the number of heads.
        assert d_model % self.num_heads == 0

        # the dimension of each individual head.
        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        #self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        if self.nm_mha:
            # Note: the constraint is the sequence length needs to be always equal to max_seq_len.
            self.nm_layers_logit = [tf.keras.layers.Dense(self.max_seq_len) for _ in range(self.num_heads)]

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        '''
        Function: split_heads
        Description: Splits the last dimension into (num_heads, depth).
            Then transpose the tensor to the following permutation, [0,2,1,3].
        Input:
            x: (tf.Tensor: [batch_size, enc_seq_len, d_model]) Input tensor.
            batch_size: (int) Batch size during training.
        Return:
            (tf.tensor; [batch_size, num_heads, enc_seq_len, depth])
        '''
        # num_heads * depth = d_model
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0,2,1,3])

    def call(self, v, k, q, gate_inp_logits=None, mask=None):
        '''
        Function: call
        Purpose: Code to be executed when the keras layer class is called.
        Input:
            v: (tf.Tensor; [batch_size, seq_len_v, word_dim|d_model]) Value input tensor.
            k: (tf.Tensor; [batch_size, seq_len_k, word_dim|d_model]) Key input tensor.
            q: (tf.Tensor; [batch_size, seq_len_q, word_dim|d_model]) Query input tensor.
            gate_inp_logits: (tf.Tensor; [batch_size])
            mask: (tf.Tensor; [batch_size, ..., ..., seq_len]) Mask to apply to the attention matrix.
                Note: ... means that the exact dimension size can vary from encoder to decoder.
                Encoder: (batch_size, 1, 1, seq_len)
                Decoder 1st block: (batch_size, 1, seq_len, seq_len)
                Decoder 2nd block: (batch_size, 1, 1, seq_len)
                Python broadcasting allows this to work as intended.
        Return:
            output: (tf.Tensor; [batch, seq_len_q, d_model])
            attention_weights: (tf.Tensor; [batch_size, num_heads, seq_len_q, seq_len_k])
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

        if gate_inp_logits is not None:
            #assert gate_inp_logits is not None, "gate_inp_logits should not be None when use_nm is set to True."
            if not self.nm_mha:
                raise Exception("gate_input should be None if self.nm_mha is set to False")
            scaled_attention, attention_weights = self.neuromodulation_attention(q, k, v, gate_inp_logits, mask)
        else:
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
            q: (tf.Tensor; [batch_size, num_heads, seq_len_q, depth])
            k: (tf.Tensor; [batch_size, num_heads, seq_len_k, depth])
            v: (tf.Tensor; [batch_size, num_heads, seq_len_v, depth])
            mask: (tf.Tensor; [batch_size, ..., ..., seq_len]) Mask to apply to the attention matrix.
                Note: ... means that the exact dimension size can vary from encoder to decoder.
                Encoder: (batch_size, 1, 1, seq_len)
                Decoder 1st block: (batch_size, 1, seq_len, seq_len)
                Decoder 2nd block: (batch_size, 1, 1, seq_len)
                Python broadcasting allows this to work as intended.
        Return:
            output: (tf.Tensor; [batch_size, num_heads, seq_len_q, depth_v])
            attention_weights: (tf.Tensor; [batch_size, num_heads, seq_len_q, seq_len_k])
        '''

        matmul_qk = tf.matmul(q, k, transpose_b = True) # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask + -1e9)

        # softmax is normalized on the last axis (seq_len_k) so the scores add to 1.
        # Technically you could do the other axis as long as it is consistent through out training.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v) # (..., seq_len_q, depth_v)

        return output, attention_weights

    def neuromodulation_attention(self, q, k, v, gate_inp_logits, mask = None):
        '''
        Function: neuromodulated_attention
        Purpose: Perform scaled dot-product attention with a query, key, value and a mask as input.
            Specifically calculate the attention weights and multiply them with the value, v.
        Input:
            note: seq_len_q == seq_len_k for all practical purposes (only this is supported currently).
            q: (tf.Tensor; [batch_size, num_heads, seq_len_q, depth])
            k: (tf.Tensor; [batch_size, num_heads, seq_len_k, depth])
            v: (tf.Tensor; [batch_size, num_heads, seq_len_v, depth])
            gate_inp_logits: (tf.Tensor; [batch_size, seq_len_q, seq_len_k])
            mask: (tf.Tensor; [batch_size, ..., ..., seq_len]) Mask to apply to the attention matrix.
                Note: ... means that the exact dimension size can vary from encoder to decoder.
                Encoder: (batch_size, 1, 1, seq_len)
                Decoder 1st block: (batch_size, 1, seq_len, seq_len)
                Decoder 2nd block: (batch_size, 1, 1, seq_len)
                Python broadcasting allows this to work as intended.
        Return:
            output: (tf.Tensor; [batch_size, num_heads, seq_len_q, depth_v])
            attention_weights: (tf.Tensor; [batch_size, num_heads, seq_len_q, seq_len_k])
        '''

        assert q.shape[2] == k.shape[2], "The key's and queries sequence length should be the same. (" \
                                         "This is why it can't be applied to decoder-encoder attention layer as they differ)"
        assert self.nm_mha, "nm_mha should be set to True instead of False!"

        matmul_qk = tf.matmul(q, k, transpose_b = True) # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # TODO comment below what is being done.
        gate_input_head = None
        # below `expands' the output for each head.
        for i in range(self.num_heads):
            # get a difference for each head by multiplying by a separate dense layer.
            z = self.nm_layers_logit[i](gate_inp_logits)
            # expand axis = 1, is to match dimension with scaled attention logits (i.e. one for each head).
            # This dimension will be equal to num_heads.
            if gate_input_head is None:
                gate_input_head = tf.expand_dims(z, axis=1)
                #print("test \n", gate_input_head.shape, "\n")
            else:
                gate_input_head = tf.concat([gate_input_head, tf.expand_dims(z, axis=1)], 1)
        #gate_input_head.shape == (batch_size, num_heads, max_seq_len_dec, max_seq_len_dec)

        #print("\n", gate_input_head.shape,"\n")

        # add the mask to the scaled tensor.
        assert scaled_attention_logits.shape == gate_input_head.shape, f"Dimensions don't match!\nscaled_attention_logits:{scaled_attention_logits.shape}\ngate_input_head:{gate_input_head.shape}" \
                                                                       f"The Query and Key are the following values: Q{q} \n K{k}"

        if mask is not None:
            gate_input_head += (mask * -1e9) # mask the gated head aswell before softmax is performed so weight isn't given to invalid tokens.

        gate_inp = tf.nn.softmax(gate_input_head, axis=-1)

        attention_weights = gate_inp * scaled_attention_logits # (batch_size, seq_len_q, seq_len_k)

        if mask is not None:
            # uncomment below if softmax to the gated inputs.
            attention_weights += (mask * -1e9) # note: the mask locations should be zero, so applying this here if softmax is to come is necessary.
            # uncomment below if softmax is not to be used. # to be fair because gate_inp is
            # padded below it has no effect whatsoever as it would be multiplied by zero anyway.
            # flip zeroes to ones and ones to zeroes.
            # attention_weights *= tf.cast(tf.math.equal(mask, 0), tf.float32) # flip 0's and 1's and multiply. Has the same effect for when softmax isn't used.

        # here attention is entirely replaced with gating from the neuromodulation encoder.
        # comment out below if want the softmax as per the original scaled dot product attention.
        # best to have below for stable training.
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)

        output = tf.matmul(attention_weights, v) # (..., seq_len_q, depth_v)

        return output, attention_weights

def create_ffn_dict(names, *list_arg):
    '''
    Function: create_ffn_dict
    Description: Initializes and returns dictionary containing parameters for feed forward network initialization.
    Input:
        names: (list of strings) Each string in the list is the name of one parallel feed forward layer.
        *list_arg: (list; varying number of lists) Each list represents parameters for a neural network.
            e.g. position 0: (int) number of layers (e.g. 3 layers in the following three positions.
                 position 1: (list) layer 1 params.
                    [dff, 'relu', (boolean; transpose)]
                 position 2: (list) layer 2 params.
                    [d_model, 'none', (booblean; transpose)]
                 position 3: (list) layer 3 params.
                    [max_seq_len, 'softmax', (boolean; transpose)]
                    note: transpose is done at the beginning of the position.
                .
                .
                .
                position n: (list) layer n params.
                    [dimension (int), activation function (string), transpose (boolean)]
    Return:
        (dict)
    '''
    counter = 0
    for val in list_arg:
        counter += 1

    assert counter == len(names), "Parameter sizes do not match!"
    assert names[0] == "default", "The first parameter in the list names, must be called 'default'."

    ffn_dict = dict()
    for name in names:
        ffn_dict[name] = None

    # keeps track of iteration through names list.
    #counter = 0
    for i, val in enumerate(list_arg):
        dict_ = dict()
        for j in range(len(val)):
            if j == 0:
                #print("\n",val,"\n")
                assert isinstance(val[0], int), "val should be an integer and is not!"
                dict_["num_layers"] = val[0] # this is an integer
                continue

            # val is a list here.
            #print("\n", val, len(val), "\n")
            assert len(val) == dict_["num_layers"]+1, "num_layers parameter doesn't match the input!"
            assert len(val[j]) == 3 or len(val[j]) == 4, "length of this list should be 3 or 4!"

            dict_["W_"+str(j)] = val[j][0]
            dict_["W_"+str(j)+"_activation_function"] = val[j][1]
            assert isinstance(val[j][2], bool), f"The third element should be of type bool, its current type is {type(val[j][2])}"
            dict_["W_"+str(j)+"_transpose?"] = val[j][2]
            # below if statement is here so old unit tests aren't broken.
            if len(val[j]) >= 4:
                assert isinstance(val[j][3], bool), f"The fourth element should be of type bool, its current type is {type(val[j][3])}"
                dict_["W_"+str(j)+"_layernorm?"] = val[j][3]

        ffn_dict[names[i]] = dict_

    return ffn_dict

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                                 np.arange(d_model)[np.newaxis, :],
                                 d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)