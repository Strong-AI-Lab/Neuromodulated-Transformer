'''
File name: NMEncoder.py
Author: Kobe Knowles
Date created: 05/07/21
Data last modified: 12/08/21
Python Version: 3.6
Tensorflow version: 2
'''

import tensorflow as tf
import numpy as np

import sys
sys.path.append("..")

from models.NMMultiHeadAttention import NMMultiHeadAttention
from models.FeedForwardNetwork import * #FeedForwardNetwork
from models.PositionEncoding import get_angles, positional_encoding
from models.Encoder import EncoderLayer

class NMEncoderLayer(tf.keras.layers.Layer):
    '''
    Class: NMEncoderLayer \n
    Description: Implementation of a neuromodulation encoder layer. \n
    Attributes:
        max_seq_len: (int) The maximum sequence length of the input. (always pad to this length) \n
        mha: Multi-head attention (attends solely to itself). \n
        ffn: Feed forward network. \n
        layernorm1: Layernormalization layer, occuring after the multi-head attention layer (mha). \n
        layernorm2: Layernormalization layer, occuring after the feed-forward network (ffn). \n
        dropout1: Dropout layer which occurs after the multi-head attention layer and before the residual connection
            and layer normalization. \n
        dropout2: Dropout layer which occurs after the feed-forward layer and before the residual connection
            and layer normalization.
    '''
    def __init__(self, d_model, num_heads, dff, max_seq_len, rate=0.1, rel_pos_emb=True):
        '''
        Function: __init__ \n
        Description: Initializes a neuromodulation encoder layer with the passed parameters. \n
        Input:
            d_model: (int) The dimension of the transformer layer. \n
            num_heads: (int) The number of heads in the multi-head attention component. \n
            dff: (int) The dimension of the feed forward network layer. \n
            max_seq_len: (int) The maximum sequence length to be passed as input. \n
            rate: (float) The dropout rate for dropout layers throughout the layer.
                Defaults to 0.1. \n
            rel_pos_emb: (bool) True if relative position embeddings are to be used; False otherwise (i.e. absolute position embeddings)
        '''
        super(NMEncoderLayer, self).__init__()

        self.max_seq_len = max_seq_len

        self.mha = NMMultiHeadAttention(d_model, num_heads, max_seq_len, nm_gating=False, rel_pos_emb=rel_pos_emb)
        self.ffn = FeedForwardNetwork(init_vanilla_ffn(d_model, dff))

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for multi-head attention layer.
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for feed forward network.

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        '''
        Function: call \n
        Description: Overrides parent class's call function (i.e. one run through EncoderLayer). \n
        Input:
            x: (tf.Tensor; [batch_size, max_seq_len(_target), d_model]) Input tensor to the decoder layer. \n
            training: (bool) True if Dropout layers are to be in training mode; False otherwise. \n
            mask: (tf.Tensor) Mask for the multi-head attention layer. \n
        Return:
            out2: (tf.Tensor; [batch_size, max_seq_len, d_model])
            attn_weights: (tf.Tensor; [batch_size, num_heads, (max_)seq_len, (max_)seq_len])
        '''
        assert self.max_seq_len == x.shape[1], f"x.shape[1] should equal {self.max_seq_len}, got {x.shape[1]}!"

        x_ = self.layernorm1(x)
        attn1, attn_weights = self.mha(x_, x_, x_, nm_inp_gating=None, mask=mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = (x + attn1)

        out1_ = self.layernorm2(out1)
        out2 = self.ffn(out1_)
        out2 = self.dropout2(out2, training=training)
        out2 = out1 + out2

        return out2, attn_weights

class NMEncoder(tf.keras.layers.Layer):
    '''
    Class: NMEncoder \n
    Description: Implementation of the neuromodulation encoder in a transformer. \n
    Attributes: todo update below.
        num_layers: (int) The number of layers of the encoder (i.e. number of encoder layers). \n
        d_model: (int) The dimension of the encoder|transformer. \n
        max_seq_len: (int) The maximum sequence length of the input. (always pad to this length) \n
        mode: (string) Whether or not to process each layer one by one with multiple calls 'one', or all at once 'n_layers'. \n
        counter: (int) Current layer -1 that we are to process next, resets to 0 when finish processing the final layer. \n
        embedding: (tf.keras.layers.Embedding) The embedding layer which convers the input from ids to vectors. \n
        pos_encoding: (tf.Tensor) The positional embedding tensor to append to the input vectors to provide positional information
            (i.e. bring the vector of adjacent words closer to one another and distance words further away). \n
        encoder_layers: (list; DecoderLayer) A list of {num_layer} decoder layers. \n
        dropout: (tf.keras.layers.Dropout) A dropout layer to be applied the the input embeddings after positional encoding
            has been applied.
    '''
    def __init__(self, num_layers, d_model, num_heads, dff, max_seq_len, input_vocab_size, max_position_encoding=10000,
                 rate=0.1, parallel_layers={}, rel_pos_emb=True):
        '''
        Function: __init__ \n
        Description: Initialization of the encoder class. \n
        Input:
            num_layers: (int) The number of encoder layers in the encoder. \n
            d_model: (int) The dimension of the transformer layer. \n
            num_heads: (int) The number of heads in the multi-head attention component. \n
            dff: (int) The dimension of the feed forward network layer. \n
            max_seq_len: (int) The maximum sequence length to be passed as input. \n
            input_vocab_size: (int) The vocabulary size of the input (language). \n
            max_position_encoding: (int) The maximum position encoding to be generated (along sequence length dimension).
                It should greater than max_seq_len. Defaults to 10000. \n
            rate: (float) The dropout rate for dropout layers throughout the layer.
                Defaults to 0.1. \n
            parallel_layers: (dict) Dictionary containing layer names, and the Layer name to initialize.
                sample layer:
                    {
                    "nm_gate_attention": "init_attn_gate_ffn" # Here is a feed forward network to initialize the layer with.
                    "nm_gate_eol": "init_vanilla_ffn"
                    }
        '''
        super(NMEncoder, self).__init__()

        assert max_position_encoding >= max_seq_len, f"The max_position_encoding ({max_position_encoding}) should be" \
                                                     f"greater than max_seq_len ({max_seq_len})!"

        self.num_layers = num_layers
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.rel_pos_emb = rel_pos_emb

        # possible values are ["n_layers", "one"]
        self.mode = "n_layers"
        self.counter = 0

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_position_encoding, d_model)

        self.encoder_layers = [NMEncoderLayer(d_model, num_heads, dff, max_seq_len,
                                            rate, rel_pos_emb=rel_pos_emb) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

        self.parallel_layers = {}

        for key, layer in parallel_layers.items(): # modify layer to be a list of size 2. [layer name (str), The number of layers (int)]
            if layer[0] == "EncoderLayer":
                self.parallel_layers[key] = [[NMEncoderLayer(d_model, num_heads, dff, max_seq_len, rate) for _ in range(layer[1])]]
                # below corresponds to the generation of output predictions used as an auxiliary loss.
                self.parallel_layers[key].append(tf.keras.layers.Dense(self.d_model) if layer[2] else None)
            elif layer[0] == "NMEncoderLayerNoRC":
                self.parallel_layers[key] = [[NMEncoderLayer(d_model, num_heads, dff, max_seq_len, rate) for _ in range(layer[1]-1)] + \
                                            [NMEncoderLayerNoRC(d_model, num_heads, dff, max_seq_len, rate)]]
                # below corresponds to the generation of output predictions used as an auxiliary loss.
                self.parallel_layers[key].append(tf.keras.layers.Dense(self.d_model) if layer[2] else None)
            elif layer[0] == "GateLayerAttn":
                self.parallel_layers[key] = [[NMEncoderLayer(d_model, num_heads, dff, max_seq_len, rate) for _ in range(layer[1]-1)] + \
                                            [GateLayerAttn(d_model, num_heads, dff, max_seq_len, rate)]]
                # below corresponds to the generation of output predictions used as an auxiliary loss.
                self.parallel_layers[key].append(tf.keras.layers.Dense(self.d_model) if layer[2] else None)
            elif layer[0] == "MetacognitionSequenceLayer":
                self.parallel_layers[key] = [[NMEncoderLayer(d_model, num_heads, dff, max_seq_len, rate) for _ in range(layer[1]-1)] + \
                                            [MetacognitionSequenceLayer(d_model, num_heads, dff, max_seq_len, rate)]]
                # below corresponds to the generation of output predictions used as an auxiliary loss.
                self.parallel_layers[key].append(tf.keras.layers.Dense(self.d_model) if layer[2] else None)
            elif layer[0] == "MetacognitionSingleLayer":
                self.parallel_layers[key] = [[NMEncoderLayer(d_model, num_heads, dff, max_seq_len, rate) for _ in range(layer[1]-1)] + \
                                            [MetacognitionSingleLayer(d_model, num_heads, dff, max_seq_len, rate)]]
                # below corresponds to the generation of output predictions used as an auxiliary loss.
                self.parallel_layers[key].append(tf.keras.layers.Dense(self.d_model) if layer[2] else None)

    def call(self, x, training, mask, restrictions=[]):
        '''
        Function: call \n
        Description: Overrides the parent class' call function (i.e. run through the encoder). \n
        Input:
            x: (tf.Tensor [int]; [batch_size, max_seq_len(_input)]) Input tensor to the decoder layer. \n
            training: (bool) True if Dropout layers are to be in training mode; False otherwise. \n
            mask: (tf.Tensor) Mask for multi-head attention layer 1. \n
            restrictions: (list; string) List of the last layer names which are not to be computed.
        Return:
            x_dict: (dict; tuple; tf.Tensor; [batch_size, varies, varies] (output from a layer) | tf.Tensor;
                [batch_size, num_heads, max_seq_len, max_seq_len] (attention weights for that layer) or dict of tensors of this shape) \n
        '''
        assert x.shape[1] == self.max_seq_len, f"The tensor x should have a dimension 1 (python indices) size of {self.max_seq_len}." \
                                           f"Got {x.shape[1]} instead!"
        seq_len = x.shape[1]

        if self.counter == 0:
            x = self.embedding(x)

            if not self.rel_pos_emb:
                x *= tf.math.sqrt(tf.cast(self.d_model, tf.dtypes.float32))
                x += self.pos_encoding[:, :seq_len, :]
                x /= tf.math.sqrt(tf.cast(self.d_model, tf.dtypes.float32))
            x = self.dropout(x, training=training)

        attention_weights = dict()
        if self.mode == "n_layers":
            for i in range(self.num_layers):
                x, block1 = self.encoder_layers[i](x, training, mask)
                attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            self.reset_counter()  # make sure that the counter is 0 for the next time this class is called.
        elif self.mode == "one":
            x, block1 = self.encoder_layers[i](x, training, mask)
            attention_weights[f'decoder_layer{self.counter + 1}_block1'] = block1
            self.increment_counter()  # note: when at the final layer, this function resets it.
        else:
            raise Exception(f"Invalid mode parameter. Got mode: {self.mode}! \n"
                            f"It should be equal to \"n_layers\" or \"one\"!")
        x_dict = dict()
        if self.counter == 0: # i.e. we are at the end and it has been reset to zero.
            for key, layer in self.parallel_layers.items():
                if key in restrictions: continue
                #if key == "nm_eol_gate": x_dict[key] = layer(x, training, mask) # this will be a tuple containing the output (x, attn_weights)
                #else: x_dict[key] = layer(tf.stop_gradient(x), training, mask) # only want the baseline network to be updated once.
                # todo iterate through each layer
                out, out_no_grad, attn_weights= x, tf.stop_gradient(x), []
                for l in layer[0]:
                    out, aw = l[0](out, training, mask) # out.shape = (batch_size, seq_len, varies)
                    attn_weights.append(aw)
                aux_pred = None
                # run through again but with gradient cutoff from `base' layers
                if layer[1] is not None:
                    for l in layer[0]:
                        out_no_grad, aw = l[0](out_no_grad, training, mask) # out.shape = (batch_size, seq_len, varies)
                    aux_pred = l[1](out_no_grad) # modifies the last dimension to d_mod
                x_dict[key] = (out, attn_weights, aux_pred)  # this will be a tuple containing the output (x, attn_weights)
            if len(x_dict.keys()) == 0: x_dict["default"] = (x, attention_weights) # i.e. return x if the dictionary is empty.
        return x_dict

    def increment_counter(self):
        self.counter += 1
        if self.counter == self.num_layers:
            self.reset_counter()

    def reset_counter(self):
        self.counter = 0

class NMEncoderLayerNoRC(tf.keras.layers.Layer):
    '''
    Class: NMEncoderLayerNoRC \n
    Description: Implementation of a neuromodulation encoder layer with the last residual connection removed. \n
    Attributes:
        max_seq_len: (int) The maximum sequence length of the input. (always pad to this length) \n
        mha: Multi-head attention (attends solely to itself). \n
        ffn: Feed forward network. \n
        layernorm1: Layernormalization layer, occuring after the multi-head attention layer (mha). \n
        layernorm2: Layernormalization layer, occuring after the feed-forward network (ffn). \n
        dropout1: Dropout layer which occurs after the multi-head attention layer and before the residual connection
            and layer normalization. \n
        dropout2: Dropout layer which occurs after the feed-forward layer and before the residual connection
            and layer normalization.
    '''
    def __init__(self, d_model, num_heads, dff, max_seq_len, rate=0.1):
        '''
        Function: __init__ \n
        Description: Initializes a neuromodulation encoder layer with the passed parameters. \n
        Input:
            d_model: (int) The dimension of the transformer layer. \n
            num_heads: (int) The number of heads in the multi-head attention component. \n
            dff: (int) The dimension of the feed forward network layer. \n
            max_seq_len: (int) The maximum sequence length to be passed as input. \n
            rate: (float) The dropout rate for dropout layers throughout the layer.
                Defaults to 0.1. \n
        '''
        super(NMEncoderLayerNoRC, self).__init__()

        self.max_seq_len = max_seq_len

        self.mha = NMMultiHeadAttention(d_model, num_heads, max_seq_len, nm_gating=False)
        self.ffn = FeedForwardNetwork(init_vanilla_ffn(d_model, dff))

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for multi-head attention layer.
        #self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for feed forward network.

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        '''
        Function: call \n
        Description: Overrides parent class's call function (i.e. one run through EncoderLayer). \n
        Input:
            x: (tf.Tensor; [batch_size, max_seq_len(_target), d_model]) Input tensor to the decoder layer. \n
            training: (bool) True if Dropout layers are to be in training mode; False otherwise. \n
            mask: (tf.Tensor) Mask for the multi-head attention layer. \n
        Return:
            out2: (tf.Tensor; [batch_size, max_seq_len, d_model])
            attn_weights: (tf.Tensor; [batch_size, num_heads, (max_)seq_len, (max_)seq_len])
        '''
        assert self.max_seq_len == x.shape[1], f"x.shape[1] should equal {self.max_seq_len}, got {x.shape[1]}!"

        x_ = self.layernorm1(x)
        attn1, attn_weights = self.mha(x_, x_, x_, nm_inp_gating=None, mask=mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = x + attn1

        out2 = self.ffn(out1)
        out2 = self.dropout2(out2, training=training) # allowed here unlike other layers.
        #out2 = self.layernorm2(out2)

        return out2, attn_weights

class GateLayerAttn(tf.keras.layers.Layer):
    '''
    Class: GateLayerAttn \n
    Description: Implementation of an encoder layer with the gated supported feed-forward network (for attn). \n
    Attributes:
        max_seq_len: (int) The maximum sequence length of the input. (always pad to this length) \n
        mha: Multi-head attention (attends solely to itself). \n
        ffn: Feed forward network. \n
        layernorm1: Layernormalization layer, occuring after the multi-head attention layer (mha). \n
        layernorm2: Layernormalization layer, occuring after the feed-forward network (ffn). \n
        dropout1: Dropout layer which occurs after the multi-head attention layer and before the residual connection
            and layer normalization. \n
        dropout2: Dropout layer which occurs after the feed-forward layer and before the residual connection
            and layer normalization.
    '''
    def __init__(self, d_model, num_heads, dff, max_seq_len, rate=0.1):
        '''
        Function: __init__ \n
        Description: Initializes a neuromodulation encoder layer with the passed parameters. \n
        Input:
            d_model: (int) The dimension of the transformer layer. \n
            num_heads: (int) The number of heads in the multi-head attention component. \n
            dff: (int) The dimension of the feed forward network layer. \n
            max_seq_len: (int) The maximum sequence length to be passed as input. \n
            rate: (float) The dropout rate for dropout layers throughout the layer.
                Defaults to 0.1. \n
        '''
        super(GateLayerAttn, self).__init__()

        self.max_seq_len = max_seq_len

        self.mha = NMMultiHeadAttention(d_model, num_heads, max_seq_len, nm_gating=False)
        self.ffn = FeedForwardNetwork(init_attn_gate_ffn(dff, max_seq_len))

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for multi-head attention layer.
        #self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for feed forward network.

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        '''
        Function: call \n
        Description: Overrides parent class's call function (i.e. one run through EncoderLayer). \n
        Input:
            x: (tf.Tensor; [batch_size, max_seq_len(_target), d_model]) Input tensor to the decoder layer. \n
            training: (bool) True if Dropout layers are to be in training mode; False otherwise. \n
            mask: (tf.Tensor) Mask for the multi-head attention layer. \n
        Return:
            out2: (tf.Tensor; [batch_size, max_seq_len, d_model])
            attn_weights: (tf.Tensor; [batch_size, num_heads, (max_)seq_len, (max_)seq_len])
        '''
        assert self.max_seq_len == x.shape[1], f"x.shape[1] should equal {self.max_seq_len}, got {x.shape[1]}!"

        x_ = self.layernorm1(x)
        attn1, attn_weights = self.mha(x_, x_, x_, nm_inp_gating=None, mask=mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = x + attn1

        out2 = self.ffn(out1)
        out2 = self.dropout2(out2, training=training)
        # residual connection removed here as never want these values to equal 0 and the residual connection modifies the dimenisons.

        return out2, attn_weights

class MetacognitionSequenceLayer(tf.keras.layers.Layer):
    '''
    Class: MetacognitionSequenceLayer \n
    Description: Implementation of an encoder layer with the metacognition sequence feed-forward network. \n
    Attributes:
        max_seq_len: (int) The maximum sequence length of the input. (always pad to this length) \n
        mha: Multi-head attention (attends solely to itself). \n
        ffn: Feed forward network. \n
        layernorm1: Layernormalization layer, occuring after the multi-head attention layer (mha). \n
        layernorm2: Layernormalization layer, occuring after the feed-forward network (ffn). \n
        dropout1: Dropout layer which occurs after the multi-head attention layer and before the residual connection
            and layer normalization. \n
        dropout2: Dropout layer which occurs after the feed-forward layer and before the residual connection
            and layer normalization.
    '''
    def __init__(self, d_model, num_heads, dff, max_seq_len, rate=0.1):
        '''
        Function: __init__ \n
        Description: Initializes a neuromodulation encoder layer with the passed parameters. \n
        Input:
            d_model: (int) The dimension of the transformer layer. \n
            num_heads: (int) The number of heads in the multi-head attention component. \n
            dff: (int) The dimension of the feed forward network layer. \n
            max_seq_len: (int) The maximum sequence length to be passed as input. \n
            rate: (float) The dropout rate for dropout layers throughout the layer.
                Defaults to 0.1. \n
        '''
        super(MetacognitionSequenceLayer, self).__init__()

        self.max_seq_len = max_seq_len

        self.mha = NMMultiHeadAttention(d_model, num_heads, max_seq_len, nm_gating=False)
        self.ffn = FeedForwardNetwork(init_metacognition_sequence_ffn(dff))

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for multi-head attention layer.
        #self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for feed forward network.

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        '''
        Function: call \n
        Description: Overrides parent class's call function (i.e. one run through EncoderLayer). \n
        Input:
            x: (tf.Tensor; [batch_size, max_seq_len(_target), d_model]) Input tensor to the decoder layer. \n
            training: (bool) True if Dropout layers are to be in training mode; False otherwise. \n
            mask: (tf.Tensor) Mask for the multi-head attention layer. \n
        Return:
            out2: (tf.Tensor; [batch_size, max_seq_len, d_model])
            attn_weights: (tf.Tensor; [batch_size, num_heads, (max_)seq_len, (max_)seq_len])
        '''
        assert self.max_seq_len == x.shape[1], f"x.shape[1] should equal {self.max_seq_len}, got {x.shape[1]}!"

        x_ = self.layernorm1(x)
        attn1, attn_weights = self.mha(x_, x_, x_, nm_inp_gating=None, mask=mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = x + attn1

        out2 = self.ffn(out1)
        out2 = self.dropout2(out2, training=training)
        # residual and dropout removed here as never want these values to equal 0 and the residual connection modifies the dimenisons???

        return out2, attn_weights

class MetacognitionSingleLayer(tf.keras.layers.Layer):
    '''
    Class: MetacognitionSingleLayer \n
    Description: Implementation of an encoder layer with the metacognition single feed-forward network. \n
    Attributes:
        max_seq_len: (int) The maximum sequence length of the input. (always pad to this length) \n
        mha: Multi-head attention (attends solely to itself). \n
        ffn: Feed forward network. \n
        layernorm1: Layernormalization layer, occuring after the multi-head attention layer (mha). \n
        layernorm2: Layernormalization layer, occuring after the feed-forward network (ffn). \n
        dropout1: Dropout layer which occurs after the multi-head attention layer and before the residual connection
            and layer normalization. \n
        dropout2: Dropout layer which occurs after the feed-forward layer and before the residual connection
            and layer normalization.
    '''

    def __init__(self, d_model, num_heads, dff, max_seq_len, rate=0.1):
        '''
        Function: __init__ \n
        Description: Initializes a neuromodulation encoder layer with the passed parameters. \n
        Input:
            d_model: (int) The dimension of the transformer layer. \n
            num_heads: (int) The number of heads in the multi-head attention component. \n
            dff: (int) The dimension of the feed forward network layer. \n
            max_seq_len: (int) The maximum sequence length to be passed as input. \n
            rate: (float) The dropout rate for dropout layers throughout the layer.
                Defaults to 0.1. \n
        '''
        super(MetacognitionSingleLayer, self).__init__()

        self.max_seq_len = max_seq_len

        self.mha = NMMultiHeadAttention(d_model, num_heads, max_seq_len, nm_gating=False)
        self.ffn = FeedForwardNetwork(init_metacognition_single_ffn(dff))

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for multi-head attention layer.
        #self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # for feed forward network.

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, traMetacognitionSingleLayerining, mask):
        '''
        Function: call \n
        Description: Overrides parent class's call function (i.e. one run through EncoderLayer). \n
        Input:
            x: (tf.Tensor; [batch_size, max_seq_len(_target), d_model]) Input tensor to the decoder layer. \n
            training: (bool) True if Dropout layers are to be in training mode; False otherwise. \n
            mask: (tf.Tensor) Mask for the multi-head attention layer. \n
        Return:
            out2: (tf.Tensor; [batch_size, max_seq_len, d_model])
            attn_weights: (tf.Tensor; [batch_size, num_heads, (max_)seq_len, (max_)seq_len])
        '''
        assert self.max_seq_len == x.shape[1], f"x.shape[1] should equal {self.max_seq_len}, got {x.shape[1]}!"

        x_ = self.layernorm1(x)
        attn1, attn_weights = self.mha(x_, x_, x_, nm_inp_gating=None, mask=mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = x + attn1

        out2 = self.ffn(out1)
        out2 = self.dropout2(out2, training=training)
        # residual and dropout removed here as never want these values to equal 0 and the residual connection modifies the dimenisons.

        return out2, attn_weights

if __name__ == "__main__":
    num_layers, d_model, num_heads, dff, max_seq_len, input_vocab_size, max_position_encoding, rate = 6, 100, 10, 200, 6, 124, 1007, 0.1
    batch_size = 2
    # num_layers, d_model, num_heads, dff, max_seq_len, input_vocab_size, max_position_encoding=10000,
    #                  rate=0.1, parallel_layers={}
    #                    "nm_attn_gate_lm",
    #                    "nm_eol_gate_lm",
    #                    "unk_reading_strategy",
    #                    "highlighting_reading_strategy",
    #                    "aoi_reading_strategy",
    #                    "re_read_reading_strategy",
    #                    "paraphrase_reading_strategy",
    #                    "summarization_reading_strategy"
    # MetacognitionSingleLayer
    # MetacognitionSequenceLayer
    # GateLayerAttn
    # NMEncoderLayer
    parallel_layers = {}
    parallel_layers["nm_attn_gate_lm"] = "GateLayerAttn"
    parallel_layers["nm_eol_gate_lm"] = "NMEncoderLayerNoRC" #"EncoderLayer"
    parallel_layers["unk_reading_strategy"] = "MetacognitionSequenceLayer"
    parallel_layers["highlighting_reading_strategy"] = "MetacognitionSingleLayer"
    parallel_layers["aoi_reading_strategy"] = "MetacognitionSingleLayer"
    parallel_layers["re_read_reading_strategy"] = "MetacognitionSingleLayer"
    parallel_layers["paraphrase_reading_strategy"] = "MetacognitionSingleLayer"
    parallel_layers["summarization_reading_strategy"] = "MetacognitionSingleLayer"

    nm_encoder = NMEncoder(num_layers, d_model, num_heads, dff, max_seq_len, input_vocab_size,
                           max_position_encoding, rate, parallel_layers)
    nm_encoder.mode = "n_layers"
    # x, training, mask, restrictions=[]
    x = tf.random.uniform((batch_size, max_seq_len))
    training, mask = True, None
    restrictions = ["unk_reading_strategy"]
    dict_ = nm_encoder(x, training, mask, restrictions)
    print(len(dict_.keys()))
    #print(dict_)
    for key, value in dict_.items():
        print(f"{key}: {value[0].shape} \t {value[1].shape}")

    print(f"nm_attn_gate_lm: {dict_['nm_attn_gate_lm'][0]}")
    print(f"unk: {dict_['unk_reading_strategy'][0]}")
    print(f"highlighting: {dict_['paraphrase_reading_strategy'][0]}")
