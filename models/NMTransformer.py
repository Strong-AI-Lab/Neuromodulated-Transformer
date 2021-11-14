import tensorflow as tf
import numpy as np

import sys
sys.path.append("..")

from models.AttentionMasks import *
from models.Decoder import Decoder, DecoderLayer
from models.OutputDecoder import OutputDecoder
from models.NMDecoder import NMDecoder
from models.PositionEncoding import get_angles, positional_encoding
from models.MultiHeadAttention import MultiHeadAttention
from models.FeedForwardNetwork import *
from models.config_model import *

class NMTransformer(tf.keras.Model):
    '''
    An implementation of version 4 of the neuromodulated transformer with reading strategy/metacognition support.
    '''

    def __init__(self, num_layers_vanilla, num_layers_nm, num_layers_mc, num_layers_output,
                 d_model, num_heads, dff, input_vocab_size, output_vocab_size, max_position_encoding=10000,
                 max_seq_len_dec=768, num_aux_toks=9, mask_strategy="default",
                 rate=0.1, parallel_layers=None, output_layers=None,
                 aux_tok_output_layer_map={}, mode_ids={}):
        '''
        :param num_layers_vanilla: (int) An integer representing the number of layers in the "vanilla set".
        :param num_layers_nm: (int) An integer representing the number of layers in the "neuormodulation set".
        :param num_layers_mc: (int) An integer representing the number of layers in the "metacognition sets".
        :param num_layers_output: (int) An integer representing the number of layers in the "output sets".
        :param d_model: (int) An integer specifying dimension of the NMT.
        :param num_heads: (int) An integer specifying the number of heads in each multi-head attention layer.
        :param dff: (int) An integer specifying the dimension of all feed-forward layers.
        :param input_vocab_size: (int) An integer specifying the number vocabulary size of the input language.
        :param output_vocab_size: (int) An integer specifying the number vocabulary size of the output language.
        :param max_position_encoding: (int) An integer specifying the maximum position encoding length for absolute fixed
            position embeddings.
        :param max_seq_len_dec: (int) An integer specifying the maximum sequence length of the input.
        :param num_aux_toks: (int) An integer specifying the number of auxiliary tokens -> used to get max_seq_len_nm.
        :param mask_strategy: (str) A string specifying the mask strategy for the NMT, currenty not supported.
        :param rate: (float) A floating point number specifying the dropout rate for all dropout layers.
        :param parallel_layers: (list) A list of strings specifying the name of each parallel layer in the neuromodulation set.
        :param output_layers: (list) A list of strings specifying the name of each parallel output set (there are multiple output sets).
        :param aux_tok_output_layer_map: (dict) A dictionary which maps the key (aux id): to an item (string representing a set of layers in the output set)
        :param mode_ids: (dict) A dictionary which maps a key (string|aux_tok representing output layer name) to an item (int|id of that token)
        '''

        #TODO add relevant token ids for later and a description of each parameters...
        super(NMTransformer, self).__init__()
        self.num_layers_vanilla = num_layers_vanilla
        self.num_layers_nm = num_layers_nm
        self.num_layers_mc = num_layers_mc # number of metacognition layers.
        self.num_layers_output = num_layers_output

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.max_position_encoding = max_position_encoding

        self.max_seq_len_dec = max_seq_len_dec
        self.max_seq_len_nm = max_seq_len_dec + num_aux_toks
        self.num_aux_toks = num_aux_toks # includes the mandatory <cls> token at the beginning.

        assert max_position_encoding >= self.max_seq_len_nm, f"the max_position_encoding should be greater than the max sequence" \
                                                             f" length of the nm encoder input. ({max_position_encoding} > {self.max_seq_len_nm})"

        self.parallel_layers = parallel_layers
        if self.parallel_layers is None: # set to an empty list ([] or list()) if don't want to include this component.
            self.parallel_layers = ["aoint_rs",
                                    "highlighting_rs",
                                    "unknown_rs",
                                    "summarization_rs",
                                    "paraphrasing_rs"]

        self.mask_strategy = mask_strategy

        self.output_layers = dict()
        if output_layers is None:
            output_layers = ["lm",
                             "mqa",
                             "generate_answer",
                             "highlighting",
                             "summarization",
                             "paraphrasing"] # default to generate answer.
        self.fixed_output_layer = None # This is to manually be changed if it is to be used.

        self.aux_tok_output_layer_map = aux_tok_output_layer_map # maps an id to a string, representing the output layer.
        for key, item in self.aux_tok_output_layer_map.items():
            if item not in output_layers:
                raise Exception(f"Invalid item in aux_tok_output_layer_map, it should match an item in output_layers and doesn't!")
        self.mode_ids = mode_ids # dictionary containing ids for essential auxiliary tokens.

        # the third set of 'A' layers.
        for key in output_layers:
            self.output_layers[key] = OutputDecoder(num_layers_output, d_model, num_heads, dff,
                                                    max_seq_len=self.max_seq_len_nm, mask_strategy=mask_strategy,
                                                    rate=rate, name=key)

        # the first set fo layers.
        self.vanilla_set = Decoder(num_layers_vanilla, d_model, num_heads, dff, max_seq_len=max_seq_len_dec,
                              mask_strategy=mask_strategy, rate=rate)

        # the second set of layers.
        self.nm_set = NMDecoder(num_layers_nm, num_layers_mc, d_model, num_heads, dff, max_seq_len=self.max_seq_len_nm,
                 mask_strategy=mask_strategy, rate=rate, parallel_layers=self.parallel_layers)

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.dropout = tf.keras.layers.Dropout(rate=rate, input_shape=(d_model,))
        self.pos_encoding = positional_encoding(max_position_encoding, d_model)

        self.final_layer = tf.keras.layers.Dense(output_vocab_size, input_shape=(d_model,))
        self.vanilla_set_final_layer = tf.keras.layers.Dense(output_vocab_size, input_shape=(d_model,))

    def call(self, str_inp, id_inp, training, mask, reading_strat_mc_bool=False, vanilla_set_aux_loss_bool=False,
             fixed_output=False):
        '''
        Description: One run through the NMTransformer that generates an output based on the prediction.
        :param str_inp: (tf.Tenor{tf.dtypes.string}; [batch_size, max_seq_len_nm (nothing actually enforces this)])
            A tensor which represent the input to the model in string format.
        :param id_inp: (tf.Tensor; [batch_size, max_seq_len_nm (nothing actually enforces this)])
            A tensor which represents the input to the model in id (integer) format.
        :param training: (bool) A boolean value which represents whether or not we are in training mode.
        :param mask: (tf.Tensor; [batch_size, ..., ..., seq_len]) A tensor representing the mask to be used in multi-head attention.
        :param reading_strat_mc_bool: (bool) A boolean value that represents if we are to process in reading strategies mode or not.
        :param vanilla_set_aux_loss_bool: (bool) A boolean value representing if we are updating the vanilla set's weights
            independently of the architecture as a whole as an auxiliary loss.
        :return:
            vanilla_set_output: (tf.Tensor; [])
        '''

        if fixed_output: assert self.fixed_output_layer is not None
        if not fixed_output: assert self.fixed_output_layer is None

        assert id_inp.shape[1] == self.max_seq_len_nm, f"The sequence length should be equal to {self.max_seq_len_nm}, " \
                                                       f"got {id_inp.shape[1]}!"

        vanilla_set_output = None  # need to pass through final_layer and train - train both components, but at a much less rate.
        nm_decoder_mc = None  # should be a dictionary when its value is generated.
        # above is to be trained as an auxiliary loss manner.
        task_prediction = None  # the normal model prediction.
        gating_weights = None  # return the gating weights (after sigmoid has already been applied to it).
        attention_weights = {}

        if not reading_strat_mc_bool:
            task_prediction, attention_weights, vanilla_set_output, gating_weights = self.run_no_rs(id_inp, training, mask, fixed_output)
            if vanilla_set_aux_loss_bool:
                vanilla_set_output = tf.nn.softmax(self.vanilla_set_final_layer(vanilla_set_output), axis=-1)
                # (batch_size, max_seq_len_dec, target_vocab_size)
                # this only goes through the vanilla_decoder and final layer.
        else: pass

        return vanilla_set_output, nm_decoder_mc, task_prediction, gating_weights, attention_weights

    def _pos_encoding_helper(self, x, training, max_seq_len):
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.dtypes.float32))
        # so the position encoding doesn't have a drastic effect and take away information in the actual word embedding.
        x += self.pos_encoding[:, :max_seq_len, :]
        x /= tf.math.sqrt(tf.cast(self.d_model, tf.dtypes.float32))  # re-normalize the input embeddings.
        return self.dropout(x, training=training)

    def run_no_rs(self, id_inp, training, mask, fixed_output):
        mode_id = tf.stop_gradient(id_inp[0, 2])  # first batch item and the (third) item in the sequence. <cls> <dec> <lm> ...
        ## NOTE (important): that in a batch only one mode_id is supported.

        id_inp = self.embedding(id_inp)  # (batch_size, max_seq_len_nm, d_model)
        id_inp = self._pos_encoding_helper(id_inp, training, self.max_seq_len_nm)

        attention_weights = {}
        if mask.shape[-2] == 1:
            vanilla_mask = mask[:, :, :, -self.max_seq_len_dec:]
        else:
            vanilla_mask = mask[:, :, -self.max_seq_len_dec:, -self.max_seq_len_dec:]

        vanilla_output, attn_weights_vanilla = self.vanilla_set_run(id_inp[:, -self.max_seq_len_dec:, :], training,
                                                                    vanilla_mask)
        # vanilla_output.shape = (batch_size, max_seq_len_dec, d_model)

        # don't want the vanilla_output gradient flowing back through the nm_inp
        nm_inp = tf.concat([id_inp[:, :self.num_aux_toks, :], tf.stop_gradient(vanilla_output)], axis=1)
        # nm_inp.shape = (batch_size, max_seq_len_nm, d_model)

        # below has been removed as there is no reason to apply it twice.
        #nm_inp = self._pos_encoding_helper(nm_inp, training, self.max_seq_len_nm)

        nm_output, attn_weights_nm, mc_output_dict = self.neuromodulation_set_run(nm_inp, training, mask, mode="default")

        gating_weights = tf.sigmoid(nm_output[:, -self.max_seq_len_dec:, :])

        if not fixed_output:
            output_, attn_weights_output = self.output_set_run(gating_weights * vanilla_output,
                                                           training, vanilla_mask, mode_id)
        else:
            output_, attn_weights_output = self.fixed_output_set_run(gating_weights * vanilla_output,
                                                           training, vanilla_mask)

        attention_weights["vanilla_attention_weights"] = attn_weights_vanilla
        attention_weights["nm_attention_weights"] = attn_weights_nm
        attention_weights["mc_output_attention_weightse"] = mc_output_dict
        attention_weights["output_attention_weights"] = attn_weights_output

        output_ = tf.nn.softmax(self.final_layer(output_), axis=-1) # (batch_size, max_seq_len_dec, target_vocab_size)
        #task_prediction, attention_weights, vanilla_set_output, gating_weights
        return output_, attention_weights, vanilla_output, gating_weights

    def run_metacognition(self):
        # run up to the metacognition components and stop. no outputdecoder.
        pass

    def run_reading_strategy(self):
        # the process of running a specific reading strategy.
        pass

    # note id_inp should have the auxiliary tokens removed from both the mask and id_inp...
    def vanilla_set_run(self, id_inp, training, mask):
        return self.vanilla_set(id_inp, training=training, mask=mask)

    # includes auxiliary tokens.
    def neuromodulation_set_run(self, id_inp, training, mask, mode):
        return self.nm_set(id_inp, training, mask, mode=mode)

    # if aux tokens are to be removed, do them before being passed as input.
    def output_set_run(self, id_inp, training, mask, mode_id):
        # mode_id specifies the id. if it matches one in aux

        #key_ = self._get_layer_helper(mode_id)
        #print(f"\n\n{self.aux_tok_output_layer_map.keys()}\n\n")
        #print(f"\n\n{self.aux_tok_output_layer_map.keys()}\n\n")
        mode_id = mode_id.numpy().tolist()
        assert isinstance(mode_id, int)
        key_ = self.aux_tok_output_layer_map[mode_id]
        assert isinstance(key_, str)
        return self.output_layers[key_](id_inp, training, mask)

    def fixed_output_set_run(self, id_inp, training, mask):
        return self.fixed_output_layer(id_inp, training, mask)

    def _get_layer_helper(self, mode_id):
        val_ = None
        for key, value in self.aux_tok_output_layer_map.items():
            if key == mode_id: val_ = value
        #raise Exception(f"Invalid mode_id, it doesn't match a key specified in aux_tok_output_layer_map")
        return val_


if __name__ == "__main__":
    config = V4ConfigMediumSize(strategy=None, batch_size=2, loss_object=None, learning_rate=None)

    transformer = NMTransformer(num_layers_vanilla=config.num_layers_vanilla, num_layers_nm=config.num_layers_nm,
                                num_layers_mc=config.num_layers_mc, num_layers_output=config.num_layers_output,
                                d_model=config.d_model+12, num_heads=config.num_heads, dff=config.dff,
                                input_vocab_size=config.input_vocab_size, output_vocab_size=config.output_vocab_size,
                                max_position_encoding=config.max_position_encoding,
                                max_seq_len_dec=config.max_seq_len_dec, num_aux_toks=config.num_aux_toks,
                                mask_strategy=config.mask_strategy, rate=config.rate,
                                parallel_layers=config.parallel_layers, output_layers=config.output_layers,
                                aux_tok_output_layer_map=config.aux_tok_output_layer_map, mode_ids=config.mode_ids)

    #str_inp, id_inp, training, mask, reading_strat_mc_bool = False, vanilla_set_aux_loss_bool = False
    str_inp = None
    x1 = tf.random.uniform((config.batch_size, config.max_seq_len_dec), minval=0, maxval=24, dtype=tf.dtypes.int64)
    x2 = tf.constant([[0, 0, config.lm_tok_id],[0, 0, config.highlighting_rs_id]], dtype=tf.dtypes.int64)
    id_inp = tf.concat([x2, x1], axis=-1)
    print(f"id_inp.shape: {id_inp.shape}")

    training = True
    mask = create_padding_mask(id_inp, padding_id=0)
    #mask = create_combined_mask(id_inp, padding_id=0, num_aux_tok=3)
    reading_strat_mc_bool = False
    vanilla_set_aux_loss_bool = True

    vanilla_set_output, nm_decoder_mc, task_prediction, gating_weights, \
    attention_weights = transformer(str_inp, id_inp, training, mask, reading_strat_mc_bool, vanilla_set_aux_loss_bool)

    print(f"vanilla_set_output: {vanilla_set_output.shape}, \ntask_prediction: {task_prediction.shape}"
          f"\nnm_decoder_mc: {nm_decoder_mc}\ngating_weights.shape {gating_weights.shape}")