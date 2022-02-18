import copy

import tensorflow as tf
import numpy as np
import re

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

from transformers import TFGPT2Model, TFSharedEmbeddings

class NMTransformer(tf.keras.Model):
    '''
    An implementation of version 4 of the neuromodulated transformer with reading strategy/metacognition support.
    '''

    def __init__(self, num_layers_vanilla, num_layers_nm, num_layers_mc, num_layers_output,
                 d_model, num_heads, dff, input_vocab_size, output_vocab_size, max_position_encoding=10000,
                 max_seq_len_dec=768, num_aux_toks=9, mask_strategy="default",
                 rate=0.1, parallel_layers=None, output_layers=None,
                 aux_tok_output_layer_map={}, mode_ids={}, gpt2_117=False):
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

        # the first set of layers.
        self.gpt2_bool = False
        self.vanilla_set = None
        if not gpt2_117:
            self.vanilla_set = Decoder(num_layers_vanilla, d_model, num_heads, dff, max_seq_len=max_seq_len_dec,
                                  mask_strategy=mask_strategy, rate=rate)
        else:
            self.gpt2_bool = True
            self.vanilla_set = TFGPT2Model.from_pretrained('gpt2') # this is small -- actually 124 million parameters...

            old_embeddings_weights = self.vanilla_set.transformer.wte.weights[0] # old vocab_size weights
            assert input_vocab_size >= old_embeddings_weights[0].shape[0]
            #print(f"\n\n\n\nold_embedding_weights_vocab_size: {old_embeddings_weights.shape[0]}\n\n\n\n")
            self.vanilla_set.transformer.wte = TFSharedEmbeddings(input_vocab_size, d_model,
                                                                  initializer_range=0.02, # hardcoded.
                                                                  name="wte") # we want our to add our own tokens to the embedding layer. This is done below.
            # manually build it here.
            self.vanilla_set.transformer.wte(1)
            #print(f"\n\n\n\nself.vanilla_set.transformer.wte: {self.vanilla_set.transformer.wte.weights[0].shape[0]}\n\n\n\n")
            # below updates the weights to our new vocabulary.
            self.vanilla_set.transformer.wte.set_weights([tf.concat([old_embeddings_weights,
                                                                     self.vanilla_set.transformer.wte.weights[0][old_embeddings_weights.shape[0]:, :]],
                                                                                                       axis=0)])

            # The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.
            # note: that the only mask to be applied here is a padding mask... where a 1 means we don't pad/mask.
            # This is the opposite to how we have done it where a 1 means we do mask/pad because it is multipled by a very large negative value.

        # the second set of layers.
        self.nm_set = NMDecoder(num_layers_nm, num_layers_mc, d_model, num_heads, dff, max_seq_len=self.max_seq_len_nm,
                 mask_strategy=mask_strategy, rate=rate, parallel_layers=self.parallel_layers)

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        # quickly build so below assertion works.
        self.embedding(1)

        if gpt2_117: assert self.embedding.weights[0].shape == self.vanilla_set.transformer.wte.weights[0].shape

        self.dropout = tf.keras.layers.Dropout(rate=rate, input_shape=(d_model,))
        self.pos_encoding = positional_encoding(max_position_encoding, d_model)

        self.final_layer = tf.keras.layers.Dense(output_vocab_size, input_shape=(d_model,), kernel_regularizer=tf.keras.regularizers.L2(0.01))
        self.vanilla_set_final_layer = tf.keras.layers.Dense(output_vocab_size, input_shape=(d_model,), kernel_regularizer=tf.keras.regularizers.L2(0.01))

    def set_output_layers_equal_to_lm(self, lm_key):
        keys = self.output_layers.keys()
        lm_output_set = self.output_layers[lm_key] # this is hardcoded.
        for key in keys:
            if key == lm_key: continue
            else:
                self.output_layers[key] = lm_output_set
                self.output_layers[key]._name = str(key)+" output_set"
                print(f"{self.output_layers[key]} name: {self.output_layers[key].name}")



    def build_custom(self, cls_tok_id, dec_tok_id, lst_of_task_ids, minval=1, maxval=24):
        x = tf.cast(tf.random.uniform((1, self.max_seq_len_dec), minval=minval, maxval=maxval), dtype=tf.dtypes.int64)
        for id_ in lst_of_task_ids:
            y = tf.concat([tf.cast(tf.convert_to_tensor([[cls_tok_id, dec_tok_id, id_]]), dtype=tf.dtypes.int64),x], axis=-1)
            #print(f"id_: {id_}")
            output = self.run_no_rs(y, training=False, mask=None, gpt_pad_mask=None, fixed_output=False,
                                    stop_gradient=False)
            #print(f"\nREACH\n")






    def call(self, str_inp, id_inp, training, mask, gpt_pad_mask=None, reading_strat_mc_bool=False, vanilla_set_aux_loss_bool=False,
             fixed_output=False, stop_gradient=True):
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
            task_prediction, attention_weights, vanilla_set_output, gating_weights = self.run_no_rs(id_inp, training,
                                                                                                    mask, gpt_pad_mask,
                                                                                                    fixed_output,
                                                                                                    stop_gradient)
            if vanilla_set_aux_loss_bool:
                vanilla_set_output = tf.nn.softmax(self.vanilla_set_final_layer(vanilla_set_output), axis=-1)
                # (batch_size, max_seq_len_dec, target_vocab_size)
                # this only goes through the vanilla_decoder and final layer.
            else:
                vanilla_set_output = None
        else: pass

        return vanilla_set_output, nm_decoder_mc, task_prediction, gating_weights, attention_weights

    def _pos_encoding_helper(self, x, training, max_seq_len):
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.dtypes.float32))
        # so the position encoding doesn't have a drastic effect and take away information in the actual word embedding.
        x += self.pos_encoding[:, :max_seq_len, :]
        x /= tf.math.sqrt(tf.cast(self.d_model, tf.dtypes.float32))  # re-normalize the input embeddings.
        return self.dropout(x, training=training)

    def run_no_rs(self, id_inp, training, mask, gpt_pad_mask, fixed_output, stop_gradient):
        id_orig = id_inp
        if id_inp.shape[0] != 0:
            mode_id = tf.stop_gradient(id_inp[0, 2])  # first batch item and the (third) item in the sequence. <cls> <dec> <lm>
        else: mode_id = tf.zeros(1) # so no error when batch size is zero in multi-gpu training.
        ## NOTE (important): that in a batch only one mode_id is supported.

        # note: if using gpt2 then embedding doesn't utilize non auxiliary tokens.
        id_inp = self.embedding(id_inp)  # (batch_size, max_seq_len_nm, d_model)
        # don't need positional information for aux toks, while this is handled by the gpt2 model.
        if not self.gpt2_bool: id_inp = self._pos_encoding_helper(id_inp, training, self.max_seq_len_nm)

        attention_weights = {}
        vanilla_mask = None
        if mask is not None:
            if mask.shape[-2] == 1:
                vanilla_mask = mask[:, :, :, -self.max_seq_len_dec:]
            else:
                vanilla_mask = mask[:, :, -self.max_seq_len_dec:, -self.max_seq_len_dec:]

        if not self.gpt2_bool:
            vanilla_output, attn_weights_vanilla = self.vanilla_set_run(id_inp[:, -self.max_seq_len_dec:, :], training,
                                                                        vanilla_mask)
        else: # gpt2=True
            attn_weights_vanilla=None
            vanilla_output = None
            if stop_gradient: # stop_gradient doubles up here if we are using gpt2, no need to define another variable as it makes no difference.
                if gpt_pad_mask is None:
                    vanilla_output = tf.stop_gradient(
                        self._vanilla_gpt_helper(id_orig[:, -self.max_seq_len_dec:], training, None))
                else:
                    vanilla_output = tf.stop_gradient(self._vanilla_gpt_helper(id_orig[:,-self.max_seq_len_dec:], training,
                                                                           gpt_pad_mask[:,-self.max_seq_len_dec:]))
            else:
                if gpt_pad_mask is None:
                    vanilla_output = self._vanilla_gpt_helper(id_orig[:, -self.max_seq_len_dec:], training, None)
                else:
                    vanilla_output = self._vanilla_gpt_helper(id_orig[:,-self.max_seq_len_dec:], training,
                                                              gpt_pad_mask[:,-self.max_seq_len_dec:])
        # vanilla_output.shape = (batch_size, max_seq_len_dec, d_model)

        if stop_gradient: #== True
            # note: if stop gradient with gpt2 model, then stopping gradient here does nothing as it has already been stopped.
            nm_inp = tf.concat([id_inp[:, :self.num_aux_toks, :], tf.stop_gradient(vanilla_output)], axis=1)
        else: # == False
            nm_inp = tf.concat([id_inp[:, :self.num_aux_toks, :], vanilla_output], axis=1)
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

    def _vanilla_gpt_helper(self, input_ids, training, mask):
        return self.vanilla_set(input_ids=input_ids, attention_mask=mask, training=training)['last_hidden_state']

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

    def generate_answer(self, str_inp, id_inp, training, mask, gpt_pad_mask=None, reading_strat_mc_bool=False,
                        vanilla_set_aux_loss_bool=False, fixed_output=False, stop_gradient=False,
                        pad_tok_id=None, end_tok_id=None, gen_len_max=50):

        # TODO add support for reading strategies and metacognition.

        if training: print(f"\ntraining is True meaning that dropout layers are turned on, is this what you want?\n")

        batch_size_ = id_inp.shape[0]

        generation_done = set()
        # id_inp includes the auxliary token.
        generated_ids = [[] for _ in range(id_inp.shape[0])] # list that contains generated ids for each batch.

        #num_aux_toks = abs(nm_inp.shape[1] - dec_inp.shape[1])
        #self.num_aux_toks

        outer_id_inp = []
        #outer_nm_inp = []
        # print(f"outer_dec_inp: {outer_dec_inp}")
        for i in range(gen_len_max):
            # print(f"outer_dec_inp: {outer_dec_inp}")
            if i > 0:  # i.e. not the first input which is correct.
                assert len(outer_id_inp) > 0, f"Error: outer_dec_inp is empty!"
                id_inp = tf.cast(tf.convert_to_tensor(np.asarray(outer_id_inp)), dtype=tf.dtypes.int64)
                assert id_inp.shape[1] == self.max_seq_len_nm, f"The id_inp.shape[1] should be equal to the nm_max_sl!"
                assert id_inp.shape[0] == batch_size_, f"new input batch size {id_inp.shape[0]} doesn't match the " \
                                                       f"previous batch size {batch_size_}" \
                                                       f"\n {print(id_inp.shape)}"

                # create new padding masks for new id_inp with new generated tokens.
                gpt_pad_mask = create_padding_mask_gpt(id_inp, padding_id=pad_tok_id)
                mask = create_combined_mask(id_inp, pad_tok_id, self.num_aux_toks)  # [batch_size, 1, seq_len, seq_len]

            new_id_inp = []

            #id_inp, training, mask, gpt_pad_mask, fixed_output, stop_gradient
            task_prediction, _, _, _ = self.run_no_rs(id_inp=id_inp, training=training, mask=mask,
                                                      gpt_pad_mask=gpt_pad_mask, fixed_output=fixed_output,
                                                      stop_gradient=stop_gradient)
            # task_prediction.shape = (batch_size, max_seq_len_dec, target_vocab_size)
            #print(f"task_prediction: {task_prediction.numpy().tolist()}")

            for b in range(batch_size_):

                if b in generation_done:
                    new_id_inp.append(id_inp[b, :].numpy().tolist())  # no changes here.
                    continue  # generation is done for this batch item.

                ## get new prediction and add to first pad index... stop when end_tok_id is reached or gen_len_max is reached. (add another outside loop)
                id_prediction, pred_index, end, first_pad_element = self._get_pred_id_helper(
                    tf.expand_dims(task_prediction[b, :, :], axis=0),
                    tf.expand_dims(id_inp[b, self.num_aux_toks:], axis=0),
                    pad_tok_id)

                # Note: that the end token is not appended to the generated tokens list.
                if id_prediction == end_tok_id:
                    # print(f"b: {b} is finished, shouldn't see it processed any more!")
                    new_id_inp.append(id_inp[b, :].numpy().tolist())  # no changes made here.
                    #new_nm_inp.append(nm_inp[b, :].numpy().tolist())  # no changes made here.
                    generation_done.add(b)
                    continue

                # add new prediction to id_inp, if no pad token at the max seq length, then remove 1st item to make room...
                id_inp_np = id_inp[b, :].numpy().tolist()  # 1D list (of integers).
                new_input = None
                if end:  # move all to the left once and append the prediction to the end.
                    new_input = id_inp_np[:self.num_aux_toks] + id_inp_np[self.num_aux_toks+1:] + [id_prediction]
                else:
                    if pred_index is not None:
                        # +1 becuase pred_index is the index we want to include up to, however, rhs of : is not inclusive
                        # so add a 1 here to include this index, which represents the last non pad_id token...
                        # the lhs of : is inclusive so +2 is correct.
                        new_input = id_inp_np[:self.num_aux_toks+pred_index+1] + [id_prediction] + \
                                    id_inp_np[self.num_aux_toks+pred_index+2:]
                    # else: # is handled below with first_pad_element.

                if first_pad_element:  # also means pred_index will be None. handles one case above.
                    assert pred_index is None, f"pred_index should be None, got {pred_index}!"
                    new_input = id_inp_np  # reset to all pad tokens, nothing happens.
                    generation_done.add(b)  # all pad tokens, thus just skip. This is a set so no duplicates.

                assert new_input is not None, f"new_input should not be None, something went wrong in the code logic!"

                new_id_inp.append(new_input)
                if id_prediction != end_tok_id: # don't want the end tokens in the prediction.
                    generated_ids[b].append(id_prediction)

            outer_id_inp = new_id_inp

            if len(generation_done) >= batch_size_: break  # we are done here.
        return generated_ids

    def _get_pred_id_helper(self, prediction, id_inp, pad_tok_id):

        assert prediction.shape[0] == 1, f"The first dimension, the batch size should be equal to 1 for the " \
                                         f"prediction vector, got {prediction.shape[0]}!"
        assert id_inp.shape[0] == 1, f"The first dimension, the batch size should be equal to 1 for the model " \
                                     f"input, got {id_inp.shape[0]}!"

        pred_index = None
        for i in range(1, id_inp.shape[1]): # iterate across the sequence length.
            if id_inp[0, i] == pad_tok_id:
                pred_index = i-1  # we get the prediction at the previous token.
                break
        first_pad_element = False
        if id_inp[0, 0] == pad_tok_id: first_pad_element = True

        end = None
        if pred_index is not None:
            #print(f"prediction vocabulary: {prediction[0, pred_index, -100:].numpy().tolist()}")
            id_prediction = tf.argmax(prediction[0, pred_index, :])
            end = False
        else:  # is None.
            id_prediction = tf.argmax(prediction[0, -1, :])  # get last seq_len item to get next prediction.
            end = True

        id_prediction = id_prediction.numpy().tolist()
        if isinstance(id_prediction, int):
            pass
        elif isinstance(id_prediction, list):
            id_prediction = id_prediction[0]
        else:
            raise Exception(f"id_prediction is not of type int or list, got {type(id_prediction)}!")

        return id_prediction, pred_index, end, first_pad_element

    def generate_natural_language_top_k(self, string_input, tokenizer, pad_to_length, padding_id, num_aux_tokens, num_to_generate, k=10):
        '''
        Function: generate_natural_language. \n
        Description: Function that generates output text given a string as input using top-k sampling. \n
            Parts taken from https://medium.com/deep-learning-with-keras/sampling-in-text-generation-b2f4825e1dad
        Input:
            string_input: (str) Input to be used as context to generate text. \n
            pad_to_length: (int) The number of tokens to pad to (i.e. to max_seq_len). \n
            num_aux_tokens: (int) The number of aux_tokens at the beginning of the neuromodulation encoder's input. \n
            num_to_generate: (int) The number of words to generate on top of the input.
        Return:
            (string)
        '''

        def softmax(x, theta=2.0):
            ps = np.exp(x * theta)
            ps /= np.sum(ps)
            return ps

        original = string_input
        new = []

        enc_str = tokenizer.encode_single(string_input)
        for i in range(num_to_generate):
            pad_enc_str = enc_str[-pad_to_length:] + [padding_id for _ in range(pad_to_length-len(enc_str[-pad_to_length:]))]
            encoded_inp = tf.cast(tf.convert_to_tensor(np.asarray(pad_enc_str)),
                                  dtype=tf.dtypes.int64)  # shape: (inp_seq_len)
            encoded_inp = tf.expand_dims(encoded_inp, axis=0)  # shape: (batch_size (1), inp_seq_len)

            gpt_pad_mask = create_padding_mask_gpt(encoded_inp, padding_id=padding_id)  # look ahead padding is handled by huggingface gpt2 model.
            mask = create_combined_mask(encoded_inp, padding_id=padding_id, num_aux_tok=num_aux_tokens)

            #nm_mask = create_combined_mask(encoded_inp, padding_id, num_aux_tokens)
            #dec_mask = create_combined_mask(encoded_inp, padding_id)

            output, attention_weights, vanilla_output, gating_weights = self.run_no_rs(encoded_inp, False, mask,
                                                                                       gpt_pad_mask, False, False)

            #output, _, _ = self.model(encoded_inp, encoded_inp, training=False, padding_id=padding_id,
            #                               num_aux_tok=num_aux_tokens, nm_mask=nm_mask, dec_mask=dec_mask)
            #predictions = tf.nn.log_softmax(output, axis=-1) # shape: (batch_size (1), max_seq_len, tar_vocab_size)
            #predictions = output
            predictions = tf.squeeze(output) # shape: (max_seq_len, tar_vocab_size)

            # -pad_to_length removes tokens if there is more generated than the maximum allowed sequence length. -1 and length of it gets the last index.
            index = None
            for i, id in enumerate(enc_str):
                if id == padding_id: break
                index = i
            assert index is not None
            #print(index)
            predictions = tf.squeeze(predictions[index-num_aux_tokens,:]) # -num_aux_toks because index inludes aux tokens where the generates output doesn't.
            # shape: (target_vocab_size)
            #pred = tf.math.argmax(predictions) # note that for the tokenizer the first index id starts at 0 for my case.
            top_k_prob, top_k_indices = tf.math.top_k(predictions, k=k, sorted=True)
            top_k_indices = np.asarray(top_k_indices).astype("int32")

            top_k_redist_prob = softmax(top_k_prob)
            top_k_redist_prob = np.asarray(top_k_redist_prob).astype("float32")

            sampled_tok_index = np.random.choice(top_k_indices, p=top_k_redist_prob)
            #print(f"The sampled token (top_k) is {sampled_tok_index}\n"
            #      f" tokenizer.decode(sampled_tok_index) {tokenizer.decode([sampled_tok_index])}\n"
            #      f"enc_str: {enc_str}")
            enc_str.append(sampled_tok_index)
            #print("tok", tok)
            new.append(sampled_tok_index)

        return tokenizer.decode(enc_str), original, tokenizer.decode(new)


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