import copy

import tensorflow as tf
import numpy as np
import re

import sys
sys.path.append("..")

#import os
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#GPUS_AVAILABLE = 1

from transformers import TFGPT2Model, TFSharedEmbeddings
from models.config_model import *
from models.custom_lr_schedules import *
from models.AttentionMasks import *

class GPT2Class(tf.keras.Model):
    '''
    An implementation of version 4 of the neuromodulated transformer with reading strategy/metacognition support.
    '''

    def __init__(self, d_model, input_vocab_size, output_vocab_size, max_seq_len_dec=768, num_aux_toks=3,
                 gpt_pretrained_model="gpt2-medium", rate=0.1):
        '''
        :param input_vocab_size: (int) An integer specifying the number vocabulary size of the input language.
        :param output_vocab_size: (int) An integer specifying the number vocabulary size of the output language.
        :param max_position_encoding: (int) An integer specifying the maximum position encoding length for absolute fixed
            position embeddings.
        :param max_seq_len_dec: (int) An integer specifying the maximum sequence length of the input.
        :param num_aux_toks: (int) An integer specifying the number of auxiliary tokens -> used to get max_seq_len_nm.
        :param gpt_pretrained_model: (str) String representing the pre-trained model to load.
        '''

        self.d_model = d_model
        self.rate = rate

        super(GPT2Class, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        self.max_seq_len_dec = max_seq_len_dec
        self.max_seq_len_nm = max_seq_len_dec + num_aux_toks
        self.num_aux_toks = num_aux_toks # includes the mandatory <cls> token at the beginning.

        assert gpt_pretrained_model in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
        self.gpt_pretrained_model = gpt_pretrained_model

        self.vanilla_set = TFGPT2Model.from_pretrained(gpt_pretrained_model) # note: b/c the weights for the embeddings are changed we can't use the language model head.

        old_embeddings_weights = self.vanilla_set.transformer.wte.weights[0] # old vocab weights
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

        self.dropout = tf.keras.layers.Dropout(rate=rate, input_shape=(d_model,))

        # manually insert final head.
        self.final_layer = tf.keras.layers.Dense(output_vocab_size, input_shape=(d_model,), kernel_regularizer=tf.keras.regularizers.L2(0.01))

    def build_custom(self, cls_tok_id, dec_tok_id, lst_of_task_ids, minval=1, maxval=24):
        x = tf.cast(tf.random.uniform((1, self.max_seq_len_dec), minval=minval, maxval=maxval), dtype=tf.dtypes.int64)
        for id_ in lst_of_task_ids:
            y = tf.concat([tf.cast(tf.convert_to_tensor([[cls_tok_id, dec_tok_id, id_]]), dtype=tf.dtypes.int64),x], axis=-1)
            #print(f"id_: {id_}")
            output = self.run_no_rs(y, training=False, mask=None, gpt_pad_mask=None, fixed_output=False,
                                    stop_gradient=False)

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

        # NOTE: keep idential to NMTransformer.py class so the same training scripts can be used.

        assert id_inp.shape[1] == self.max_seq_len_nm, f"The sequence length should be equal to {self.max_seq_len_nm}, " \
                                                       f"got {id_inp.shape[1]}!"

        vanilla_set_output = None  # need to pass through final_layer and train - train both components, but at a much less rate.
        nm_decoder_mc = None  # should be a dictionary when its value is generated.
        # above is to be trained as an auxiliary loss manner.
        task_prediction = None  # the normal model prediction.
        gating_weights = None  # return the gating weights (after sigmoid has already been applied to it).
        attention_weights = {}

        task_prediction, attention_weights, vanilla_set_output, gating_weights = self.run_no_rs(id_inp, training,
                                                                                                mask, gpt_pad_mask,
                                                                                                fixed_output,
                                                                                                stop_gradient)


        return vanilla_set_output, nm_decoder_mc, task_prediction, gating_weights, attention_weights


    def run_no_rs(self, id_inp, training, mask, gpt_pad_mask, fixed_output, stop_gradient):

        attention_weights = {}
        attn_weights_vanilla=None
        vanilla_output = None
        if gpt_pad_mask is None:
            vanilla_output = self._vanilla_gpt_helper(id_inp[:, -self.max_seq_len_dec:], training, None)
        else:
            vanilla_output = self._vanilla_gpt_helper(id_inp[:,-self.max_seq_len_dec:], training,
                                                          gpt_pad_mask[:,-self.max_seq_len_dec:])


        attention_weights["vanilla_attention_weights"] = attn_weights_vanilla


        output_ = tf.nn.softmax(self.final_layer(vanilla_output), axis=-1) # (batch_size, max_seq_len_dec, target_vocab_size)
        #task_prediction, attention_weights, vanilla_set_output, gating_weights
        return output_, attention_weights, vanilla_output, None # None was the gating weights.

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

    config = V4ConfigMediumSize(strategy="MirroredStrategy", batch_size=2,
                                loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,
                                                                                          reduction='none'),
                                # learning_rate=tf.keras.optimizers.schedules.CosineDecay(0.0001, decay_steps=1000000),
                                # learning_rate=0.00001,
                                learning_rate=CosineDecayLW(start_lr=0.00005, lower_bound_lr=0.00001,upper_bound_lr=0.0001,
                                                            warmup_steps=2000, decay_steps=400000),
                                vocab_filepath="/data/kkno604/Neuromodulated-Transformer/vocabulary/vocab1.txt",
                                gpt2_117=True,
                                tokenizer="gpt2")

    transformer = GPT2Class(d_model=1024, input_vocab_size=config.input_vocab_size, output_vocab_size=config.output_vocab_size,
                            max_seq_len_dec=768, num_aux_toks=3, gpt_pretrained_model="gpt2-medium")

    #str_inp, id_inp, training, mask, reading_strat_mc_bool = False, vanilla_set_aux_loss_bool = False
    str_inp = None
    x1 = tf.random.uniform((config.batch_size, config.max_seq_len_dec), minval=0, maxval=24, dtype=tf.dtypes.int64)
    x2 = tf.constant([[0, 0, config.lm_tok_id],[0, 0, config.highlighting_rs_id]], dtype=tf.dtypes.int64)
    id_inp = tf.concat([x2, x1], axis=-1)
    print(f"id_inp.shape: {id_inp.shape}")

    training = True
    mask = create_padding_mask(id_inp, padding_id=0)
    gpt_pad_mask = create_padding_mask_gpt(id_inp, padding_id=0)
    #mask = create_combined_mask(id_inp, padding_id=0, num_aux_tok=3)
    reading_strat_mc_bool = False
    vanilla_set_aux_loss_bool = True

    vanilla_set_output, nm_decoder_mc, task_prediction, gating_weights, \
    attention_weights = transformer(str_inp=str_inp, id_inp=id_inp, training=training, mask=mask,
                                    gpt_pad_mask=gpt_pad_mask, reading_strat_mc_bool=False,
                                    vanilla_set_aux_loss_bool=False, fixed_output=False, stop_gradient=False)

    #str_inp, id_inp, training, mask, gpt_pad_mask = None, reading_strat_mc_bool = False, vanilla_set_aux_loss_bool = False,
    #fixed_output = False, stop_gradient = True

    print(f"GPT-2 last hidden state: {vanilla_set_output}, \ntask_prediction: {task_prediction}"
          f"\nnm_decoder_mc: {nm_decoder_mc}\ngating_weights.shape {gating_weights}")

    vanilla_set_output, nm_decoder_mc, task_prediction, gating_weights, attention_weights