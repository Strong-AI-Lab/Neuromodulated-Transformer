'''
File name: fine_tuning_class.py
Author: Kobe Knowles
Date created: 04/02/22
Data last modified: 04/02/22
'''

import tensorflow as tf
import numpy as np
import math
import time
import re
import rouge
import collections
import string

'''
rouge_l_evaluator = rouge.Rouge(
    metrics=["rouge-l"],
    max_n=4,
    limit_length=True,
    length_limit=100,
    length_limit_type="words",
    apply_avg=True,
    apply_best=True,
    alpha=0.5,
    weight_factor=1.2,
    stemming=True,
)
'''
rouge_l_evaluator = rouge.Rouge(
    metrics=["rouge-l"],
)

def rouge_l(p, g): # p:rediction g:round truth
    return rouge_l_evaluator.get_scores(p, g)

import sys
sys.path.append("..")

from models.AttentionMasks import *

class FineTuningClass:
    '''
    Description: Class that implements the training and evaluation code for fine-tuning the NMT.
    Input:
        (note: tensorflow models/losses and optimizers are the only that are supported,
            however the only tokenizers that are supported are from huggingface wrapped up in custom made tokenizer class)
        model: The model to train on or get test results..
        optimizer: (tf.keras.optimzers) A tensorflow/keras optimizer.
        loss_object: e.g. SparseCategoricalCrossentropy
        loss_function: A function that calculated the loss for each item in the sequence, except for <pad> tokens.
        tokenizer: A huggingface tokenizer wrapped in a higher level class.
        checkpoint_path_recent: (string) path to the recent checkpoint folder.
        strategy: (tf.distribute...) Strategy for splitting the training across multiple gpus; otherwise None.
        pad_token: (str) A token in string format that represents the pad token. <pad> is the default.
        end_tok: (str) A token in string format that represents the end token. </s> is the default.
        recent_to_keep: (int) The number of (recent) checkpoints to keep.
        load_recent: (bool) True if we are to load the most recent checkpoints; False otherwise.
        load_specific_path: (str) A filepath to a checkpoint for the model to load. Overrides the recent checkpoints.
            If it is ""---an empty string---then it acts as a no op.
        enc_tok_id: (None|str) The encoder auxiliary token in string format.
        dec_tok_id: (None|str) The decoder auxiliary token in string format.
            Note: enc_tok_id and dec_tok_id are for future additions to this class. It is required that an integer is
                passed as input, otherwise an error is raised.
        output_layer_name: (None|str) A string representing a output layer name in the NMT; it sets a specific
            output layer for training, allowing for tf.function to be used. None means that tf.function will not
             be used.
        fixed_output: (bool) True if the output layer is fixed; False otherwise.
        stop_gradient: (bool) True if gradient is to be stopped (stop all together if gpt-2 is used as the vanilla set,
            otherwise it stops gradient flowing from the neuromodulatory set to the vanilla set); False if gradient
            is not to be stopped.
        reading_strat_mc_bool: (bool) True if we are to perform the reading strategies; False otherwise.
        vanilla_set_aux_loss_bool: (bool) True if an auxiliary loss for the vanilla set is to be used (what task depends
            on the input; however, normally it is the task of the whole NMT); False if no aux loss for the
            vanilla set.
        lambda_vanilla_set: (float) Floating point number between 0 and 1, dictating how much to scale the auxiliary
            loss for the vanilla set.
        lm_aux_loss_global: (bool) True if a language model auxiliary loss is to be used (while fine-tuning);
            False otherwise.
        lambda_lm: (float) Floating point number between 0 and 1, dictating how much to scale the auxiliary loss for
            the language modelling auxiliary loss (done to the output of the model as a whole).
        train_cutoff: (int) An integer representing how many epochs until the dropout layers are turned on.
            (Note: if tf.function is utilised, it won't switch over, need to restart the script again)
            To disable this -> to have no effect, then set to 0 (i.e., it does nothing).

    '''
    def __init__(self, model, optimizer, loss_object, loss_function, tokenizer,
                 checkpoint_path_recent, strategy, pad_token="<pad>", end_tok="</s>",
                 recent_to_keep=5, load_recent=False, load_specific_path='',
                 enc_tok=None, dec_tok=None, output_layer_name="lm",
                 fixed_output=True, stop_gradient=False, reading_strat_mc_bool=False,
                 vanilla_set_aux_loss_bool=False, lambda_vanilla_set=0.5, lambda_lm=0.1,
                 lm_aux_loss_global=False, train_cutoff=0):

        self.model = model
        self.optimizer = optimizer
        self.loss_object = loss_object
        self.loss_function = loss_function
        self.tokenizer = tokenizer

        self.checkpoint_path_recent = checkpoint_path_recent
        self.strategy = strategy

        self.fixed_output = fixed_output
        self.stop_gradient = stop_gradient
        self.reading_strat_mc_bool = reading_strat_mc_bool

        self.vanilla_set_aux_loss_bool = vanilla_set_aux_loss_bool
        self.lambda_vanilla_set = lambda_vanilla_set
        self.lambda_lm = lambda_lm
        self.lm_aux_loss_global = lm_aux_loss_global

        self.train_epoch = 0
        self.train_cutoff = train_cutoff

        self.padding_id = self.tokenizer.encode_single(pad_token)
        if len(self.padding_id) == 1:
            self.padding_id = self.padding_id[0]
        else:
            raise Exception("The padding token should only have one id! (it hasn't been added to the vocabulary)")

        self.padding_id = self.tokenizer.encode_single(pad_token)
        if len(self.padding_id) == 1:
            self.padding_id = self.padding_id[0]
        else:
            raise Exception("The padding token should only have one id! (it hasn't been added to the vocabulary)")

        self.end_tok_id = self.tokenizer.encode_single(end_tok)
        if len(self.end_tok_id) == 1:
            self.end_tok_id = self.end_tok_id[0]
        else:
            raise Exception("The end token should only have one id! (it hasn't been added to the vocabulary)")

        # the recent checkpoints are required.
        self.create_recent_checkpoint(recent_to_keep)
        if load_recent: self._load_recent_checkpoint()
        if load_specific_path != "":
            if load_recent: raise Exception(f"if load_recent is set to True, then load_specific path should be the "
                                            f"empty string \"\"")
            self._restore_specific_checkpoint(load_specific_path)

        assert enc_tok is not None and dec_tok is not None, f"It is required that the encoder and decoder token " \
                                                                  f"in string format is passed as input during " \
                                                                  f"initialization!"

        self.enc_tok_id = self.tokenizer.encode_single(enc_tok)
        if len(self.enc_tok_id) == 1:
            self.enc_tok_id = self.enc_tok_id[0]
        else:
            raise Exception("The enc token should only have one id! (it hasn't been added to the vocabulary)")

        self.dec_tok_id = self.tokenizer.encode_single(dec_tok)
        if len(self.dec_tok_id) == 1:
            self.dec_tok_id = self.dec_tok_id[0]
        else:
            raise Exception("The enc token should only have one id! (it hasn't been added to the vocabulary)")

        if output_layer_name is not None:
            self.model.fixed_output_layer = self.model.output_layers[output_layer_name]
            assert fixed_output, f"If we are setting a specific output layer before hand, " \
                                 f"then fixed_output should be True"
        else: assert not fixed_output, f"if output_layer_name is None, then fixed_output should not be."

    def create_recent_checkpoint(self, keep):
        self.ckpt = tf.train.Checkpoint(model=self.model,
                                        optimizer=self.optimizer)
        #self.ckpt = tf.train.Checkpoint(model=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path_recent, max_to_keep=keep)  # maybe remove max_to_keep?

    def _load_recent_checkpoint(self):
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("Latest checkpoint restored.")

    def _restore_specific_checkpoint(self, filepath):
        self.ckpt.restore(filepath)

    def train_step_MQA(self, input_string, input_id, label_id, aux_label,
                       aoint_indices, sample_weights, num_aux_tokens):
        #TODO add support for aoint_indices
        # look ahead padding is handled by huggingface gpt2 model.
        gpt_pad_mask = create_padding_mask_gpt(input_id,
                                               padding_id=self.padding_id)  # the model will handel whether or not to use this mask.
        mask = create_combined_mask(input_id, padding_id=self.padding_id,
                                    num_aux_tok=num_aux_tokens)  # no look ahead padding for the num_aux_tokens auxiliary tokens.

        loss, size = 0, 0
        # if train_cutoff is set to 0, then it is always True as train_epoch is 0 at the lowest. 0<0 is False 1<0 False ...
        # note that with tf.function it does not switch over after cutoff, so need to keep this in mind.
        train_ = False if self.train_epoch < self.train_cutoff else True
        with tf.GradientTape() as tape:
            vanilla_set_output, _, task_prediction, _, _ = self.model(input_string, input_id, training=train_,
                                                                      mask=mask,
                                                                      gpt_pad_mask=gpt_pad_mask,
                                                                      reading_strat_mc_bool=self.reading_strat_mc_bool,
                                                                      vanilla_set_aux_loss_bool=self.vanilla_set_aux_loss_bool,
                                                                      fixed_output=self.fixed_output,
                                                                      stop_gradient=self.stop_gradient)
            # ret (vanilla_set_output, nm_decoder_mc, task_prediction, gating_weights, attention_weights)
            # vanilla_set_output is after it has been passed through dense layer if vanilla_set_aux_loss_bool is True

            loss, size = self.loss_function(label_id, task_prediction, self.loss_object, self.padding_id)
            loss_ = loss / size

            if self.vanilla_set_aux_loss_bool:
                loss_aux, size_aux = self.loss_function(label_id, vanilla_set_output, self.loss_object, self.padding_id)
                loss_aux_ = loss_aux / size_aux
                loss_ = loss_ + (loss_aux_ * self.lambda_vanilla_set)

            if self.lm_aux_loss_global:
                loss_aux, size_aux = self.loss_function(aux_label, task_prediction, self.loss_object, self.padding_id)
                loss_aux_2 = loss_aux / size_aux
                loss_ = loss_ + (loss_aux_2 * self.lambda_lm)

        gradients = tape.gradient(loss_, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        loss = tf.convert_to_tensor([loss], dtype=tf.dtypes.float32)
        size = tf.convert_to_tensor([size], dtype=tf.dtypes.float32)

        return loss, size

    @tf.function
    def _distributed_train_step_tf_function_MQA(self, input_string, input_id, label_id, aux_label, aoint_indices,
                                                sample_weights, num_aux_tokens):
        loss_dec, size_dec = None, None
        if self.strategy is not None:
            loss_dec, size_dec = self.strategy.run(self.train_step_MQA, args=(
                input_string, input_id, label_id, aux_label, aoint_indices, sample_weights, num_aux_tokens,))

            if self.strategy.num_replicas_in_sync > 1:
                loss_dec = tf.reduce_sum(loss_dec.values)
                size_dec = tf.reduce_sum(size_dec.values)
            else:
                loss_dec = tf.reduce_sum(loss_dec)
                size_dec = tf.reduce_sum(size_dec)
        else:
            loss_dec, size_dec = self.train_step_MQA(input_string, input_id, label_id, aux_label, sample_weights,
                                                     aoint_indices, num_aux_tokens)

        return loss_dec, size_dec

    def _distributed_train_step_no_tf_function_MQA(self, input_string, input_id, label_id, aux_label, aoint_indices,
                                                   sample_weights, num_aux_tokens):
        loss_dec, size_dec = None, None
        if self.strategy is not None:
            loss_dec, size_dec = self.strategy.run(self.train_step_MQA, args=(
                input_string, input_id, label_id, aux_label, aoint_indices, sample_weights, num_aux_tokens,))

            if self.strategy.num_replicas_in_sync > 1:
                loss_dec = tf.reduce_sum(loss_dec.values)
                size_dec = tf.reduce_sum(size_dec.values)
            else:
                loss_dec = tf.reduce_sum(loss_dec)
                size_dec = tf.reduce_sum(size_dec)
        else:
            loss_dec, size_dec = self.train_step_MQA(input_string, input_id, label_id, aux_label, sample_weights,
                                                     aoint_indices, num_aux_tokens)

        return loss_dec, size_dec

    def train_batch_MQA(self, epoch_start, epoch_end, save_filepath_train, save_filepath_val, data_dict,
                        num_aux_tokens, save_end_epoch=True, print_every_iterations=100):

        if not save_end_epoch: print(f"save_end_epoch is False, the model's parameters will never be saved!")

        for e in range(epoch_start, epoch_end):

            start = time.time()
            self.train_epoch = e # for self.train_cutoff.

            batch = 0
            epoch_loss_total = 0
            epoch_size_total = 0

            for (input_string, input_id, label_id, aux_label, aoint_indices, sample_weights) in data_dict["train"]:

                batch += 1

                loss_dec, size_dec = None, None
                if self.fixed_output:
                    loss_dec, size_dec = self._distributed_train_step_tf_function_MQA(input_string, input_id, label_id,
                                                          aux_label, aoint_indices, sample_weights, num_aux_tokens)
                else: # if no fixed output then we don't want to run with @tf.function.
                    loss_dec, size_dec = self._distributed_train_step_no_tf_function_MQA(input_string, input_id,
                                                      label_id, aux_label,aoint_indices, sample_weights, num_aux_tokens)

                if size_dec == 0:
                    print(f"The size is zero for the currenet batch, skipping to next batch!")
                    continue

                epoch_loss_total += loss_dec
                epoch_size_total += size_dec

                loss_ = loss_dec / size_dec
                loss_ = tf.cast(loss_, dtype=tf.dtypes.float32)

                if iteration_counter % print_every_iterations == 0:
                    print(f'Batch: {batch} Epoch: {e + 1} Loss: {loss_:.4f}')

                header = True if (batch == 1 and e == 0) else False
                self._save_batch_loss_only("train", e+1, batch, save_filepath_train, header, loss_)

            total_loss = epoch_loss_total / epoch_size_total

            header_ = True if e == 0 else False
            self._save_epoch_loss_only("train", e+1, save_filepath_train, header_, total_loss)

            print(f'Epoch: {e+1} Loss: {total_loss:.4f}')
            print(f'Time taken for epoch {e+1}: {time.time() - start:.2f} secs\n')

            if save_end_epoch:
                ckpt_save_path = self.ckpt_manager.save()
                print(f'Saving checkpoint for epoch {e+1} at {ckpt_save_path}')

            if "val" in data_dict.keys():
                print(f"Running through the validation set now!")
                self.run_validation_MQA(e, save_filepath_val, data_dict["val"], num_aux_tokens)  # note e+1 is not correct.

    def train_step_GQA(self, input_string, input_id, label_id, aux_label, sample_weights, num_aux_tokens):

        # look ahead padding is handled by huggingface gpt2 model.
        gpt_pad_mask = create_padding_mask_gpt(input_id, padding_id=self.padding_id) # the model will handel whether or not to use this mask.
        mask = create_combined_mask(input_id, padding_id=self.padding_id, num_aux_tok=num_aux_tokens) # no look ahead padding for the num_aux_tokens auxiliary tokens.

        loss, size = 0, 0
        # if train_cutoff is set to 0, then it is always True as train_epoch is 0 at the lowest. 0<0 is False 1<0 False ...
        # note that with tf.function it does not switch over after cutoff, so need to keep this in mind.
        train_ = False if self.train_epoch < self.train_cutoff else True
        with tf.GradientTape() as tape:
            vanilla_set_output, _, task_prediction, _, _ = self.model(input_string, input_id, training=train_,
                                                                      mask=mask,
                                                                      gpt_pad_mask=gpt_pad_mask,
                                                                      reading_strat_mc_bool=self.reading_strat_mc_bool,
                                                                      vanilla_set_aux_loss_bool=self.vanilla_set_aux_loss_bool,
                                                                      fixed_output=self.fixed_output,
                                                                      stop_gradient=self.stop_gradient)
            # ret (vanilla_set_output, nm_decoder_mc, task_prediction, gating_weights, attention_weights)
            # vanilla_set_output is after it has been passed through dense layer if vanilla_set_aux_loss_bool is True

            loss, size = self.loss_function(label_id, task_prediction, self.loss_object, self.padding_id)
            loss_ = loss / size

            if self.vanilla_set_aux_loss_bool:
                loss_aux, size_aux = self.loss_function(label_id, vanilla_set_output, self.loss_object, self.padding_id)
                loss_aux_ = loss_aux / size_aux
                loss_ = loss_ + (loss_aux_ * self.lambda_vanilla_set)

            if self.lm_aux_loss_global:
                loss_aux, size_aux = self.loss_function(aux_label, task_prediction, self.loss_object, self.padding_id)
                loss_aux_2 = loss_aux / size_aux
                loss_ = loss_ + (loss_aux_2 * self.lambda_lm)

        gradients = tape.gradient(loss_, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        loss = tf.convert_to_tensor([loss], dtype=tf.dtypes.float32)
        size = tf.convert_to_tensor([size], dtype=tf.dtypes.float32)

        return loss, size

    @tf.function
    def _distributed_train_step_tf_function_GQA(self, input_string, input_id, label_id, aux_label, sample_weights,
                                                num_aux_tokens):
        loss_dec, size_dec = None, None
        if self.strategy is not None:
            loss_dec, size_dec = self.strategy.run(self.train_step_GQA, args=(
            input_string, input_id, label_id, aux_label, sample_weights, num_aux_tokens,))

            if self.strategy.num_replicas_in_sync > 1:
                loss_dec = tf.reduce_sum(loss_dec.values)
                size_dec = tf.reduce_sum(size_dec.values)
            else:
                loss_dec = tf.reduce_sum(loss_dec)
                size_dec = tf.reduce_sum(size_dec)
        else:
            loss_dec, size_dec = self.train_step_GQA(input_string, input_id, label_id, aux_label, sample_weights,
                                                 num_aux_tokens)

        return loss_dec, size_dec

    def _distributed_train_step_no_tf_function_GQA(self, input_string, input_id, label_id, aux_label, sample_weights,
                                                   num_aux_tokens):
        loss_dec, size_dec = None, None
        if self.strategy is not None:
            loss_dec, size_dec = self.strategy.run(self.train_step_GQA, args=(
                input_string, input_id, label_id, aux_label, sample_weights, num_aux_tokens,))

            if self.strategy.num_replicas_in_sync > 1:
                loss_dec = tf.reduce_sum(loss_dec.values)
                size_dec = tf.reduce_sum(size_dec.values)
            else:
                loss_dec = tf.reduce_sum(loss_dec)
                size_dec = tf.reduce_sum(size_dec)
        else:
            loss_dec, size_dec = self.train_step_GQA(input_string, input_id, label_id, aux_label, sample_weights,
                                                     num_aux_tokens)

        return loss_dec, size_dec

    def train_batch_GQA(self, epoch_start, epoch_end, save_filepath_train, save_filepath_val, data_dict,
                            num_aux_tokens, save_end_epoch=True, print_every_iterations=100):

            if not save_end_epoch: print(f"save_end_epoch is False, the model's parameters will never be saved!")

            for e in range(epoch_start, epoch_end):

                start = time.time()
                self.train_epoch = e # for self.train_cutoff.

                batch = 0
                epoch_loss_total = 0
                epoch_size_total = 0

                for (input_string, input_id, label_id, aux_label, sample_weights) in data_dict["train"]:

                    batch += 1

                    loss_dec, size_dec = None, None
                    if self.fixed_output:
                        loss_dec, size_dec = self._distributed_train_step_tf_function_GQA(input_string, input_id, label_id,
                                                              aux_label, sample_weights, num_aux_tokens)
                    else: # if no fixed output then we don't want to run with @tf.function.
                        loss_dec, size_dec = self._distributed_train_step_no_tf_function_GQA(input_string, input_id,
                                                          label_id, aux_label, sample_weights, num_aux_tokens)

                    if size_dec == 0:
                        print(f"The size is zero for the currenet batch, skipping to next batch!")
                        continue

                    epoch_loss_total += loss_dec
                    epoch_size_total += size_dec

                    loss_ = loss_dec / size_dec
                    loss_ = tf.cast(loss_, dtype=tf.dtypes.float32)

                    if iteration_counter % print_every_iterations == 0:
                        print(f'Batch: {batch} Epoch: {e + 1} Loss: {loss_:.4f}')

                    header = True if (batch == 1 and e == 0) else False
                    self._save_batch_loss_only("train", e+1, batch, save_filepath_train, header, loss_)

                total_loss = epoch_loss_total / epoch_size_total

                header_ = True if e == 0 else False
                self._save_epoch_loss_only("train", e+1, save_filepath_train, header_, total_loss)

                print(f'Epoch: {e+1} Loss: {total_loss:.4f}')
                print(f'Time taken for epoch {e+1}: {time.time() - start:.2f} secs\n')

                if save_end_epoch:
                    ckpt_save_path = self.ckpt_manager.save()
                    print(f'Saving checkpoint for epoch {e+1} at {ckpt_save_path}')

                if "val" in data_dict.keys():
                    print(f"Running through the validation set now!")
                    self.run_validation_MQA(e, save_filepath_val, data_dict["val"], num_aux_tokens)  # note e+1 is not correct.

    def val_step_MQA(self, input_string, input_id, label_id, aux_label, aoint_indices, sample_weights, num_aux_tokens):
        # look ahead padding is handled by huggingface gpt2 model.
        gpt_pad_mask = create_padding_mask_gpt(input_id, padding_id=self.padding_id)  # the model will handel whether or not to use this mask.
        mask = create_combined_mask(input_id, padding_id=self.padding_id, num_aux_tok=num_aux_tokens)  # no look ahead padding for the num_aux_tokens auxiliary tokens.

        loss, size = 0, 0

        with tf.GradientTape() as tape:
            vanilla_set_output, _, task_prediction, _, _ = self.model(input_string, input_id, training=False,
                                                                      mask=mask,
                                                                      gpt_pad_mask=gpt_pad_mask,
                                                                      reading_strat_mc_bool=self.reading_strat_mc_bool,
                                                                      vanilla_set_aux_loss_bool=False,
                                                                      fixed_output=self.fixed_output,
                                                                      stop_gradient=self.stop_gradient)
            # ret (vanilla_set_output, nm_decoder_mc, task_prediction, gating_weights, attention_weights)
            # vanilla_set_output is after it has been passed through dense layer if vanilla_set_aux_loss_bool is True

            loss, size = self.loss_function(label_id, task_prediction, self.loss_object, self.padding_id)

        loss = tf.convert_to_tensor([loss], dtype=tf.dtypes.float32)
        size = tf.convert_to_tensor([size], dtype=tf.dtypes.float32)

        return loss, size

    @tf.function
    def _distributed_val_step_tf_function_MQA(self, input_string, input_id, label_id, aux_label, aoint_indices,
                                              sample_weights, num_aux_tokens):
        loss_dec, size_dec = None, None
        if self.strategy is not None:
            loss_dec, size_dec = self.strategy.run(self.val_step_MQA, args=(
                input_string, input_id, label_id, aux_label, aoint_indices, sample_weights, num_aux_tokens,))

            if self.strategy.num_replicas_in_sync > 1:
                loss_dec = tf.reduce_sum(loss_dec.values)
                size_dec = tf.reduce_sum(size_dec.values)
            else:
                loss_dec = tf.reduce_sum(loss_dec)
                size_dec = tf.reduce_sum(size_dec)
        else:
            loss_dec, size_dec = self.val_step_MQA(input_string, input_id, label_id, aux_label, aoint_indices,
                                                   sample_weights, num_aux_tokens)

        return loss_dec, size_dec

    def _distributed_val_step_no_tf_function_MQA(self, input_string, input_id, label_id, aux_label, aoint_indices,
                                                 sample_weights, num_aux_tokens):
        loss_dec, size_dec = None, None
        if self.strategy is not None:
            loss_dec, size_dec = self.strategy.run(self.val_step_MQA, args=(
                input_string, input_id, label_id, aux_label, aoint_indices, sample_weights, num_aux_tokens,))

            if self.strategy.num_replicas_in_sync > 1:
                loss_dec = tf.reduce_sum(loss_dec.values)
                size_dec = tf.reduce_sum(size_dec.values)
            else:
                loss_dec = tf.reduce_sum(loss_dec)
                size_dec = tf.reduce_sum(size_dec)
        else:
            loss_dec, size_dec = self.val_step_MQA(input_string, input_id, label_id, aux_label, aoint_indices,
                                                   sample_weights, num_aux_tokens)

        return loss_dec, size_dec

    def run_validation_MQA(self, e, save_filepath_val, data, num_aux_tokens):

        start = time.time()
        epoch_loss_total = 0
        epoch_size_total = 0
        for (input_string, input_id, label_id, aux_label, aoint_indices, sample_weights) in data:

            loss_dec, size_dec = None, None
            if self.fixed_output:
                loss_dec, size_dec = self._distributed_val_step_tf_function_MQA(input_string, input_id, label_id, aux_label,
                                                                        aoint_indices, sample_weights, num_aux_tokens)
            else:
                loss_dec, size_dec = self._distributed_val_step_no_tf_function_MQA(input_string, input_id, label_id,
                                                                               aux_label, aoint_indices,
                                                                               sample_weights, num_aux_tokens)
            if size_dec == 0:
                print(f"The size is zero for the currenet batch, skipping to next batch!")
                continue

            epoch_loss_total += loss_dec
            epoch_size_total += size_dec

        total_loss = epoch_loss_total / epoch_size_total

        print(f'Validation Epoch: {e+1} Loss: {total_loss:.4f}')
        print(f'Time taken for validation epoch {e+1}: {time.time()-start:.2f} secs\n')

        header = True if e == 0 else False
        self._save_epoch_loss_only("val", e+1, save_filepath_val, header, total_loss)

    def val_step_GQA(self, input_string, input_id, label_id, aux_label, sample_weights, num_aux_tokens):
        # look ahead padding is handled by huggingface gpt2 model.
        gpt_pad_mask = create_padding_mask_gpt(input_id, padding_id=self.padding_id)  # the model will handel whether or not to use this mask.
        mask = create_combined_mask(input_id, padding_id=self.padding_id,
                                    num_aux_tok=num_aux_tokens)  # no look ahead padding for the num_aux_tokens auxiliary tokens.

        loss, size = 0, 0

        with tf.GradientTape() as tape:
            vanilla_set_output, _, task_prediction, _, _ = self.model(input_string, input_id, training=False,
                                                                      mask=mask,
                                                                      gpt_pad_mask=gpt_pad_mask,
                                                                      reading_strat_mc_bool=self.reading_strat_mc_bool,
                                                                      vanilla_set_aux_loss_bool=False,
                                                                      fixed_output=self.fixed_output,
                                                                      stop_gradient=self.stop_gradient)
            # ret (vanilla_set_output, nm_decoder_mc, task_prediction, gating_weights, attention_weights)
            # vanilla_set_output is after it has been passed through dense layer if vanilla_set_aux_loss_bool is True

            loss, size = self.loss_function(label_id, task_prediction, self.loss_object, self.padding_id)

        loss = tf.convert_to_tensor([loss], dtype=tf.dtypes.float32)
        size = tf.convert_to_tensor([size], dtype=tf.dtypes.float32)

        return loss, size

    @tf.function
    def _distributed_val_step_tf_function_GQA(self, input_string, input_id, label_id, aux_label, sample_weights,
                                              num_aux_tokens):
        loss_dec, size_dec = None, None
        if self.strategy is not None:
            loss_dec, size_dec = self.strategy.run(self.val_step_GQA, args=(
                input_string, input_id, label_id, aux_label, sample_weights, num_aux_tokens,))

            if self.strategy.num_replicas_in_sync > 1:
                loss_dec = tf.reduce_sum(loss_dec.values)
                size_dec = tf.reduce_sum(size_dec.values)
            else:
                loss_dec = tf.reduce_sum(loss_dec)
                size_dec = tf.reduce_sum(size_dec)
        else:
            loss_dec, size_dec = self.val_step_GQA(input_string, input_id, label_id, aux_label, sample_weights,
                                                     num_aux_tokens)

        return loss_dec, size_dec

    def _distributed_val_step_no_tf_function_GQA(self, input_string, input_id, label_id, aux_label, sample_weights,
                                                 num_aux_tokens):
        loss_dec, size_dec = None, None
        if self.strategy is not None:
            loss_dec, size_dec = self.strategy.run(self.val_step_GQA, args=(
                input_string, input_id, label_id, aux_label, sample_weights, num_aux_tokens,))

            if self.strategy.num_replicas_in_sync > 1:
                loss_dec = tf.reduce_sum(loss_dec.values)
                size_dec = tf.reduce_sum(size_dec.values)
            else:
                loss_dec = tf.reduce_sum(loss_dec)
                size_dec = tf.reduce_sum(size_dec)
        else:
            loss_dec, size_dec = self.val_step_GQA(input_string, input_id, label_id, aux_label, sample_weights,
                                                   num_aux_tokens)

        return loss_dec, size_dec

    def run_validation_GQA(self, e, save_filepath_val, data, num_aux_tokens):

        start = time.time()
        epoch_loss_total = 0
        epoch_size_total = 0
        for (input_string, input_id, label_id, aux_label, sample_weights) in data:

            loss_dec, size_dec = None, None
            if self.fixed_output:
                loss_dec, size_dec = self._distributed_val_step_tf_function_GQA(input_string, input_id, label_id,
                                                                                aux_label, sample_weights,
                                                                                num_aux_tokens)
            else:
                loss_dec, size_dec = self._distributed_val_step_no_tf_function_GQA(input_string, input_id, label_id,
                                                                            aux_label, sample_weights, num_aux_tokens)
            if size_dec == 0:
                print(f"The size is zero for the currenet batch, skipping to next batch!")
                continue

            epoch_loss_total += loss_dec
            epoch_size_total += size_dec

        total_loss = epoch_loss_total / epoch_size_total

        print(f'Validation Epoch: {e+1} Loss: {total_loss:.4f}')
        print(f'Time taken for validation epoch {e+1}: {time.time()-start:.2f} secs\n')

        header = True if e == 0 else False
        self._save_epoch_loss_only("val", e+1, save_filepath_val, header, total_loss)

    def test_MQA_label_only_accuracy(self, input_string, input_id, all_labels, correct_label, aoint_indices,
                                     num_aux_tokens, max_generate_len):

        gpt_pad_mask = create_padding_mask_gpt(input_id, padding_id=self.padding_id)  # look ahead padding is handled by huggingface gpt2 model.
        mask = create_combined_mask(input_id, padding_id=self.padding_id, num_aux_tok=num_aux_tokens)

        generated_ids = self.model.generate_answer(input_string, input_id, training=False, mask=mask,
                                                   gpt_pad_mask=gpt_pad_mask,
                                                   reading_strat_mc_bool=self.reading_strat_mc_bool,
                                                   vanilla_set_aux_loss_bool=False,
                                                   stop_gradient=self.stop_gradient,
                                                   fixed_output=self.fixed_output, pad_tok_id=self.padding_id,
                                                   end_tok_id=self.end_tok_id, gen_len_max=1)

        convert_byte_to_string = lambda i: i.decode("utf-8")
        correct_ao = [convert_byte_to_string(x) for x in
                      correct_label.numpy().tolist()]  # list of strings. # one for each batch item.

        correct, total = self._accuracy_helper_label(correct_label, generated_ids)  # for a fully generated answer.
        # cor, tot = self._accuracy_helper2(correct_ao, generated_labels) # for a single label only...

        print(f"Batch accuracy: {correct / total} \t Correct: {correct} \t Total: {total}")

        correct = tf.convert_to_tensor([correct], dtype=tf.dtypes.float32)
        total = tf.convert_to_tensor([total], dtype=tf.dtypes.float32)

        return correct, total

    def _accuracy_helper_label(self, cor_label, generated_ids):
        #all_labels and cor_label are list of strings.
        #all_labels
        # generated_ids is a list of integers.
        correct = 0
        total = 0

        for i, batch_item in enumerate(cor_label):
            correct_ans = self.tokenizer.batch_encode([batch_item])["input_ids"][0] # list of integers.
            #print(f"correct_ans \\t generated_ids: {correct_ans} \t {generated_ids[i]}")
            if correct_ans[0] == generated_ids[i][0]:
                correct += 1
            total += 1
        return correct, total

    def _distributed_test_MQA_label_only_accuracy(self, input_string, input_id, all_labels, correct_label,
                                                        aoint_indices, num_aux_tokens, max_generate_len):
        correct, total = None, None
        if self.strategy is not None:
            correct, total = self.strategy.run(self.test_MQA_label_only_accuracy, args=(
                input_string, input_id, all_labels, correct_label, aoint_indices, num_aux_tokens, max_generate_len,))

            # The if else may be totally irrelevant.
            if self.strategy.num_replicas_in_sync > 1:
                correct = tf.reduce_sum(correct.values)
                total = tf.reduce_sum(total.values)
            else:
                correct = tf.reduce_sum(correct)
                total = tf.reduce_sum(total)
        else:
            correct, total = self.test_MQA_label_only_accuracy(input_string, input_id, all_labels, correct_label,
                                                               aoint_indices, num_aux_tokens, max_generate_len)

        return correct, total

    def test_MQA_GQA_accuracy(self, input_string, input_id, all_labels, correct_label,
                              aoint_indices, num_aux_tokens, max_generate_len):

        gpt_pad_mask = create_padding_mask_gpt(input_id, padding_id=self.padding_id)  # look ahead padding is handled by huggingface gpt2 model.
        mask = create_combined_mask(input_id, padding_id=self.padding_id, num_aux_tok=num_aux_tokens)

        generated_ids = self.model.generate_answer(input_string, input_id, training=False, mask=mask,
                                                   gpt_pad_mask=gpt_pad_mask,
                                                   reading_strat_mc_bool=self.reading_strat_mc_bool,
                                                   vanilla_set_aux_loss_bool=False,
                                                   stop_gradient=self.stop_gradient,
                                                   fixed_output=self.fixed_output, pad_tok_id=self.padding_id,
                                                   end_tok_id=self.end_tok_id, gen_len_max=max_generate_len)

        convert_byte_to_string = lambda i: i.decode("utf-8")

        all_labels = [convert_byte_to_string(x) for x in all_labels.numpy().tolist()]  # list of strings.
        correct_label = [convert_byte_to_string(x) for x in correct_label.numpy().tolist()]  # list of strings.


        correct, total = self._accuracy_helper(all_labels, correct_label, generated_ids)  # for a fully generated answer.
        print(f"Batch accuracy: {correct / total}")

        correct = tf.convert_to_tensor([correct], dtype=tf.dtypes.float32)
        total = tf.convert_to_tensor([total], dtype=tf.dtypes.float32)
        return correct, total

    def _accuracy_helper(self, all_labels, cor_label, generated_ids):
        # all_labels and cor_label are list of strings.
        # all_labels
        # generated_ids is a list of integers.
        correct = 0
        total = 0

        all_labels_split = []
        for batch_item in all_labels:
            temp_list = re.split(r"\(\d\)", batch_item)  # batch item will be a string.
            # temp_list if a list of strings, with each item being an answer option.
            for i in range(len(temp_list)-1, -1, -1):
                if temp_list[i].strip() == "":
                    temp_list.pop(i)
                else:
                    temp_list[i] = temp_list[i].strip()

            all_labels_split.append(temp_list) # all_labels_split is a list containing each batch item, which itself
            # contains all answer options. [["answer option 1", "answer option 2", ...], ..., []]

        for i, batch_item in enumerate(all_labels_split):  # batch_item will be a list of strings, each representing one answer option.
            temp1 = self.tokenizer.batch_encode(batch_item)["input_ids"]  # answer options. list of integers.

            indice = self._intersection_ids_counter(temp1, generated_ids[i]) # indice is an integer.

            cor, tot = self._check_correct_helper(indice, cor_label[i])
            correct += cor
            total += tot
        return correct, total

    def _intersection_ids_counter(self, all_labels, answer):
        # note: this should work for both strings and integers.
        # all_labels = [[], [], []] # note: that order should be preserved.
        # answer = []
        answer_unique = set(answer)
        all_labels_unique = [] # list of sets.
        for lst in all_labels: all_labels_unique.append(set(lst))

        label_scores = []
        for set_ in all_labels_unique:
            total_unique_union = set_.union(answer_unique)
            total_intersection = set_.intersection(answer_unique)

            label_scores.append(len(total_intersection) / len(total_unique_union))

        max_score = max(label_scores)
        max_indices = [i for i, j in enumerate(label_scores) if j == max_score]

        # note: bias towards first elment in list instead of say random for consistency if a tie.
        return max_indices[0] + 1  # returns an integer from 1 to len(all_labels) inclusive.

    def _check_correct_helper(self, indice, answer):
        # indice will represent the answer position -- i.e. 1 means A, 2 means B, ...
        # answer will be in string format.
        correct = 0
        total = 0
        if indice == 1:
            if answer.strip() == "(1)":
                correct += 1
            total += 1
        elif indice == 2:
            if answer.strip() == "(2)":
                correct += 1
            total += 1
        elif indice == 3:
            if answer.strip() == "(3)":
                correct += 1
            total += 1
        elif indice == 4:
            if answer.strip() == "(4)":
                correct += 1
            total += 1
        elif indice == 5:
            if answer.strip() == "(5)":
                correct += 1
            total += 1
        elif indice == 6:
            if answer.strip() == "(6)":
                correct += 1
            total += 1
        elif indice == 7:
            if answer.strip() == "(7)":
                correct += 1
            total += 1
        elif indice == 8:
            if answer.strip() == "(8)":
                correct += 1
            total += 1
        elif indice == 9:
            if answer.strip() == "(9)":
                correct += 1
            total += 1
        else:
            raise Exception(f"Invalid index value: {indice}!")
        return correct, total

    def _distributed_test_MQA_GQA_accuracy(self, input_string, input_id, label_id, aux_label,
                                           aoint_indices, sample_weights, num_aux_tokens, max_generate_len):
        correct, total = None, None
        if self.strategy is not None:
            correct, total = self.strategy.run(self.test_MQA_GQA_accuracy, args=(
                input_string, input_id, all_labels, correct_label, aoint_indices, num_aux_tokens, max_generate_len,))

            # The if else may be totally irrelevant.
            if self.strategy.num_replicas_in_sync > 1:
                correct = tf.reduce_sum(correct.values)
                total = tf.reduce_sum(total.values)
            else:
                correct = tf.reduce_sum(correct)
                total = tf.reduce_sum(total)
        else:
            correct, total = self.test_MQA_GQA_accuracy(input_string, input_id, all_labels, correct_label,
                                                        aoint_indices, num_aux_tokens, max_generate_len)

        return correct, total

    def test_GQA(self, input_string, input_id, answers, num_aux_tokens,
                 max_generate_len, metrics, multiple_answers):

        gpt_pad_mask = create_padding_mask_gpt(input_id, padding_id=self.padding_id)  # look ahead padding is handled by huggingface gpt2 model.
        mask = create_combined_mask(input_id, padding_id=self.padding_id, num_aux_tok=num_aux_tokens)

        f1_score, f1_score_count = 0, 0
        em_score, em_score_count = 0, 0
        rouge_l_score_f, rouge_l_score_f_count = 0, 0
        generated_ids = self.model.generate_answer(input_string, input_id, training=False, mask=mask,
                                                   gpt_pad_mask=gpt_pad_mask,
                                                   reading_strat_mc_bool=self.reading_strat_mc_bool,
                                                   vanilla_set_aux_loss_bool=False,
                                                   stop_gradient=self.stop_gradient,
                                                   fixed_output=self.fixed_output, pad_tok_id=self.padding_id,
                                                   end_tok_id=self.end_tok_id, gen_len_max=max_generate_len)

        convert_byte_to_string = lambda i: i.decode("utf-8")
        if not multiple_answers: answers_ = [convert_byte_to_string(x) for x in answers.numpy().tolist()]#list of strings
        else:
            answers_ = []
            tmp_list = answers.numpy().tolist() #list of lists of strings, each list within in a different answer option
            for batch_item in tmp_list:
                answers_.append([convert_byte_to_string(x) for x in batch_item])


        # "f1-score", "em-score", "rouge-l-score"
        if "f1-score" in metrics:
            f1_score, f1_score_count = self._f1_score_helper(generated_ids, answers_, multiple_answers)
        if "em-score" in metrics:
            em_score, em_score_count = self._em_score_helper(generated_ids, answers_, multiple_answers)
        if "rouge-l-score" in metrics:
            rouge_l_score_f, rouge_l_score_f_count = self._rouge_l_score_helper(generated_ids, answers_, multiple_answers)

        if f1_score_count == 0: print_f1=0
        else: print_f1=f1_score/f1_score_count

        if em_score_count == 0: print_em=0
        else: print_em=em_score/em_score_count

        if rouge_l_score_f_count == 0: print_rouge=0
        else: print_rouge=rouge_l_score_f/rouge_l_score_f_count

        print(f"F1-score: {print_f1}\tEm-score: {print_em}\tRouge-l-score (f): {print_rouge}")

        f1_score = tf.convert_to_tensor([f1_score], dtype=tf.dtypes.float32)
        em_score = tf.convert_to_tensor([em_score], dtype=tf.dtypes.float32)
        rouge_l_score_f = tf.convert_to_tensor([rouge_l_score_f], dtype=tf.dtypes.float32)

        f1_score_count = tf.convert_to_tensor([f1_score_count], dtype=tf.dtypes.float32)
        em_score_count = tf.convert_to_tensor([em_score_count], dtype=tf.dtypes.float32)
        rouge_l_score_f_count = tf.convert_to_tensor([rouge_l_score_f_count], dtype=tf.dtypes.float32)

        return f1_score, em_score, rouge_l_score_f, f1_score_count, em_score_count, rouge_l_score_f_count

    # All code below that is used in f1-score calculation has been taken from
    # https://github.com/allenai/unifiedqa/blob/master/evaluation/evaluate_squad2.py
    # so that similar processing is done.

    def _f1_score_helper(self, generated_ids, answers_, multiple_answers):
        # TODO testing still needed.
        generated_strings = self.tokenizer.batch_decode(generated_ids) # list of strings for each batch item, each representing an answer.

        f1_function = lambda p, r: (2 * p * r) / (p + r)

        f1_score_sum = 0
        #print(f"generated_strings: {generated_strings}")
        count = len(generated_strings)
        #print(f"count: {count}")
        for i, pred_ans in enumerate(generated_strings): # answer is a string
            precision = None
            recall = None

            pred_answer_lst = self.get_tokens(pred_ans)
            #print(f"pred_answer_lst: {pred_answer_lst}")
            if not multiple_answers: # then it is a list of strings. ["","","",""] each string is from a different answer.
                answer_lst = self.get_tokens(answers_[i]) # ["",""] # strings are from a single answer.

                common = collections.Counter(answer_lst) & collections.Counter(pred_answer_lst)
                num_same = sum(common.values())
                if len(answer_lst) == 0 or len(pred_answer_lst) == 0:
                    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
                    f1_score_sum += int(answer_lst == pred_answer_lst)
                    continue
                if num_same == 0: continue
                precision = 1.0 * num_same / len(pred_answer_lst)
                recall = 1.0 * num_same / len(answer_lst)
                f1_score_sum += f1_function(precision, recall)
            else: # list of list of strings. [["",""],["",""],["",""]]
                answer_lst = [self.get_tokens(x) for x in answers_[i]]
                #print(f"answer_lst: {answer_lst}\nanswers_[i]: {answers_[i]}")
                # [["","",""], ["","",""]] # each string will be an individual token and the outer list represents.
                max_f1_score = 0
                for zz, ans in enumerate(answer_lst): # and wll be a list of strings, each part of a single answer.
                    if ans == "<pad>": continue
                    common = collections.Counter(ans) & collections.Counter(pred_answer_lst)
                    #print(f"common.values: {common.values()}")
                    num_same = sum(common.values())
                    if len(ans) == 0 or len(pred_answer_lst) == 0:
                        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
                        max_f1_score = max(int(ans == pred_answer_lst), max_f1_score)
                    if num_same == 0: continue
                    precision = 1.0 * num_same / len(pred_answer_lst)
                    recall = 1.0 * num_same / len(ans)
                    #print(f"precision: {precision}\nrecall: {recall}\nf1-score: {f1_function(precision, recall)}")
                    max_f1_score = max(f1_function(precision, recall), max_f1_score)
                f1_score_sum += max_f1_score
        #print(f"f1_score_sum: {f1_score_sum}\n"
        #      f"count: {count}")
        return f1_score_sum, count

    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def get_tokens(self, s): # s is a string
        if not s: return []
        return self.normalize_answer(s).split()

    # from https://github.com/allenai/unifiedqa/blob/master/evaluation/evaluate_squad2.py
    def compute_exact(self, a_gold, a_pred):
        return int(self.normalize_answer(a_gold) == self.normalize_answer(a_pred))

    def _em_score_helper(self, generated_ids, answers_, multiple_answers):
        # TODO testing still needed.
        generated_strings = self.tokenizer.batch_decode(generated_ids)  # list of strings for each batch item, each representing an answer.

        em_score_sum = 0
        count = len(generated_strings)
        for i, pred_ans in enumerate(generated_strings):  # answer is a string
            # pred_ans is a string.
            if not multiple_answers:
                # answers_[i] is a string.
                em_score_sum += self.compute_exact(answers_[i], pred_ans)
            else:
                answer_lst = [self.get_tokens(x) for x in answers_[i]]
                # [["","",""], ["","",""]] # each string will be an individual token and the outer list represents.
                max_em_score = 0
                for ans in answer_lst:  # ans will be a string.
                    max_em_score = max(self.compute_exact(ans, pred_ans), max_em_score)
                em_score_sum += max_em_score
        return em_score_sum, count

    def _rouge_l_score_helper(self, generated_ids, answers_, multiple_answers):

        generated_strings = self.tokenizer.batch_decode(generated_ids)  # list of strings for each batch item, each representing an answer.

        rougel_score_sum = 0
        count = len(generated_strings)
        for i, pred_ans in enumerate(generated_strings):  # answer is a string
            # pred_ans is a string.
            if not multiple_answers:
                # answers_[i] is a string.
                rougel_score_sum += rouge_l(pred_ans, answers_[i])["rouge-l"]["f"]
            else:
                answer_lst = [self.get_tokens(x) for x in answers_[i]]

                max_rougel_score = 0
                for ans in answer_lst:  # ans will be a string.
                    max_rougel_score = max(rouge_l(pred_ans, ans)["rouge-l"]["f"], max_rougel_score)
                rougel_score_sum += max_rougel_score
        return rougel_score_sum, count

    def _distributed_test_GQA(self, input_string, input_id, answers, num_aux_tokens, max_generate_len,
                              metrics, multiple_answers):

        f1_score, f1_score_count = None, None
        em_score, em_score_count = None, None
        rouge_l_score_f, rouge_l_score_f_count = None, None
        if self.strategy is not None:
            f1_score, em_score, rouge_l_score_f, f1_score_count, em_score_count, rouge_l_score_f_count = \
                self.strategy.run(self.test_GQA, args=(input_string, input_id, answers, num_aux_tokens,
                                                       max_generate_len, metrics, multiple_answers,))

            if self.strategy.num_replicas_in_sync > 1:
                f1_score = tf.reduce_sum(f1_score.values)
                em_score = tf.reduce_sum(em_score.values)
                rouge_l_score_f = tf.reduce_sum(rouge_l_score_f.values)
                f1_score_count = tf.reduce_sum(f1_score_count.values)
                em_score_count = tf.reduce_sum(em_score_count.values)
                rouge_l_score_f_count = tf.reduce_sum(rouge_l_score_f_count.values)
            else:
                f1_score = tf.reduce_sum(f1_score)
                em_score = tf.reduce_sum(em_score)
                rouge_l_score_f = tf.reduce_sum(rouge_l_score_f)
                f1_score_count = tf.reduce_sum(f1_score_count)
                em_score_count = tf.reduce_sum(em_score_count)
                rouge_l_score_f_count = tf.reduce_sum(rouge_l_score_f_count)
        else:
            f1_score, em_score, rouge_l_score_f, f1_score_count, em_score_count, rouge_l_score_f_count = \
                self.test_GQA(input_string, input_id, answers, num_aux_tokens,
                              max_generate_len, metrics, multiple_answers)

        return f1_score, em_score, rouge_l_score_f, f1_score_count, em_score_count, rouge_l_score_f_count

    def get_test_results(self, e, save_filepath, data, num_aux_tokens, max_generate_len=100,
                         filename_prefix="test", metrics=[], mode=None, multiple_answers=False):
        # multiple_answers -> some datasets have multiple answers, this boolean value indicates this so the code
        #   further down the pipeline can handle it.

        # possible_metric values: "accuracy", "f1-score", "em-score", "rouge-l-score"

        assert len(metrics) > 0, f"a metric needs to be chosen"
        assert mode is not None, f"the mode must not be None!"


        # note: there is no @tf.function during evaluation/test mode.
        if mode == "MQA_label_only": # only generate one label.
            if len(metrics) == 1:
                if metrics[0] == "accuracy":
                    start = time.time()
                    correct_samples, total_samples = 0, 0
                    for (input_string, input_id, all_labels, correct_label, aoint_indices) in data:

                        # note: all_labels isn't used in the instance below.
                        correct, total = self._distributed_test_MQA_label_only_accuracy(input_string, input_id, all_labels, correct_label,
                                                                       aoint_indices, num_aux_tokens, max_generate_len)

                        correct_samples += correct
                        total_samples += total

                    total_accuracy = correct_samples / total_samples
                    print(f'Test Accuracy {total_accuracy} correct: {correct_samples} total: {total_samples}')
                    print(f'Time taken: {time.time() - start:.2f} secs\n')

                    header = True if e == 0 else False
                    self._save_epoch_accuracy_only(filename_prefix, e+1, save_filepath, header, total_accuracy,
                                                   correct_samples, total_samples)


        elif mode == "MQA_label_GQA": # Generate a label, but wait until </s>---the end token---is reached.
            raise Exception(f"{mode} is currently not supported!")
        elif mode == "MQA_GQA": # Generate the whole answer, not the label. Stop when </s>---the end token---is reached.
            if len(metrics) == 1:
                if metrics[0] == "accuracy":
                    start = time.time()
                    correct_samples, total_samples = 0, 0
                    for (input_string, input_id, all_labels, correct_label, aoint_indices) in data:

                        # note: all_labels isn't used in the instance below.
                        correct, total = self._distributed_test_MQA_GQA_accuracy(input_string, input_id, all_labels,
                                                                                 correct_label, aoint_indices,
                                                                                 num_aux_tokens, max_generate_len)

                        correct_samples += correct
                        total_samples += total

                    total_accuracy = correct_samples / total_samples
                    print(f'Test Accuracy {total_accuracy} correct: {correct_samples} total: {total_samples}')
                    print(f'Time taken: {time.time() - start:.2f} secs\n')

                    header = True if e == 0 else False
                    self._save_epoch_accuracy_only(filename_prefix, e+1, save_filepath, header, total_accuracy,
                                                   correct_samples, total_samples)
        elif mode == "GQA": # Generate an answer, stop when </s>---the end token---is reached.
            start = time.time()
            f1_score, f1_score_count = 0, 0
            em_score, em_score_count = 0, 0
            rouge_l_score_f, rouge_l_score_f_count = 0, 0
            for (input_string, input_id, answers) in data:
                # note: all_labels isn't used in the instance below.
                f1_score_, em_score_, rouge_l_score_f_, f1_score_count_, em_score_count_, rouge_l_score_f_count_ = \
                    self._distributed_test_GQA(input_string, input_id, answers, num_aux_tokens, max_generate_len,
                                               metrics, multiple_answers)

                f1_score += f1_score_
                f1_score_count += f1_score_count_
                em_score += em_score_
                em_score_count += em_score_count_
                rouge_l_score_f += rouge_l_score_f_
                rouge_l_score_f_count += rouge_l_score_f_count_

            macro_f1_score, avg_em_score, avg_rouge_l_score = None, None, None
            if f1_score_count != 0: macro_f1_score = f1_score / f1_score_count
            if em_score_count != 0: avg_em_score = em_score / em_score_count
            if rouge_l_score_count != 0: avg_rouge_l_score = rouge_l_score_f / rouge_l_score_count

            print(f'Test F1-score (macro): {macro_f1_score} em_score (avg): {avg_em_score} '
                  f'rouge_l_score_f (avg): {avg_rouge_l_score}')
            print(f'Time taken: {time.time() - start:.2f} secs\n')

            header = True if e == 0 else False
            self._save_epoch_GQA(filename_prefix, e+1, save_filepath, header, macro_f1_score, avg_em_score,
                                 avg_rouge_l_score, f1_score, em_score, rouge_l_score_f, f1_score_count,
                                 em_score_count, rouge_l_score_count)


    # functions for printing the results.

    def _save_batch_loss_only(self, type_, epoch, batch, save_filepath, header, loss):

        assert isinstance(type_, str) and isinstance(save_filepath, str)
        file = save_filepath + type_ + "batch" + ".txt"
        with open(file, "a") as f:
            if header: f.write("Epoch Batch Loss \n")
            f.write(f"{epoch} {batch} {loss} \n")

    def _save_epoch_loss_only(self, type_, epoch, save_filepath, header, loss):

        assert isinstance(type_, str) and isinstance(save_filepath, str)
        file = save_filepath + type_ + "epoch" + ".txt"
        with open(file, "a") as f:
            if header: f.write("Epoch Loss \n")
            f.write(f"{epoch} {loss} \n")

    # functions for saving accuracy.
    def _save_epoch_accuracy_only(self, type_, epoch, save_filepath, header, accuracy, correct, total):

        assert isinstance(type_, str) and isinstance(save_filepath, str)
        file = save_filepath + type_ + "epoch" + ".txt"
        with open(file, "a") as f:
            if header: f.write("Epoch Accuracy Correct Total \n")
            f.write(f"{epoch} {accuracy} {correct} {total} \n")

    def _save_batch_accuracy_only(self, type_, epoch, batch, save_filepath, header, accuracy, correct, total):

        assert isinstance(type_, str) and isinstance(save_filepath, str)
        file = save_filepath + type_ + "batch" + ".txt"
        with open(file, "a") as f:
            if header: f.write("Epoch Batch Accuracy Correct Total \n")
            f.write(f"{epoch} {batch} {accuracy} {correct} {total}\n")

    # Save f1-score, em-score, rouge-l score.
    def _save_epoch_GQA(self, type_, epoch, save_filepath, header, macro_f1_score, avg_em_score,
                        avg_rouge_l_score, f1_score, em_score, rouge_l_score_f, f1_score_count,
                        em_score_count, rouge_l_score_count):

        assert isinstance(type_, str) and isinstance(save_filepath, str)
        file = save_filepath + type_ + "batch" + ".txt"
        with open(file, "a") as f:
            if header: f.write("Epoch Macro-f1-score Avg-em-score Avg-rouge-l-score F1-score Em-score Rouge-l-score"
                               "F1-score-count Em-score-count Rouge-l-score-count \n")
            f.write(f"{epoch}\t{macro_f1_score}\t{avg_em_score}\t{avg_rouge_l_score}\t{f1_score}" 
                    f"\t{em_score}\t{rouge_l_score_f}\t{f1_score_count}\t{em_score_count}\t{rouge_l_score_count}\n")

def loss_function(target, prediction, loss_object, padding_id):

    mask = tf.math.logical_not(tf.math.equal(target, padding_id))  # padding mask
    loss_ = loss_object(target, prediction) # get loss from the loss_object
    mask = tf.cast(mask, dtype=loss_.dtype) # convert the mask to the correct format.
    loss_ *= mask # apply masking to positions that correspond to a <pad> token in the target.
    return tf.reduce_sum(loss_), tf.reduce_sum(mask)