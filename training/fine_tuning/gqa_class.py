# General version of SQuAD train class -> generalised to all tasks in the unifiedQA format that are not multiple
# choice quetsion answering.

import tensorflow as tf
import numpy as np
import math
import time
import re

#import checkmate
#from chedecckmate import BestCheckPointSaver

import sys
sys.path.append("../..")

from models.AttentionMasks import *
from training.fine_tuning.parent_fine_tune_train import *

class GQA_class(ParentFineTuningNL):
    '''
    Description: Parent class that implements basic functions needed for fine-tuning/training.
    Input:
        (note: tensorflow models/losses and optimizers are the only that are supported,
            however the only tokenizers that are supported are from huggingface wrapped up in custom made tokenizer class)
        model: The previously initialized model.
        optimizer: The optimizer to use during training.
        loss_object: e.g. SparseCategoricalCrossentropy
        loss_function: Uses the loss_object to get the loss.
        tokenizer: huggingface transformer.
        checkpoint_path_recent: (string) path to the recent checkpoint folder.
        strategy: (tf.distribute...) Strategy for splitting the training across multiple gpus.
    '''
    def __init__(self, model, optimizer, loss_object, loss_function, tokenizer,
                 checkpoint_path_recent, strategy, pad_token="<pad>", end_tok="</s>",
                 recent_to_keep=5, load_recent=False, load_specific_path='',
                 enc_tok_id=None, dec_tok_id=None, output_layer_name="lm", fine_tuning=True,
                 fixed_output=True, stop_gradient=False, reading_strat_mc_bool=False,
                 vanilla_set_aux_loss_bool=False, lambda_vanilla_set=0.5, lambda_lm=0.1,
                 lm_aux_loss_global=False, train_cutoff=0):

        super(GQA_class, self).__init__(model, optimizer, loss_object, loss_function, tokenizer,
                                              checkpoint_path_recent, strategy, pad_token, end_tok,
                                              recent_to_keep, load_recent, load_specific_path,
                                              enc_tok_id, dec_tok_id, output_layer_name, fine_tuning,
                                              fixed_output, stop_gradient, reading_strat_mc_bool,
                                              vanilla_set_aux_loss_bool, lambda_vanilla_set, lambda_lm,
                                              lm_aux_loss_global, train_cutoff)

    def train_step(self, input_string, input_id, label_id, aux_label, sample_weights, num_aux_tokens):

        gpt_pad_mask = create_padding_mask_gpt(input_id, padding_id=self.padding_id)  # look ahead padding is handled by huggingface gpt2 model.
        mask = create_combined_mask(input_id, padding_id=self.padding_id, num_aux_tok=num_aux_tokens)
        #mask = create_padding_mask(input_id, padding_id=self.padding_id)

        loss, size = 0, 0
        train_ = False if self.train_epoch < self.train_cutoff else True
        with tf.GradientTape() as tape:
            vanilla_set_output, _, task_prediction, _, _ = self.model(input_string, input_id, training=train_, mask=mask,
                                                                      gpt_pad_mask=gpt_pad_mask,
                                                                      reading_strat_mc_bool=self.reading_strat_mc_bool,
                                                                      vanilla_set_aux_loss_bool=self.vanilla_set_aux_loss_bool,
                                                                      fixed_output=self.fixed_output,
                                                                      stop_gradient=self.stop_gradient)
            # ret (vanilla_set_output, nm_decoder_mc, task_prediction, gating_weights, attention_weights)
            #vanilla_set_output is after it has been passed through dense layer...

            loss, size = self.loss_function(label_id, task_prediction, self.loss_object, self.padding_id)
            loss_ = loss / size
            # uncomment below for an auxiliary loss...
            #print(f"vanilla_set_output: {vanilla_set_output}")
            if self.vanilla_set_aux_loss_bool:
                loss_aux, size_aux = self.loss_function(label_id, vanilla_set_output, self.loss_object, self.padding_id)
                loss_aux_ = loss_aux / size_aux
                loss_ = loss_ + (loss_aux_*self.lambda_vanilla_set)
            if self.lm_aux_loss_global:
                loss_aux, size_aux = self.loss_function(aux_label, task_prediction, self.loss_object, self.padding_id)
                loss_aux_ = loss_aux / size_aux
                loss_ = loss_ + (loss_aux_ * self.lambda_lm)

        gradients = tape.gradient(loss_, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        loss = tf.convert_to_tensor([loss], dtype=tf.dtypes.float32)
        size = tf.convert_to_tensor([size], dtype=tf.dtypes.float32)

        return loss, size

    @tf.function
    def _distributed_train_step(self, input_string, input_id, label_id, aux_label,  sample_weights, num_aux_tokens):
        loss_dec, size_dec = None, None
        if self.strategy is not None:
            loss_dec, size_dec = self.strategy.run(self.train_step, args=(input_string, input_id, label_id, aux_label,  sample_weights, num_aux_tokens,))

            # The if else may be totally irrelevant.
            if self.strategy.num_replicas_in_sync > 1:
                loss_dec = tf.reduce_sum(loss_dec.values)
                size_dec = tf.reduce_sum(size_dec.values)
            else:
                loss_dec = tf.reduce_sum(loss_dec)
                size_dec = tf.reduce_sum(size_dec)
        else:
            loss_dec, size_dec = self.train_step(input_string, input_id, label_id, aux_label,  sample_weights, num_aux_tokens)

        return loss_dec, size_dec

    def train_batch(self, epoch_start, epoch_end, save_filepath_train, save_filepath_val,
                          data_dict, num_aux_tokens, save_end_epoch=True, print_every_iterations=100,
                          save_every_iterations=5000, iteration_counter=0):

        for e in range(epoch_start, epoch_end): # epoch starts at 0...

            start = time.time()

            self.train_epoch = e

            #iteration_counter = 0
            batch = 0
            epoch_loss_total = 0
            epoch_size_total = 0
            for (input_string, input_id, label_id, aux_label, sample_weights) in data_dict["train"]:
                iteration_counter += 1

                loss_dec, size_dec = self._distributed_train_step(input_string, input_id, label_id, aux_label,
                                                                  sample_weights, num_aux_tokens)
                if size_dec == 0:
                    print(f"The size is zero, skip the current batch, it will not be counted due to an error!")
                    continue  # start the next batch.

                epoch_loss_total += loss_dec
                epoch_size_total += size_dec

                loss_ = loss_dec / size_dec
                loss_ = tf.cast(loss_, dtype=tf.dtypes.float32)

                batch += 1
                if iteration_counter % print_every_iterations == 0:
                    print(f'Batch {batch} Epoch {e+1} Loss {loss_:.4f}')

                header = True if (batch == 1 and e == 0) else False
                self._save_batch_loss_only("train", e+1, batch, save_filepath_train, header, loss_)

            total_loss = epoch_loss_total / epoch_size_total

            header = True if e == 0 else False
            self._save_epoch_loss_only("train", e+1, save_filepath_train, header, total_loss)

            print(f'Epoch {e+1} Loss {total_loss:.4f}')
            print(f'Time taken for epoch {e+1}: {time.time() - start:.2f} secs\n')

            if save_end_epoch:
                ckpt_save_path = self.ckpt_manager.save()
                print(f'Saving checkpoint for epoch {e+1} at {ckpt_save_path}')

            if "val" in data_dict.keys():
                print(f"Running through the validation set now!")
                self.run_validation(e, save_filepath_val, data_dict["val"], num_aux_tokens) # note e+1 is not correct.

    def val_step(self, input_string, input_id, label_id, aux_label,  sample_weights, num_aux_tokens):

        gpt_pad_mask = create_padding_mask_gpt(input_id, padding_id=self.padding_id)  # look ahead padding is handled by huggingface gpt2 model.
        mask = create_combined_mask(input_id, padding_id=self.padding_id, num_aux_tok=num_aux_tokens)
        #mask = create_padding_mask(input_id, padding_id=self.padding_id)

        loss, size = 0, 0
        with tf.GradientTape() as tape:
            vanilla_set_output, _, task_prediction, _, _ = self.model(input_string, input_id, training=False, mask=mask,
                                                                      gpt_pad_mask=gpt_pad_mask,
                                                                      reading_strat_mc_bool=self.reading_strat_mc_bool,
                                                                      vanilla_set_aux_loss_bool=self.vanilla_set_aux_loss_bool,
                                                                      fixed_output=self.fixed_output,
                                                                      stop_gradient=self.stop_gradient)
            # ret (vanilla_set_output, nm_decoder_mc, task_prediction, gating_weights, attention_weights)
            #vanilla_set_output is after it has been passed through dense layer...

            loss, size = self.loss_function(label_id, task_prediction, self.loss_object, self.padding_id)
            loss_ = loss / size
            # uncomment below for an auxiliary loss... no need for it here.
            #if vanilla_set_aux_loss_bool:
            #    loss_aux, size_aux = self.loss_function(label_id, vanilla_set_output, self.loss_object, self.padding_id)
            #    loss_aux_ = loss_aux / size_aux
            #    loss_ = loss_ + (loss_aux_*self.lambda_)

        gradients = tape.gradient(loss_, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        loss = tf.convert_to_tensor([loss], dtype=tf.dtypes.float32)
        size = tf.convert_to_tensor([size], dtype=tf.dtypes.float32)

        return loss, size

    @tf.function
    def _distributed_val_step(self, input_string, input_id, label_id, aux_label,  sample_weights, num_aux_tokens):
        loss_dec, size_dec = None, None
        if self.strategy is not None:
            loss_dec, size_dec = self.strategy.run(self.val_step, args=(
            input_string, input_id, label_id, aux_label,  sample_weights, num_aux_tokens,))

            # The if else may be totally irrelevant.
            if self.strategy.num_replicas_in_sync > 1:
                loss_dec = tf.reduce_sum(loss_dec.values)
                size_dec = tf.reduce_sum(size_dec.values)
            else:
                loss_dec = tf.reduce_sum(loss_dec)
                size_dec = tf.reduce_sum(size_dec)
        else:
            loss_dec, size_dec = self.val_step(input_string, input_id, label_id, aux_label,  sample_weights,
                                                 num_aux_tokens)

        return loss_dec, size_dec

    def run_validation(self, e, save_filepath_val, data, num_aux_tokens):

        start = time.time()
        epoch_loss_total = 0
        epoch_size_total = 0
        for (input_string, input_id, label_id, aux_label,  sample_weights) in data:

            loss_dec, size_dec = self._distributed_val_step(input_string, input_id, label_id, aux_label,
                                                              sample_weights, num_aux_tokens)
            if size_dec == 0:
                print(f"The size is zero, skip the current batch, it will not be counted due to an error!")
                continue  # start the next batch.

            epoch_loss_total += loss_dec
            epoch_size_total += size_dec

        total_loss = epoch_loss_total / epoch_size_total

        print(f'Validation Epoch {e+1} Loss {total_loss:.4f}')
        print(f'Time taken for validation epoch {e+1}: {time.time() - start:.2f} secs\n')

        header = True if e == 0 else False
        self._save_epoch_loss_only("val", e+1, save_filepath_val, header, total_loss)


    def test_step_generate(self, input_string, input_id, all_labels, correct_ao,
                  num_aux_tokens, max_generate_len):

        gpt_pad_mask = create_padding_mask_gpt(input_id, padding_id=self.padding_id)  # look ahead padding is handled by huggingface gpt2 model.
        mask = create_combined_mask(input_id, padding_id=self.padding_id, num_aux_tok=num_aux_tokens)
        #mask = create_padding_mask(input_id, padding_id=self.padding_id)

        generated_ids = self.model.generate_answer(input_string, input_id, training=False, mask=mask, gpt_pad_mask=gpt_pad_mask,
                                                   reading_strat_mc_bool=self.reading_strat_mc_bool,
                                                   vanilla_set_aux_loss_bool=self.vanilla_set_aux_loss_bool,
                                                   stop_gradient=self.stop_gradient,
                                                   fixed_output=self.fixed_output, pad_tok_id=self.padding_id,
                                                   end_tok_id=self.end_tok_id, gen_len_max=max_generate_len)

        convert_byte_to_string = lambda i: i.decode("utf-8")
        all_labels = [convert_byte_to_string(x) for x in all_labels.numpy().tolist()]  # list of strings.
        # print(f"all labels: {all_labels}")
        correct_ao = [convert_byte_to_string(x) for x in correct_ao.numpy().tolist()]  # list of strings.
        # print(f"cor_label: {cor_label}")

        #print(f"all_labels: {all_labels} \n correct_ao: {correct_ao} \n generated_answer: {self.tokenizer.batch_decode(generated_ids)}")

        correct, total = self._accuracy_helper(all_labels, correct_ao, generated_ids) # for a fully generated answer.
        #cor, tot = self._accuracy_helper2(correct_ao, generated_labels) # for a single label only...

        print(f"Batch accuracy: {correct / total}")
        # return loss_enc, size_enc, loss_dec, size_dec
        return correct, total

    def _save_epoch_loss_only(self, type_, epoch, save_filepath, header, loss):

        assert isinstance(type_, str) and isinstance(save_filepath, str)
        file = save_filepath + type_ + "epoch" + ".txt"
        with open(file, "a") as f:
            if header: f.write("Epoch Loss \n")
            f.write(f"{epoch} {loss} \n")

    def _save_batch_loss_only(self, type_, epoch, batch, save_filepath, header, loss):

        assert isinstance(type_, str) and isinstance(save_filepath, str)
        file = save_filepath + type_ + "batch" + ".txt"
        with open(file, "a") as f:
            if header: f.write("Epoch Batch Loss \n")
            f.write(f"{epoch} {batch} {loss} \n")

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

def loss_function(target, prediction, loss_object, padding_id):

    mask = tf.math.logical_not(tf.math.equal(target, padding_id))  # padding mask
    loss_ = loss_object(target, prediction) # get loss from the loss_object
    mask = tf.cast(mask, dtype=loss_.dtype) # convert the mask to the correct format.
    loss_ *= mask # apply masking to positions that correspond to a <pad> token in the target.
    return tf.reduce_sum(loss_), tf.reduce_sum(mask)