'''
File name: parent_train.py
Author: Kobe Knowles
Date created: 05/07/21
Data last modified: 29/07/21
Python Version: 3.6
Tensorflow version: 2
'''

import tensorflow as tf
import numpy as np
import math
import time
import re

#import checkmate
#from chedecckmate import BestCheckPointSaver

import sys
sys.path.append("..")

from models.AttentionMasks import *

class ParentFineTuningNL:
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
                 enc_tok_id=None, dec_tok_id=None, output_layer_name="lm"):

        self.model = model
        self.optimizer = optimizer
        self.loss_object = loss_object
        self.loss_function = loss_function
        self.tokenizer = tokenizer

        self.checkpoint_path_recent = checkpoint_path_recent
        self.strategy = strategy

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
            if load_recent: print(f"Note: load_recent command has been overridden.")
            self._restore_specific_checkpoint(load_specific_path)

        assert enc_tok_id is not None and dec_tok_id is not None, f"It is required that the encoder and decoder token id is passed as input during initialization!"
        self.enc_tok_id = enc_tok_id
        self.dec_tok_id = dec_tok_id

        if output_layer_name is not None:
            self.model.fixed_output_layer = self.model.output_layers[output_layer_name] # hardcoded for C4 pre-training

    def create_recent_checkpoint(self, keep):
        self.ckpt = tf.train.Checkpoint(model=self.model,
                                        optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path_recent, max_to_keep=keep)  # maybe remove max_to_keep?

    def _load_recent_checkpoint(self):
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("Latest checkpoint restored.")

    def _restore_specific_checkpoint(self, filepath):
        self.ckpt.restore(filepath)

    def train_step(self, input_string, input_id, label_id, aoint_indices, sample_weights, num_aux_tokens):

        mask = create_combined_mask(input_id, padding_id=self.padding_id, num_aux_tok=num_aux_tokens)

        lambda_ = 0.25

        loss, size = 0, 0
        with tf.GradientTape() as tape:
            vanilla_set_output, _, task_prediction, _, _ = self.model(input_string, input_id, training=True, mask=mask,
                                                                      reading_strat_mc_bool=False,
                                                                      vanilla_set_aux_loss_bool=False,
                                                                      fixed_output=True, fine_tuning=True)
            # ret (vanilla_set_output, nm_decoder_mc, task_prediction, gating_weights, attention_weights)
            #vanilla_set_output is after it has been passed through dense layer...

            loss, size = self.loss_function(label_id, task_prediction, self.loss_object, self.padding_id)
            loss_ = loss / size
            # uncomment below for an auxiliary loss...
            #loss_aux, size_aux = self.loss_function(label_id, vanilla_set_output, self.loss_object, self.padding_id)
            #loss_aux_ = loss_aux / size_aux
            #loss_ = loss_ + (loss_aux_*lambda_)

        gradients = tape.gradient(loss_, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        #gradients2 = tape.gradient(loss_aux_*lambda_, self.trainable_variables)
        #self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss, size

    @tf.function
    def _distributed_train_step(self, input_string, input_id, label_id, aoint_indices, sample_weights, num_aux_tokens):
        loss_dec, size_dec = None, None
        if self.strategy is not None:
            loss_dec, size_dec = self.strategy.run(self.train_step, args=(input_string, input_id, label_id, aoint_indices, sample_weights, num_aux_tokens,))

            # The if else may be totally irrelevant.
            if self.strategy.num_replicas_in_sync > 1:
                loss_dec = tf.reduce_sum(loss_dec.values)
                size_dec = tf.reduce_sum(size_dec.values)
            else:
                loss_dec = tf.reduce_sum(loss_dec)
                size_dec = tf.reduce_sum(size_dec)
        else:
            loss_dec, size_dec = self.train_step(input_string, input_id, label_id, aoint_indices, sample_weights, num_aux_tokens)

        return loss_dec, size_dec

    def train_batch(self, epoch_start, epoch_end, save_filepath_train, save_filepath_val,
                          data_dict, num_aux_tokens, save_end_epoch=True, print_every_iterations=100,
                          save_every_iterations=5000, iteration_counter=0):

        for e in range(epoch_start, epoch_end): # epoch starts at 0...

            start = time.time()

            #iteration_counter = 0
            batch = 0
            epoch_loss_total = 0
            epoch_size_total = 0
            for (input_string, input_id, label_id, aoint_indices, sample_weights) in data_dict["train"]:
                iteration_counter += 1

                loss_dec, size_dec = self._distributed_train_step(input_string, input_id, label_id, aoint_indices,
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

    def val_step(self, input_string, input_id, label_id, aoint_indices, sample_weights, num_aux_tokens):

        mask = create_combined_mask(input_id, padding_id=self.padding_id, num_aux_tok=num_aux_tokens)

        lambda_ = 0.25

        loss, size = 0, 0
        with tf.GradientTape() as tape:
            vanilla_set_output, _, task_prediction, _, _ = self.model(input_string, input_id, training=False, mask=mask,
                                                                      reading_strat_mc_bool=False,
                                                                      vanilla_set_aux_loss_bool=False,
                                                                      fixed_output=True, fine_tuning=True)
            # ret (vanilla_set_output, nm_decoder_mc, task_prediction, gating_weights, attention_weights)
            #vanilla_set_output is after it has been passed through dense layer...

            loss, size = self.loss_function(label_id, task_prediction, self.loss_object, self.padding_id)
            loss_ = loss / size
            # uncomment below for an auxiliary loss...
            #loss_aux, size_aux = self.loss_function(label_id, vanilla_set_output, self.loss_object, self.padding_id)
            #loss_aux_ = loss_aux / size_aux
            #loss_ = loss_ + (loss_aux_*lambda_)

        gradients = tape.gradient(loss_, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        #gradients2 = tape.gradient(loss_aux_*lambda_, self.trainable_variables)
        #self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss, size

    @tf.function
    def _distributed_val_step(self, input_string, input_id, label_id, aoint_indices, sample_weights, num_aux_tokens):
        loss_dec, size_dec = None, None
        if self.strategy is not None:
            loss_dec, size_dec = self.strategy.run(self.val_step, args=(
            input_string, input_id, label_id, aoint_indices, sample_weights, num_aux_tokens,))

            # The if else may be totally irrelevant.
            if self.strategy.num_replicas_in_sync > 1:
                loss_dec = tf.reduce_sum(loss_dec.values)
                size_dec = tf.reduce_sum(size_dec.values)
            else:
                loss_dec = tf.reduce_sum(loss_dec)
                size_dec = tf.reduce_sum(size_dec)
        else:
            loss_dec, size_dec = self.val_step(input_string, input_id, label_id, aoint_indices, sample_weights,
                                                 num_aux_tokens)

        return loss_dec, size_dec

    def run_validation(self, e, save_filepath_val, data, num_aux_tokens):

        start = time.time()
        epoch_loss_total = 0
        epoch_size_total = 0
        for (input_string, input_id, label_id, aoint_indices, sample_weights) in data:

            loss_dec, size_dec = self._distributed_val_step(input_string, input_id, label_id, aoint_indices,
                                                              sample_weights, num_aux_tokens)
            if size_dec == 0:
                print(f"The size is zero, skip the current batch, it will not be counted due to an error!")
                continue  # start the next batch.

            epoch_loss_total += loss_dec
            epoch_size_total += size_dec

        total_loss = epoch_loss_total / epoch_size_total

        print(f'Validation Epoch {e+1}Loss {total_loss:.4f}')
        print(f'Time taken for validation epoch {e+1}: {time.time() - start:.2f} secs\n')

        header = True if e == 0 else False
        self._save_epoch_loss_only("val", e+1, save_filepath_val, header, total_loss)

    def test_step(self, input_string, input_id, all_labels, correct_ao, aoint_indices,
                  num_aux_tokens, max_generate_len):

        mask = create_combined_mask(input_id, padding_id=self.padding_id, num_aux_tok=num_aux_tokens)

        generated_ids = self.model.generate_answer(input_string, input_id, training=False, mask=mask,
                                                   reading_strat_mc_bool=False, vanilla_set_aux_loss_bool=False,
                                                   fixed_output=True, pad_tok_id=self.padding_id,
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

    def _accuracy_helper(self, all_labels, cor_label, generated_ids):
        # all_labels and cor_label are list of strings.
        # all_labels
        # generated_ids is a list of integers.
        correct = 0
        total = 0

        all_labels_split = []
        for batch_item in all_labels:
            temp_list = re.split(r"\(\d\)", batch_item)  # batch item will be a string.
            for i in range(len(temp_list)-1, -1, -1):
                if temp_list[i].strip() == "":
                    temp_list.pop(i)
                else:
                    temp_list[i] = temp_list[i].strip()
            # print(f"temp_list: {temp_list}")
            all_labels_split.append(temp_list) # temp_list is a list where each item in the list corresponds to an answer option.

        for i, batch_item in enumerate(all_labels_split):  # batch_item will be a list of strings, each representing one answer option.
            temp1 = self.tokenizer.batch_encode(batch_item)["input_ids"]  # answer options. list of
            # print(f"cor_label \n {cor_label}")
            # temp2 = self.tokenizer.encode_single(cor_label[i]) # answer list of integers (in this case one integer).
            # assert len(temp2) == 1, f"There are more than 1 id in the list. There should only be one! \n This means" \
            #                        f" that the single item is not in the tokenizer's vocabulary or it is more than 1 " \
            #                        f"element to begin with!"
            # indice = self._intersection_ids_counter(temp1, temp2)
            # indice = self._intersection_size_only(temp1, temp2)
            # indice = self._intersection_size_only(temp1, generated_ids[i])
            indice = self._intersection_ids_counter(temp1, generated_ids[i])

            cor, tot = self._check_correct_helper(indice, cor_label[i])
            correct += cor
            total += tot
        return correct, total

    def _intersection_ids_counter(self, all_labels, answer):
        # note: this should work for both strings and integers.
        # all_labels = [[], [], []] # note: that order should be preserved.
        # answer = []
        answer_unique = set(answer)
        all_labels_unique = []
        for lst in all_labels: all_labels_unique.append(set(lst))

        label_scores = []
        for set_ in all_labels_unique:
            total_unique_union = set_.union(answer_unique)
            total_intersection = set_.intersection(answer_unique)

            label_scores.append(len(total_intersection) / len(total_unique_union))

        max_score = max(label_scores)
        max_indices = [i for i, j in enumerate(label_scores) if j == max_score]

        # note: bias towards first elment in list instead of random for consistency if a tie.
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

    def _distributed_test_step(self, input_string, input_id, all_labels, correct_ao, aoint_indices,
                               num_aux_tokens, max_generate_len):
        correct, total = None, None
        if self.strategy is not None:
            correct, total = self.strategy.run(self.test_step, args=(
            input_string, input_id, all_labels, correct_ao, aoint_indices, num_aux_tokens, max_generate_len,))

            # The if else may be totally irrelevant.
            if self.strategy.num_replicas_in_sync > 1:
                correct = tf.reduce_sum(correct.values)
                total = tf.reduce_sum(total.values)
            else:
                correct = tf.reduce_sum(correct)
                total = tf.reduce_sum(total)
        else:
            correct, total = self.test_step(input_string, input_id, all_labels, correct_ao, aoint_indices,
                                                num_aux_tokens, max_generate_len)

        return correct, total

    def generate_answer_test(self, e, save_filepath, data, num_aux_tokens,
                             max_generate_len=100, attn_strat="full_attn", filename_prefix="test"):  # greedy decoding.
        # print(f"This generate_answer_test function gets the loss and accuracy on the data. \n"
        #      f"The data's format should be (input_string (w/out teacher forcing), input_id (w/out teacher forcing), "
        #      f"correct_ans(e.g. A or B...) in id format)")
        # input should be (input_string(no correct ans), input_id(no correct ans), ans_options, correct_answer, )
        start = time.time()

        correct_samples = 0  # get the number of correct answers to questions.
        total_samples = 0  # the total number of correct questions.
        for (input_string, input_id, all_labels, correct_ao, aoint_indices) in data:  # note: in all labels

            correct, total = self._distributed_test_step(input_string, input_id, all_labels, correct_ao, aoint_indices,
                                                         num_aux_tokens, max_generate_len)
            correct_samples += correct
            total_samples += total

        total_accuracy = correct_samples / total_samples
        print(f'Test Accuracy {total_accuracy} correct: {correct_samples} total: {total_samples}')
        print(f'Time taken: {time.time() - start:.2f} secs\n')

        header = True if e == 0 else False
        #self._save_epoch_results_nm("test", e+1, save_filepath, header, None, total_accuracy)
        #self._save_test_results_nm_race(filename_suffix, save_filepath, total_accuracy, correct_samples, total_samples)
        self._save_epoch_accuracy_only(filename_prefix, e+1, save_filepath, header, total_accuracy, correct_samples, total_samples)

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