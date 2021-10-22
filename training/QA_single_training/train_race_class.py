'''
File name: pretrain_train.py
Author: Kobe Knowles
Date created: 24/08/21
Data last modified: 3/09/21
Python Version: 3.8
Tensorflow version: 2.5
'''

import sys
sys.path.append("..")

import tensorflow as tf
import numpy as np
import re

from training.parent_train import *

class NMTransformerEncDecTrain(ParentTrainNL):
    '''
    Class: NMTransformerEncDecTrain
    Description: Subclass of ParentTrainNL which implements a version that supports Encoder and Decoder modes
        and pre_training for C4. \n
    Input:
        (note: tensorflow models/losses and optimizers are the only that are supported,
            however the only tokenizers that are supported are from huggingface)
        model: The previously initialized model.
        optimizer: The optimizer to use during training.
        loss_object: e.g. SparseCategoricalCrossentropy
        loss_function: Uses the loss_object to get the loss.
        tokenizer: huggingface transformer.
        checkpoint_path_recent: (string) path to the recent checkpoint folder.
        checkpoint_path_best: (string) path to best checkpoint folder.
        strategy: (tf.distribute...) Strategy for splitting the training across multiple gpus.
    '''

    def __init__(self, model, optimizer, loss_object, loss_function, tokenizer,
                 checkpoint_path_recent, checkpoint_path_best, strategy, pad_token="<pad>",
                 recent_to_keep=5, load_recent=False, best_to_keep=5, load_best=False, load_specific_path='',
                 enc_tok_id=None, dec_tok_id=None, end_tok_id=None):

        super(NMTransformerEncDecTrain, self).__init__(model, optimizer, loss_object, None, tokenizer,
                 checkpoint_path_recent, checkpoint_path_best, strategy, pad_token,
                 recent_to_keep, load_recent, best_to_keep, load_best, load_specific_path)

        assert enc_tok_id is not None and dec_tok_id is not None, f"It is required that the encoder and decoder token id is passed as input during initialization!"
        self.enc_tok_id = enc_tok_id
        self.dec_tok_id = dec_tok_id
        self.end_tok_id = end_tok_id
        self.const_end_tok_id = tf.constant([self.end_tok_id], dtype=tf.dtypes.int64)

        self.loss_function = loss_function

    def train_step(self, inp_id, label, nm_inp_id, num_aux_tokens):

        # create the masks depending on if the NMTransformer is to be in encoder or decoder mode.
        mode_list = list() # holds which mode each batch item is in.
        nm_mask = None
        dec_mask = None
        for b in range(nm_inp_id.shape[0]): # iterate through each batch.
            if nm_inp_id[b,0] == self.dec_tok_id and num_aux_tokens > 0:
                nm_mask_ = create_combined_mask(tf.expand_dims(nm_inp_id[b,:], axis=0), self.padding_id, num_aux_tokens) # [batch_size, 1, seq_len, seq_len]
                dec_mask_ = create_combined_mask(tf.expand_dims(inp_id[b,:], axis=0), self.padding_id)
                mode_list.append("dec")
            elif nm_inp_id[b,0] == self.enc_tok_id and num_aux_tokens > 0:
                nm_mask_ = create_padding_mask(tf.expand_dims(nm_inp_id[b,:], axis=0), self.padding_id) # [batch, 1, 1, seq_len] TODO repeat dimension here...
                nm_mask_ = tf.repeat(nm_mask_, [nm_mask_.shape[-1]], axis=2) # [batch, 1, seq_len, seq_len]
                dec_mask_ = create_padding_mask(tf.expand_dims(inp_id[b,:], axis=0), self.padding_id)
                dec_mask_ = tf.repeat(dec_mask_, [dec_mask_.shape[-1]], axis=2) # [batch, 1, seq_len, seq_len]
                mode_list.append("enc")
            elif num_aux_tokens <= 0: # decoder masking by default.
                nm_mask_ = create_combined_mask(tf.expand_dims(nm_inp_id[b,:], axis=0), self.padding_id, num_aux_tokens)
                dec_mask_ = create_combined_mask(tf.expand_dims(inp_id[b,:], axis=0), self.padding_id)
                mode_list.append("dec")
            else:
                #raise Exception(f"Error: invalid value for auxiliary token at postion 0. \n It should represent one of the "
                #                 f"encoder or decoder tokens but doesn't!")
                nm_mask_ = tf.ones((nm_inp_id.shape[0], 1, nm_inp_id.shape[1], nm_inp_id.shape[1])) # pad everyting otherwise.
                dec_mask_ = tf.ones((inp_id.shape[0], 1, inp_id.shape[1], inp_id.shape[1]))
                mode_list.append("none")
            if nm_mask is None: nm_mask = nm_mask_
            else: nm_mask = tf.concat([nm_mask, nm_mask_], axis=0)
            if dec_mask is None: dec_mask = dec_mask_
            else: dec_mask = tf.concat([dec_mask, dec_mask_], axis=0)

        #loss_enc, size_enc, loss_dec, size_dec = 0, 0, 0, 0
        loss, size = 0, 0
        #correct, total = 0, 0
        with tf.GradientTape() as tape:
            predictions, _, _, _, _ = self.model(inp_id, nm_inp_id, training=True, nm_mask=nm_mask, dec_mask=dec_mask) # ret (output, attention weights, nm_output)
            #predictions.shape == (batch_size, seq_len, target_vocab_size)
            #pred = self._end_tok_pred_helper(predictions, inp_id) # (batch_size, target_vocab_size)
            #correct, total = self._get_accuracy(predictions, label)
            #print(f"Correct: {correct} \nTotal: {total}")
            # label.shape == (batch_size)
            # pred.shape == (batch_size, target_vocab_size)
            loss, size = self.loss_function(label, predictions, self.loss_object, self.padding_id)
            loss_ = loss / size
        gradients = tape.gradient(loss_, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        #return loss_enc, size_enc, loss_dec, size_dec
        return loss, size#, correct, total

    '''
    def _end_tok_pred_helper(self, predictions, inp_id):
        
        Function: _end_tok_pred_helper \n
        Description: Function that converts the predictions to only the end token, and eliminates the rest.
             If no end or more than one end token it raises an error. \n
        Input:
            predictions: (tf.Tensor; [batch_size, (max_)seq_len, target_vocab_size])
            inp_id: (tf.Tensor; [batch_size, (max_)seq_len])
        
        end_tok_indices = [] # each element represents the end token at an individual sample. (these are in the correct order, and only one per sample)
        for i in range(inp_id.shape[0]):
            end_tok_counter = 0
            for j in range(inp_id.shape[1]):
                #print(f"inp_id single element: {inp_id[i,j]}")
                #print(f"constant end tok id: {tf.constant([self.end_tok_id], dtype=tf.dtypes.int64)}")
                #if tf.math.equal(tf.expand_dims(inp_id[i,j],axis=0), self.const_end_tok_id):
                if inp_id[i,j] == self.end_tok_id:
                    end_tok_counter += 1
                    print(f"REACH {i} {j}")
                    end_tok_indices.append(j)
                #if end_tok_counter > 2: raise Exception(f"Too many end tokens in the input, there should be only one!")
            #if end_tok_counter == 0: raise Exception(f"There should be at least one end token in each sample, got 0!")
        assert len(end_tok_indices) == inp_id.shape[0], f"The number of indices should equal the batch size (# of samples)"
        # i.e. it equals the batch size...

        pred = None
        for i in range(predictions.shape[0]):
            if pred is None: pred = tf.expand_dims(predictions[i, end_tok_indices[i],:], axis=0)
            else: pred = tf.concat([pred, tf.expand_dims(predictions[i, end_tok_indices[i],:], axis=0)], axis=0)

        assert len(pred.shape) == 2, f"The length of pred.shape should be 2, got {len(pred.shape)}"
        return pred
    '''

    def _get_accuracy(self, predictions, label):
        '''
        Function: get_accuracy \n
        Description: Helper function that gets the accuracy of the predictions with the corrects labels. \n
        Input:
            predictions: (tf.Tensor; [batch_size, (max_)seq_len, target_vocab_size]
            label: (tf.Tensor; [batch_size, (max_)seq_len]
        Return:
            correct: (int) The total number of correct predictions. \n
            total: (int) The total number of samples a prediction is made for.
        '''
        pred = tf.argmax(predictions, axis=2) # (batch_size, (max_)seq_len)
        #total = predictions.shape[0]
        correct = 0
        total = 0
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                if label[i,j] != self.padding_id: total += 1
                else: continue
                if pred[i,j] == label[i,j]:
                    correct += 1
        return correct, total

    def _get_accuracy_test(self):
        pass

    @tf.function
    def _distributed_train_step(self, inp_id, tar_id, nm_inp_id, num_aux_tokens):
        if self.strategy is not None:
            loss, size = self.strategy.run(self.train_step, args=(inp_id, tar_id, nm_inp_id, num_aux_tokens,))

            # The if else may be totally irrelevant.
            if self.strategy.num_replicas_in_sync > 1:
                loss = tf.reduce_sum(loss.values)
                size = tf.reduce_sum(size.values)
                #correct = tf.reduce_sum(correct.values)
                #total = tf.reduce_sum(total.values)
            else:
                loss = tf.reduce_sum(loss)
                size = tf.reduce_sum(size)
                #correct = tf.reduce_sum(correct)
                #total = tf.reduce_sum(total)
        else:
            loss, size = self.train_step(inp_id, tar_id, nm_inp_id, num_aux_tokens)

        return loss, size#, correct, total

    def train_iteration(self, epoch_start, epoch_end, iteration_counter, save_filepath_train, save_filepath_val,
                        data_dict, num_aux_tokens, save_end_epoch=True, print_every_iterations=100,
                        save_every_iterations=5000):
        iteration_counter = iteration_counter
        for e in range(epoch_start, epoch_end):
            start = time.time()

            batch = 0
            epoch_loss_total = 0  # sum up all of the losses
            epoch_size_total = 0  # then divide by the total number of losses.
            correct_samples = 0
            total_samples = 0
            for (inp_str, inp_id, label, nm_inp_id) in data_dict["train"]:

                iteration_counter += 1  # one iteration is defined to be one batch.

                loss, size = self._distributed_train_step(inp_id, label, nm_inp_id, num_aux_tokens)
                if size == 0:
                    print(f"The size is zero, skip the current batch, it will not be counted due to an error!")
                    continue  # start the next batch.

                epoch_loss_total += loss
                epoch_size_total += size
                #correct_samples += correct
                #total_samples += total

                # loss for the current batch.
                #loss_ = (loss_enc+loss_dec) / (size_enc+size_dec)
                loss_ = loss / size
                loss_ = tf.cast(loss_, dtype=tf.dtypes.float32)
                # (decoder).
                #if total == 0:
                #    accuracy_ = None
                #else:
                #    accuracy_ = correct/total
                accuracy_ = None

                if iteration_counter % print_every_iterations == 0:
                    print(f'Iteration {iteration_counter} Epoch {e+1} Batch {batch+1} Loss {loss_:.4f} Accuracy {accuracy_}')
                batch += 1

                if (iteration_counter) % save_every_iterations == 0:
                    ckpt_save_path = self.ckpt_manager.save()
                    print(f'Saving checkpoint for iteration {iteration_counter} at {ckpt_save_path}')

                header = True if iteration_counter == 1 else False
                self._save_iteration_results_nm("train", iteration_counter, save_filepath_train, header, loss_, accuracy_)

            total_loss = epoch_loss_total / epoch_size_total  # this is the loss of all words divided by the number of words (while eliminating the effect of the padding tokens)
            if total_samples == 0:
                total_accuracy = None
            else:
                total_accuracy = correct_samples/total_samples
            print(f'Epoch {e+1} Loss {total_loss:.4f} Accuracy {total_accuracy}')
            print(f'Time taken for epoch {e+1}: {time.time() - start:.2f} secs\n')

            header = True if e == 0 else False
            self._save_epoch_results_nm("train", e+1, save_filepath_train, header, total_loss, total_accuracy)

            if save_end_epoch:
                ckpt_save_path = self.ckpt_manager.save()
                print(f'Saving checkpoint for epoch {e+1} at {ckpt_save_path}')

            if "val" in data_dict.keys():
                print(f"Running through the validation set now!")
                self._run_validation(e, save_filepath_val, data_dict["val"], num_aux_tokens)  # note e+1 is not correct.

    def val_step(self, inp_id, label, nm_inp_id, num_aux_tokens):

        # create the masks depending on if the NMTransformer is to be in encoder or decoder mode.
        mode_list = list() # holds which mode each batch item is in.
        nm_mask = None
        dec_mask = None
        for b in range(nm_inp_id.shape[0]): # iterate through each batch.
            if nm_inp_id[b,0] == self.dec_tok_id and num_aux_tokens > 0:
                nm_mask_ = create_combined_mask(tf.expand_dims(nm_inp_id[b,:], axis=0), self.padding_id, num_aux_tokens) # [batch_size, 1, seq_len, seq_len]
                dec_mask_ = create_combined_mask(tf.expand_dims(inp_id[b,:], axis=0), self.padding_id)
                mode_list.append("dec")
            elif nm_inp_id[b,0] == self.enc_tok_id and num_aux_tokens > 0:
                nm_mask_ = create_padding_mask(tf.expand_dims(nm_inp_id[b,:], axis=0), self.padding_id) # [batch, 1, 1, seq_len] TODO repeat dimension here...
                nm_mask_ = tf.repeat(nm_mask_, [nm_mask_.shape[-1]], axis=2) # [batch, 1, seq_len, seq_len]
                dec_mask_ = create_padding_mask(tf.expand_dims(inp_id[b,:], axis=0), self.padding_id)
                dec_mask_ = tf.repeat(dec_mask_, [dec_mask_.shape[-1]], axis=2) # [batch, 1, seq_len, seq_len]
                mode_list.append("enc")
            elif num_aux_tokens <= 0: # decoder masking by default.
                nm_mask_ = create_combined_mask(tf.expand_dims(nm_inp_id[b,:], axis=0), self.padding_id, num_aux_tokens)
                dec_mask_ = create_combined_mask(tf.expand_dims(inp_id[b,:], axis=0), self.padding_id)
                mode_list.append("dec")
            else:
                #raise Exception(f"Error: invalid value for auxiliary token at postion 0. \n It should represent one of the "
                #                 f"encoder or decoder tokens but doesn't!")
                nm_mask_ = tf.ones((nm_inp_id.shape[0], 1, nm_inp_id.shape[1], nm_inp_id.shape[1])) # pad everyting otherwise.
                dec_mask_ = tf.ones((inp_id.shape[0], 1, inp_id.shape[1], inp_id.shape[1]))
                mode_list.append("none")
            if nm_mask is None: nm_mask = nm_mask_
            else: nm_mask = tf.concat([nm_mask, nm_mask_], axis=0)
            if dec_mask is None: dec_mask = dec_mask_
            else: dec_mask = tf.concat([dec_mask, dec_mask_], axis=0)

        #loss_enc, size_enc, loss_dec, size_dec = 0, 0, 0, 0
        loss, size = 0, 0
        #correct, total = 0, 0
        with tf.GradientTape() as tape:
            predictions, _, _, _, _ = self.model(inp_id, nm_inp_id, training=False, nm_mask=nm_mask, dec_mask=dec_mask) # ret (output, attention weights, nm_output)
            #predictions.shape == (batch_size, seq_len, target_vocab_size)

            #pred = self._end_tok_pred_helper(predictions, inp_id) # (batch_size, target_vocab_size)
            #correct, total = self._get_accuracy(predictions, label)
            # label.shape == (batch_size)
            # pred.shape == (batch_size, target_vocab_size)
            loss, size = self.loss_function(label, predictions, self.loss_object, self.padding_id)

        #return loss_enc, size_enc, loss_dec, size_dec
        return loss, size#, #correct, total

    @tf.function
    def _distributed_val_step(self, inp_id, tar_id, nm_inp_id, num_aux_tokens):
        if self.strategy is not None:
            loss, size = self.strategy.run(self.val_step, args=(inp_id, tar_id, nm_inp_id, num_aux_tokens,))

            # The if else may be totally irrelevant.
            if self.strategy.num_replicas_in_sync > 1:
                loss = tf.reduce_sum(loss.values)
                size = tf.reduce_sum(size.values)
                #correct = tf.reduce_sum(correct.values)
                #total = tf.reduce_sum(total.values)
            else:
                loss = tf.reduce_sum(loss)
                size = tf.reduce_sum(size)
                #correct = tf.reduce_sum(correct)
                #total = tf.reduce_sum(total)
        else:
            loss, size = self.val_step(inp_id, tar_id, nm_inp_id, num_aux_tokens)

        return loss, size

    def _run_validation(self, e, save_filepath, validation, num_aux_tokens, iteration_counter=None):
        start = time.time()

        epoch_loss_total = 0  # sum up all of the losses
        epoch_size_total = 0  # then divide by the total number of losses.
        correct_samples = 0
        total_samples = 0
        for (inp_str, inp_id, label, nm_inp_id) in validation:
            loss, size = self._distributed_val_step(inp_id, label, nm_inp_id, num_aux_tokens)
            epoch_loss_total += loss
            epoch_size_total += size
            #correct_samples += correct
            #total_samples += total

        total_loss = epoch_loss_total / epoch_size_total  # this is the loss of all words divided by the number of words (while eliminating the effect of the padding tokens)
        if total_samples == 0:
            total_accuracy = None
        else:
            total_accuracy = correct_samples/total_samples
        print(f'Epoch {e+1} Val Loss {total_loss:.4f} Val Accuracy {total_accuracy}')
        print(f'Time taken for one epoch (val) {e+1}: {time.time() - start:.2f} secs\n')

        if iteration_counter is None:
            header = True if e == 0 else False
            self._save_epoch_results_nm("val", e+1, save_filepath, header, total_loss, total_accuracy)
        else:
            header = True if iteration_counter == 1 else False  # todo this is broken.
            # note: here the iteration refers to the iteration in the training loop.
            self._save_iteration_results_nm("val", iteration_counter, save_filepath, header, total_loss, total_accuracy)

    def _save_epoch_results_nm(self, type_, epoch, save_filepath, header, total_loss, accuracy):
        assert isinstance(type_, str) and isinstance(save_filepath, str)
        file = save_filepath+type_+"epoch"+".txt"
        with open(file, "a") as f:
            if header: f.write("Epoch Loss Accuracy \n")
            f.write(f"{epoch} {total_loss} {accuracy} \n")

    def _save_iteration_results_nm(self, type_, iteration, save_filepath, header, total_loss, accuracy): # TODO accuracy support...
        assert isinstance(type_, str) and isinstance(save_filepath, str)
        file = save_filepath+type_+"iteration"+".txt"
        with open(file, "a") as f:
            if header: f.write("Iteration Loss Accuracy \n")
            f.write(f"{iteration} {total_loss} {accuracy} \n")

    def generate_answer_test(self, e, save_filepath, data, num_aux_tokens, max_generate_len=100): # greedy decoding.
        #print(f"This generate_answer_test function gets the loss and accuracy on the data. \n"
        #      f"The data's format should be (input_string (w/out teacher forcing), input_id (w/out teacher forcing), "
        #      f"correct_ans(e.g. A or B...) in id format)")
        # input should be (input_string(no correct ans), input_id(no correct ans), ans_options, correct_answer, )
        start = time.time()

        correct_samples = 0 # get the number of correct answers to questions.
        total_samples = 0 # the total number of correct questions.
        for (inp_str, inp_id, all_labels, cor_label, nm_inp_id) in data: # note: in all labels
            #loss, size = self._distributed_val_step(inp_id, label, nm_inp_id, num_aux_tokens)
            correct, total = self._distributed_test_step(inp_str, inp_id, all_labels, cor_label, nm_inp_id, num_aux_tokens, max_generate_len)
            correct_samples += correct
            total_samples += total

        total_accuracy = correct_samples / total_samples
        print(f'Test Accuracy {total_accuracy} correct: {correct_samples} total: {total_samples}')
        print(f'Time taken: {time.time() - start:.2f} secs\n')

        header = True if e == 0 else False
        self._save_epoch_results_nm("test", e+1, save_filepath, header, None, total_accuracy)

    def _distributed_test_step(self, inp_str, inp_id, all_labels, cor_label, nm_inp_id, num_aux_tokens, max_generate_len):
        if self.strategy is not None:
            correct, total = self.strategy.run(self.test_step, args=(inp_str, inp_id, all_labels, cor_label,
                                                                     nm_inp_id, num_aux_tokens, max_generate_len,))

            # The if else may be totally irrelevant.
            if self.strategy.num_replicas_in_sync > 1:
                #loss = tf.reduce_sum(loss.values)
                #size = tf.reduce_sum(size.values)
                correct = tf.reduce_sum(correct.values)
                total = tf.reduce_sum(total.values)
            else:
                #loss = tf.reduce_sum(loss)
                #size = tf.reduce_sum(size)
                correct = tf.reduce_sum(correct)
                total = tf.reduce_sum(total)
        else:
            correct, total = self.test_step(inp_str, inp_id, all_labels, cor_label, nm_inp_id, num_aux_tokens, max_generate_len)

        return correct, total

    def test_step(self, inp_str, inp_id, all_labels, cor_label, nm_inp_id, num_aux_tokens, max_generate_len):
        mode_list = list()  # holds which mode each batch item is in.
        nm_mask = None
        dec_mask = None
        for b in range(nm_inp_id.shape[0]):  # iterate through each batch.
            if nm_inp_id[b, 0] == self.dec_tok_id and num_aux_tokens > 0:
                nm_mask_ = create_combined_mask(tf.expand_dims(nm_inp_id[b, :], axis=0), self.padding_id,
                                                num_aux_tokens)  # [batch_size, 1, seq_len, seq_len]
                dec_mask_ = create_combined_mask(tf.expand_dims(inp_id[b, :], axis=0), self.padding_id)
                mode_list.append("dec")
            elif nm_inp_id[b, 0] == self.enc_tok_id and num_aux_tokens > 0:
                nm_mask_ = create_padding_mask(tf.expand_dims(nm_inp_id[b, :], axis=0),
                                               self.padding_id)  # [batch, 1, 1, seq_len] TODO repeat dimension here...
                nm_mask_ = tf.repeat(nm_mask_, [nm_mask_.shape[-1]], axis=2)  # [batch, 1, seq_len, seq_len]
                dec_mask_ = create_padding_mask(tf.expand_dims(inp_id[b, :], axis=0), self.padding_id)
                dec_mask_ = tf.repeat(dec_mask_, [dec_mask_.shape[-1]], axis=2)  # [batch, 1, seq_len, seq_len]
                mode_list.append("enc")
            elif num_aux_tokens <= 0:  # decoder masking by default.
                nm_mask_ = create_combined_mask(tf.expand_dims(nm_inp_id[b, :], axis=0), self.padding_id,
                                                num_aux_tokens)
                dec_mask_ = create_combined_mask(tf.expand_dims(inp_id[b, :], axis=0), self.padding_id)
                mode_list.append("dec")
            else:
                # raise Exception(f"Error: invalid value for auxiliary token at postion 0. \n It should represent one of the "
                #                 f"encoder or decoder tokens but doesn't!")
                nm_mask_ = tf.ones(
                    (nm_inp_id.shape[0], 1, nm_inp_id.shape[1], nm_inp_id.shape[1]))  # pad everyting otherwise.
                dec_mask_ = tf.ones((inp_id.shape[0], 1, inp_id.shape[1], inp_id.shape[1]))
                mode_list.append("none")
            if nm_mask is None:
                nm_mask = nm_mask_
            else:
                nm_mask = tf.concat([nm_mask, nm_mask_], axis=0)
            if dec_mask is None:
                dec_mask = dec_mask_
            else:
                dec_mask = tf.concat([dec_mask, dec_mask_], axis=0)

        correct, total = 0, 0

        generated_ids = self.model.generate_answer(inp_id, nm_inp_id, training=False, nm_mask=nm_mask, dec_mask=dec_mask,
                                                   pad_tok_id=self.padding_id, end_tok_id=self.end_tok_id,
                                                   gen_len_max=max_generate_len) # [[1,2,3,4], [324,54,54,6]]                                                                                    #  [3,45,6,7]]
        #print(f"generated answers: {self.tokenizer.batch_decode(generated_ids)}")

        convert_byte_to_string = lambda i: i.decode("utf-8")
        all_labels = [convert_byte_to_string(x) for x in all_labels.numpy().tolist()] # list of strings.
        #print(f"all labels: {all_labels}")
        cor_label = [convert_byte_to_string(x) for x in cor_label.numpy().tolist()]  # list of strings.
        #print(f"cor_label: {cor_label}")

        cor, tot = self._accuracy_helper(all_labels, cor_label, generated_ids)
        print(f"Batch accuracy: {cor/tot}")
        correct += cor
        total += tot

        # return loss_enc, size_enc, loss_dec, size_dec
        return correct, total

    def _intersection_ids_counter(self, all_labels, answer):
        # note: this should work for both strings and integers.
        #all_labels = [[], [], []] # note: that order should be preserved.
        #answer = []
        answer_unique = set(answer)
        all_labels_unique = []
        for lst in all_labels: all_labels_unique.append(set(lst))

        label_scores = []
        for set_ in all_labels_unique:

            total_unique_union = set_.union(answer_unique)
            total_intersection = set_.intersection(answer_unique)

            label_scores.append(len(total_intersection)/len(total_unique_union))

        max_score = max(label_scores)
        max_indices = [i for i, j in enumerate(label_scores) if j == max_score]

        # note: bias towards first elment in list instead of random for consistency if a tie.
        return max_indices[0] + 1 # returns an integer from 1 to len(all_labels) inclusive.

    def _intersection_size_only(self, all_labels, answer):
        answer_unique = set(answer)
        all_labels_unique = []
        for lst in all_labels: all_labels_unique.append(set(lst))

        label_scores = []
        for set_ in all_labels_unique:
            total_intersection = set_.intersection(answer_unique)
            label_scores.append(len(total_intersection))

        max_score = max(label_scores)
        max_indices = [i for i, j in enumerate(label_scores) if j == max_score]

        # note: bias towards first elment in list instead of random for consistency if a tie.
        return max_indices[0] + 1  # returns an integer from 1 to len(all_labels) inclusive.

    def _accuracy_helper(self, all_labels, cor_label, generated_ids): #TODO
        #all_labels and cor_label are list of strings.
        #all_labels
        # generated_ids is a list of integers.
        correct = 0
        total = 0

        all_labels_split = []
        for batch_item in all_labels:
            temp_list = re.split(r"\(\d\)", batch_item) # batch item will be a string.
            for i in range(len(temp_list)-1,-1,-1):
                if temp_list[i].strip() == "":
                    temp_list.pop(i)
                else: temp_list[i] = temp_list[i].strip()
            #print(f"temp_list: {temp_list}")
            all_labels_split.append(temp_list)

        for i, batch_item in enumerate(all_labels_split): # batch_item will be a list of strings, each representing one answer option.
            temp1 = self.tokenizer.batch_encode(batch_item)["input_ids"] # answer options. list of
            #print(f"cor_label \n {cor_label}")
            #temp2 = self.tokenizer.encode_single(cor_label[i]) # answer list of integers (in this case one integer).
            #assert len(temp2) == 1, f"There are more than 1 id in the list. There should only be one! \n This means" \
            #                        f" that the single item is not in the tokenizer's vocabulary or it is more than 1 " \
            #                        f"element to begin with!"
            #indice = self._intersection_ids_counter(temp1, temp2)
            #indice = self._intersection_size_only(temp1, temp2)
            #indice = self._intersection_size_only(temp1, generated_ids[i])
            indice = self._intersection_ids_counter(temp1, generated_ids[i])

            cor, tot = self._check_correct_helper(indice, cor_label[i])
            correct += cor
            total += tot
        return correct, total

    def _check_correct_helper(self, indice, answer):
        #indice will represent the answer position -- i.e. 1 means A, 2 means B, ...
        #answer will be in string format.
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
        else: raise Exception(f"Invalid index value: {indice}!")
        return correct, total



# TODO loss function for decoder (language modelling only)
def loss_function_decoder(target, prediction, loss_object, padding_id):

    mask = tf.math.logical_not(tf.math.equal(target, padding_id))  # padding mask
    loss_ = loss_object(target, prediction) # get loss from the loss_object
    mask = tf.cast(mask, dtype=loss_.dtype) # convert the mask to the correct format.
    loss_ *= mask # apply masking to positions that correspond to a <pad> token in the target.
    return tf.reduce_sum(loss_), tf.reduce_sum(mask)

# TODO loss function for the encoder (masked language modelling only) - <pad> tokens in the output (target) don't process these targets (apply a pad in these positions).
def loss_function_encoder(target, prediction, loss_object, padding_id):

    mask = tf.math.logical_not(tf.math.equal(target, padding_id))  # padding mask
    loss_ = loss_object(target, prediction) # get loss from the loss_object
    mask = tf.cast(mask, dtype=loss_.dtype) # convert the mask to the correct format.
    loss_ *= mask # apply masking to positions that correspond to a <pad> token in the target.
    return tf.reduce_sum(loss_), tf.reduce_sum(mask)

def loss_function_end_tok_only(label, prediction, loss_object, padding_id):
    # same as above loss functions, could simplify but it doesn't really matter.
    mask = tf.math.logical_not(tf.math.equal(label, padding_id))  # padding mask
    loss_ = loss_object(label, prediction)  # get loss from the loss_object
    mask = tf.cast(mask, dtype=loss_.dtype)  # convert the mask to the correct format.
    loss_ *= mask  # apply masking to positions that correspond to a <pad> token in the target.
    return tf.reduce_sum(loss_), tf.reduce_sum(mask)

def pad_loss_function(label, prediction, loss_object, padding_id):
    # same as above loss functions, could simplify but it doesn't really matter.
    mask = tf.math.logical_not(tf.math.equal(label, padding_id))  # padding mask
    loss_ = loss_object(label, prediction)  # get loss from the loss_object
    mask = tf.cast(mask, dtype=loss_.dtype)  # convert the mask to the correct format.
    loss_ *= mask  # apply masking to positions that correspond to a <pad> token in the target.
    return tf.reduce_sum(loss_), tf.reduce_sum(mask)

