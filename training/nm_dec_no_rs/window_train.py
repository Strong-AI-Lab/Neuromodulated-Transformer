'''
File name: window_train.py
Author: Kobe Knowles
Date created: 24/07/21
Data last modified: 27/07/21
Python Version: 3.6
Tensorflow version: 2
'''

import sys
sys.path.append("..")

import tensorflow as tf
import numpy as np

from training.parent_train import *

class SlidingWindowTrain(ParentTrainNL):
    '''
    Class: SlidingWindowTrain
    Description: Subclass of ParentTrainNL which implements a sliding window version of language modelling.
    Input:
        (note: tensorflow models/losses and optimizers are the only that are supported,
            however the only tokenizers that are supported are from huggingface) todo update this later with custom tokenizer parent class.
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
                 recent_to_keep=5, load_recent=False, best_to_keep=5, load_best=False, window_size_train=32,
                 window_size_val=32, load_specific_path=''):
        '''

        '''
        super(SlidingWindowTrain, self).__init__(model, optimizer, loss_object, loss_function, tokenizer,
                 checkpoint_path_recent, checkpoint_path_best, strategy, pad_token,
                 recent_to_keep, load_recent, best_to_keep, load_best, load_specific_path)

        self.window_size_train = window_size_train
        self.window_size_val = window_size_val

    def train_step(self, tar_inp, tar_real, nm_inp, num_aux_tokens, isStart):

        nm_mask = create_combined_mask(nm_inp, self.padding_id, num_aux_tokens)
        dec_mask = create_combined_mask(tar_inp, self.padding_id)

        #loss, size = 0, 0
        with tf.GradientTape() as tape:
            # TODO: reminder to remove padding_id in below call as it isn't needed? (also num_aux tokens?)
            predictions, _, _ = self.model(tar_inp, nm_inp, training=True, padding_id=self.padding_id,
                                           num_aux_tok=num_aux_tokens, nm_mask=nm_mask, dec_mask=dec_mask) # ret (output, attention weights, nm_output)
            loss, size = self.loss_function(tar_real, predictions, self.loss_object, self.padding_id,
                                            self.window_size_train, isStart, domask2=False)

            loss_ = loss/size

        gradients = tape.gradient(loss_, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss, size

    @tf.function
    def _distributed_train_step(self, tar_inp, tar_real, nm_inp, num_aux_tokens, isStart):
        if self.strategy is not None:
            loss, size = self.strategy.run(self.train_step, args=(tar_inp, tar_real, nm_inp, num_aux_tokens, isStart,))

            # The if else may be totally irrelevant.
            if self.strategy.num_replicas_in_sync > 1:
                loss = tf.reduce_sum(loss.values)
                size = tf.reduce_sum(size.values)
            else:
                # todo: test below, nan values shouldn't be an issue.
                loss = tf.reduce_sum(loss)
                size = tf.reduce_sum(size)
        else:
            loss, size = self.train_step(tar_inp, tar_real, nm_inp, num_aux_tokens, isStart)

        return loss, size

    def train_iteration(self, epoch_start, epoch_end, save_filepath_train, save_filepath_val, data_dict, num_aux_tokens):
        iteration_counter = 0
        for e in range(epoch_start, epoch_end):
            start = time.time()

            batch = 0
            epoch_loss = 0  # sum up all of the losses
            epoch_size = 0  # then divide by the total number of losses.
            for (tar_inp, tar_real, nm_inp, isStart) in data_dict["train"]:

                iteration_counter += 1  # one iteration is defined to be one batch.

                loss, size = self._distributed_train_step(tar_inp, tar_real, nm_inp, num_aux_tokens, isStart)
                if size == 0:
                    print(f"The size is zero, skip the current batch, it will not be counted due to an error!")
                    continue  # start the next batch.

                epoch_loss += loss
                epoch_size += size

                # loss for the current batch.
                loss_ = loss / size
                loss_ = tf.cast(loss_, dtype=tf.dtypes.float32)
                dict_ = self.perplexity_bpc_function(loss_)

                if "perplexity" in dict_.keys():
                    perp = dict_["perplexity"]
                if "bpc" in dict_.keys():
                    bpc = dict_["bpc"]

                if iteration_counter % 25 == 0:
                    print(f'Iteration {iteration_counter} Epoch {e+1} Batch {batch} Loss {loss_:.4f}'
                          f' Perplexity {perp:.4f} Bits Per Word (bpw) {bpc:.4f}')
                batch += 1

                # every n iterations run through
                if iteration_counter % 250 == 0:
                    if "val" in data_dict.keys():
                        print(f"Running through the validation set now!")
                        self._run_validation(e, save_filepath_val, data_dict["val"],
                                             num_aux_tokens, iteration_counter)

                if (iteration_counter) % 250 == 0:
                    ckpt_save_path = self.ckpt_manager.save()
                    print(f'Saving checkpoint for iteration {iteration_counter} at {ckpt_save_path}')

                header = True if iteration_counter == 1 else False
                self._save_iteration_results("train", iteration_counter, save_filepath_train, header, loss_, perp, bpc)

            total_loss = epoch_loss / epoch_size  # this is the loss of all words divided by the number of words (while eliminating the effect of the padding tokens)
            dict__ = self.perplexity_bpc_function(total_loss)
            if "perplexity" in dict__.keys():
                epoch_perp = dict__["perplexity"]
            if "bpc" in dict_.keys():
                epoch_bpc = dict__["bpc"]
            print(
                f'Epoch {e + 1} Loss {total_loss:.4f} Perplexity {epoch_perp:.4f} Bits Per Word (bpw) {epoch_bpc:.4f}')
            print(f'Time taken for epoch {e + 1}: {time.time() - start:.2f} secs\n')
            # TODO: change from %1 later, just testing to see if works initially.

            header = True if e == 0 else False
            self._save_epoch_results("train", e + 1, save_filepath_train, header, total_loss, epoch_perp, epoch_bpc)

            if "val" in data_dict.keys():
                print(f"Running through the validation set now!")
                self._run_validation(e, save_filepath_val, data_dict["val"], num_aux_tokens)  # note e+1 is not correct. # TODO

    def _run_validation(self, e, save_filepath, validation, num_aux_tokens, iteration_counter=None):
        start = time.time()
        #batch = 0
        # Still calculate below to get the perplexity and bits per Word scores.
        epoch_loss = 0  # sum up all of the losses
        epoch_size = 0  # then divide by the total number of losses.
        for (tar_inp, tar_real, nm_inp, isStart) in validation:
            loss, size = self._distributed_val_step(tar_inp, tar_real, nm_inp, num_aux_tokens, isStart)
            epoch_loss += loss
            epoch_size += size

        total_loss = epoch_loss / epoch_size  # this is the loss of all words divided by the number of words (while eliminating the effect of the padding tokens)
        dict__ = self.perplexity_bpc_function(total_loss)
        if "perplexity" in dict__.keys():
            epoch_perp = dict__["perplexity"]
        if "bpc" in dict__.keys():
            epoch_bpc = dict__["bpc"]
        print(
            f'Epoch {e+1} Val Loss {total_loss:.4f} Val Perplexity {epoch_perp:.4f} Val Bits Per Word (bpw) {epoch_bpc:.4f}')
        print(f'Time taken for one epoch (val) {e+1}: {time.time() - start:.2f} secs\n')

        if iteration_counter is None:
            header = True if e == 0 else False
            self._save_epoch_results("val", e+1, save_filepath, header, total_loss, epoch_perp, epoch_bpc)
        else:
            header = True if iteration_counter == 1 else False # todo this is broken (in the parent class fix this)
            # note: here the iteration refers to the iteration in the training loop.
            self._save_iteration_results("val", iteration_counter, save_filepath, header, total_loss, epoch_perp, epoch_bpc)

    def val_step(self, tar_inp, tar_real, nm_inp, num_aux_tokens, isStart):

        loss, size = 0, 0
        predictions, _, _ = self.model(tar_inp, nm_inp, training=False, padding_id=self.padding_id,
                                       num_aux_tok=num_aux_tokens)  # ret (output, attention weights)
        loss, size = self.loss_function(tar_real, predictions, self.loss_object, self.padding_id,
                                        self.window_size_val, isStart, domask2=True)

        return loss, size

    @tf.function
    def _distributed_val_step(self, tar_inp, tar_real, nm_inp, num_aux_tokens, isStart):
        if self.strategy is not None:
            loss, size = self.strategy.run(self.val_step, args=(tar_inp, tar_real, nm_inp, num_aux_tokens, isStart))

            # The if else may be totally irrelevant.
            if self.strategy.num_replicas_in_sync > 1:
                loss = tf.reduce_sum(loss.values)
                size = tf.reduce_sum(size.values)
            else:
                # todo: test below, nan values shouldn't be an issue.
                loss = tf.reduce_sum(loss)
                size = tf.reduce_sum(size)
        else:
            loss, size = self.val_step(tar_inp, tar_real, nm_inp, num_aux_tokens, isStart)

        return loss, size

    def run_no_train(self, data, num_aux_tokens):
        '''
        Function: run_no_grad
        Description: Runs and returns the resulting perplexity of an iterable dataset.
        Input:
            data: Iterable data to iterate through to get results. Has to be in the correct format.
        Return:
            total_loss: (float)
            perplexity: (float)
            bpc: (float)
        '''
        epoch_loss = 0
        epoch_size = 0
        for (tar_inp, tar_real, nm_inp, isStart) in data:

            loss, size = self._distributed_val_step(tar_inp, tar_real, nm_inp, num_aux_tokens, isStart) # use _distributed_val_step becuase it does exactly what is needed already.
            epoch_loss += loss
            epoch_size += size

            #loss_ = loss / size
            #loss_ = tf.cast(loss_, dtype=tf.dtypes.float32)
            #dict_ = self.perplexity_bpc_function(loss_)

        total_loss = epoch_loss / epoch_size  # this is the loss of all words divided by the number of words (while eliminating the effect of the padding tokens)
        dict__ = self.perplexity_bpc_function(total_loss)
        if "perplexity" in dict__.keys():
            perplexity = dict__["perplexity"]
        if "bpc" in dict__.keys():
            bpc = dict__["bpc"]

        return {"loss":total_loss, "perplexity(e^loss)":perplexity,
                "bits per character":bpc}

def loss_function_window_size(real, pred, loss_object, padding_id, window_size, isStart, domask2=False):
    # TODO add suport to run original versus the sliding window version. i.e. whether or not to calculate/use mask2...
    mask = tf.math.logical_not(tf.math.equal(real, padding_id)) # padding mask

    loss_ = loss_object(real, pred)

    if domask2:
        mask2 = None
        for i in range(mask.shape[0]): # i.e. iterate through each batch.
            if isStart[i,0] == 1: # this means the first max_seq_len tokens of an article/document.
                if mask2 is None:
                    mask2 = tf.ones((1, mask.shape[1]))
                else:
                    mask2 = tf.concat([mask2, tf.ones((1, mask.shape[1]))], axis=0)
            else:
                if mask2 is None:
                    mask2 = tf.concat([tf.zeros((1,mask.shape[1]-window_size)), tf.ones((1,window_size))], axis=-1)
                else:
                    z = tf.concat([tf.zeros((1, mask.shape[1]-window_size)), tf.ones((1, window_size))], axis=-1)
                    #print(f"z.shape: {z.shape}")
                    mask2 = tf.concat([mask2, z], axis=0)

        mask = tf.cast(mask, dtype=loss_.dtype)
        # mask2 = tf.stop_gradient(tf.cast(mask2, dtype=loss_.dtype))
        if mask2 is not None:
            mask = mask * mask2  # any 0 in both masks will remain a zero, otherwise the item will be a one.
        else:
            print("mask2 is None when it shouldn't be --> running withoug mask2!\n"
                  "If during training on multiple GPUs then this handles 0 batch size inputs without raising an error")

        # if error try stop gradient?
    else:
        mask = tf.cast(mask, dtype=loss_.dtype)

    loss_ *= mask

    #return tf.reduce_sum(loss_, axis=1) / tf.reduce_sum(mask, axis=1)
    #return tf.reduce_sum(loss_) / tf.reduce_sum(mask)
    return tf.reduce_sum(loss_), tf.reduce_sum(mask)
