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
                 recent_to_keep=5, load_recent=False, best_to_keep=5, load_best=False, window_size=32):
        '''

        '''
        super(SlidingWindowTrain, self).__init__(model, optimizer, loss_object, loss_function, tokenizer,
                 checkpoint_path_recent, checkpoint_path_best, strategy, pad_token="<pad>",
                 recent_to_keep=5, load_recent=False, best_to_keep=5, load_best=False)

        self.window_size = window_size

    def train_iteration(self, epoch_start, epoch_end, save_filepath_train, save_filepath_val, data_dict,
                        num_aux_tokens):
        iteration_counter = 0
        for e in range(epoch_start, epoch_end):
            start = time.time()

            batch = 0
            epoch_loss = 0  # sum up all of the losses
            epoch_size = 0  # then divide by the total number of losses.
            for (tar_inp, tar_real, nm_inp, isStart) in data_dict["train"]:

                iteration_counter += 1  # one iteration is defined to be one batch.

                loss, size = self._distributed_train_step(tar_inp, tar_real, nm_inp, num_aux_tokens) # TODO
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
                    print(f'Iteration {iteration_counter} Epoch {e + 1} Batch {batch} Loss {loss_:.4f}'
                          f' Perplexity {perp:.4f} Bits Per Word (bpw) {bpc:.4f}')
                batch += 1

                # every 1000 iterations run through
                if iteration_counter % 250 == 0:
                    if "val" in data_dict.keys():
                        print(f"Running through the validation set now!")
                        self._run_validation(e, save_filepath_val, data_dict["val"],
                                             num_aux_tokens, iteration_counter)  # note e+1 is not correct. # TODO

                if (iteration_counter) % 500 == 0:
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

def loss_function_window_size(real, pred, loss_object, padding_id, isStart):

    mask = tf.math.logical_not(tf.math.equal(real, padding_id)) # padding mask

    isStart = tf.stop_gradient(isStart) # stop the gradient as we don't want gradient flowing through here.

    mask2 = None
    for i in range(mask.shape[0]): # i.e. iterate through each batch.
        if isStart[i,0] == 1: # this means the first max_seq_len tokens of an article/document.
            if mask2 is None:
                mask2 = tf.ones((1, mask.shape[1]))
            else:
                mask2 = tf.concat([mask2, tf.ones((1, mask.shape[1]))], axis=0)
        else:
            if mask2 is None:
                mask2 = tf.concat([tf.zeros((1,mask.shape[1]-self.window_size)), tf.ones((1,window_size))], axis=-1)
            else:
                z = tf.concat([tf.zeros((1, mask.shape[1] - window_size)), tf.ones((1, window_size))], axis=-1)
                #print(f"z.shape: {z.shape}")
                mask2 = tf.concat([mask2, z], axis=0)

    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)

    mask2 = tf.cast(mask2, dtype=loss_.dtype)
    mask = mask * mask2 # any 0 in both masks will remain a zero, otherwise the item will be a one.

    loss_ *= mask

    #return tf.reduce_sum(loss_, axis=1) / tf.reduce_sum(mask, axis=1)
    #return tf.reduce_sum(loss_) / tf.reduce_sum(mask)
    return tf.reduce_sum(loss_), tf.reduce_sum(mask)
