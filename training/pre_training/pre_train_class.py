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

from training.parent_train import *

class NMTransformerPreTrainDec(ParentTrainNL):
    '''
    Class: NMTransformerPreTrainDec
    Description: Subclass of ParentTrainNL which implements a version that supports language modelling modes
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
                 enc_tok_id=None, dec_tok_id=None):

        super(NMTransformerPreTrainDec, self).__init__(model, optimizer, loss_object, None, tokenizer,
                 checkpoint_path_recent, checkpoint_path_best, strategy, pad_token,
                 recent_to_keep, load_recent, best_to_keep, load_best, load_specific_path)

        assert enc_tok_id is not None and dec_tok_id is not None, f"It is required that the encoder and decoder token id is passed as input during initialization!"
        self.enc_tok_id = enc_tok_id
        self.dec_tok_id = dec_tok_id

        self.loss_function = loss_function
        self.model.fixed_output_layer = self.model.output_layers["lm"] # hardcoded for C4 pre-training

    def train_step(self, inp_str, inp_id, tar_id, num_aux_tokens):

        gpt_pad_mask = create_padding_mask_gpt(inp_id, padding_id=self.padding_id) # look ahead padding is handled by huggingface gpt2 model.
        mask = create_combined_mask(inp_id, padding_id=self.padding_id, num_aux_tok=num_aux_tokens)

        lambda_ = 0.25

        loss, size = 0, 0
        with tf.GradientTape() as tape:
            vanilla_set_output, _, task_prediction, _, _ = self.model(inp_str, inp_id, training=True, mask=mask,
                                                                      gpt_pad_mask=gpt_pad_mask,
                                                                      reading_strat_mc_bool=False,
                                                                      vanilla_set_aux_loss_bool=False,
                                                                      fixed_output=True, stop_gradient=False)
            # ret (vanilla_set_output, nm_decoder_mc, task_prediction, gating_weights, attention_weights)
            #vanilla_set_output is after it has been passed through dense layer...

            loss, size = self.loss_function(tar_id, task_prediction, self.loss_object, self.padding_id)
            #loss_aux, size_aux = self.loss_function(tar_id, vanilla_set_output, self.loss_object, self.padding_id)
            loss_ = loss / size
            #loss_aux_ = loss_aux / size_aux
            #loss_ = loss_ + (loss_aux_*lambda_)

        gradients = tape.gradient(loss_, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss, size

    @tf.function
    def _distributed_train_step(self, inp_str, inp_id, tar_id, num_aux_tokens):
        loss_dec, size_dec = None, None
        if self.strategy is not None:
            loss_dec, size_dec = self.strategy.run(self.train_step, args=(inp_str, inp_id, tar_id, num_aux_tokens,))

            # The if else may be totally irrelevant.
            if self.strategy.num_replicas_in_sync > 1:
                loss_dec = tf.reduce_sum(loss_dec.values)
                size_dec = tf.reduce_sum(size_dec.values)
            else:
                loss_dec = tf.reduce_sum(loss_dec)
                size_dec = tf.reduce_sum(size_dec)
        else:
            loss_dec, size_dec = self.train_step(inp_str, inp_id, tar_id, num_aux_tokens)

        return loss_dec, size_dec

    def train_iteration(self, epoch_start, epoch_end, iteration_counter, save_filepath_train,
                        data_dict, num_aux_tokens, save_end_epoch=True, print_every_iterations=100,
                        save_every_iterations=5000):
        iteration_counter = iteration_counter
        for e in range(epoch_start, epoch_end):
            start = time.time()

            batch = 0
            epoch_loss_total = 0  # sum up all of the losses
            epoch_size_total = 0  # then divide by the total number of losses.
            for (inp_str, inp_id, tar_id) in data_dict["train"]:

                iteration_counter += 1  # one iteration is defined to be one batch.

                loss_dec, size_dec = self._distributed_train_step(inp_str, inp_id, tar_id, num_aux_tokens)
                if size_dec == 0:
                    print(f"The size is zero, skip the current batch, it will not be counted due to an error!")
                    continue  # start the next batch.

                epoch_loss_total += loss_dec
                epoch_size_total += size_dec

                # loss for the current batch.
                #loss_ = (loss_enc+loss_dec) / (size_enc+size_dec)
                loss_ = loss_dec / size_dec
                loss_ = tf.cast(loss_, dtype=tf.dtypes.float32)

                if iteration_counter % print_every_iterations == 0:
                    print(f'Iteration {iteration_counter} Epoch {e+1} Batch {batch+1} Loss {loss_:.4f}')
                batch += 1

                if (iteration_counter) % save_every_iterations == 0:
                    ckpt_save_path = self.ckpt_manager.save()
                    print(f'Saving checkpoint for iteration {iteration_counter} at {ckpt_save_path}')

                header = True if iteration_counter == 1 else False
                self._save_iteration_results_nm("train", iteration_counter, save_filepath_train, header, loss_, None, None)

            # this is the loss of all words divided by the number of words (while eliminating the effect of the padding tokens)
            total_loss = epoch_loss_total / epoch_size_total
            print(f'Epoch {e+1} Loss {total_loss:.4f}')
            print(f'Time taken for epoch {e+1}: {time.time() - start:.2f} secs\n')

            header = True if e == 0 else False
            self._save_epoch_results_nm("train", e+1, save_filepath_train, header, total_loss, None, None)

            if save_end_epoch:
                ckpt_save_path = self.ckpt_manager.save()
                print(f'Saving checkpoint for epoch {e+1} at {ckpt_save_path}')

    def _save_epoch_results_nm(self, type_, epoch, save_filepath, header, total_loss, loss_enc, loss_dec):
        assert isinstance(type_, str) and isinstance(save_filepath, str)
        file = save_filepath+type_+"epoch"+".txt"
        with open(file, "a") as f:
            if header: f.write("Epoch Loss Loss_encoder Loss_decoder \n")
            f.write(f"{epoch} {total_loss} {loss_enc} {loss_dec} \n")

    def _save_iteration_results_nm(self, type_, iteration, save_filepath, header, total_loss, loss_enc, loss_dec):
        assert isinstance(type_, str) and isinstance(save_filepath, str)
        file = save_filepath+type_+"iteration"+".txt"
        with open(file, "a") as f:
            if header: f.write("Iteration Loss Loss_encoder Loss_decoder \n")
            f.write(f"{iteration} {total_loss} {loss_enc} {loss_dec} \n")


def loss_function(target, prediction, loss_object, padding_id):

    mask = tf.math.logical_not(tf.math.equal(target, padding_id))  # padding mask
    loss_ = loss_object(target, prediction) # get loss from the loss_object
    mask = tf.cast(mask, dtype=loss_.dtype) # convert the mask to the correct format.
    loss_ *= mask # apply masking to positions that correspond to a <pad> token in the target.
    return tf.reduce_sum(loss_), tf.reduce_sum(mask)