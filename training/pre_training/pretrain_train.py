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

    def __init__(self, model, optimizer, loss_object, loss_function_encoder, loss_function_decoder, tokenizer,
                 checkpoint_path_recent, checkpoint_path_best, strategy, pad_token="<pad>",
                 recent_to_keep=5, load_recent=False, best_to_keep=5, load_best=False, load_specific_path='',
                 enc_tok_id=None, dec_tok_id=None):

        super(NMTransformerEncDecTrain, self).__init__(model, optimizer, loss_object, None, tokenizer,
                 checkpoint_path_recent, checkpoint_path_best, strategy, pad_token,
                 recent_to_keep, load_recent, best_to_keep, load_best, load_specific_path)

        assert enc_tok_id is not None and dec_tok_id is not None, f"It is required that the encoder and decoder token id is passed as input during initialization!"
        self.enc_tok_id = enc_tok_id
        self.dec_tok_id = dec_tok_id

        self.loss_function_encoder = loss_function_encoder
        self.loss_function_decoder = loss_function_decoder

    def train_step(self, inp_id, tar_id, nm_inp_id, num_aux_tokens):

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
        with tf.GradientTape() as tape:
            predictions, _, _, _, _ = self.model(inp_id, nm_inp_id, training=True, nm_mask=nm_mask, dec_mask=dec_mask) # ret (output, attention weights, nm_output)
            '''
            le, se, ld, sd = 0, 0, 0, 0
            for b, item in enumerate(mode_list):
                print(f"b: {b}")
                if item == "enc": # batch_size, seq_len, vocab_size
                    #print(f"tar_id.shape: {tar_id.shape}")
                    #print(f"enc test... 1 {tf.expand_dims(tar_id[b,:], axis=0).shape}")
                    #print(f"enc test... 2 {tf.expand_dims(predictions[b,:,:], axis=0)}")
                    loss_enc_, size_enc_ = self.loss_function_encoder(tf.expand_dims(tar_id[b,:], axis=0),
                                                                      tf.expand_dims(predictions[b,:,:], axis=0),
                                                                      self.loss_object, self.padding_id)
                    le += loss_enc_
                    se += size_enc_
                elif item == "dec":
                    #print(f"tar_id.shape: {tar_id.shape}")
                    #print(f"dec test... 1 {tf.expand_dims(tar_id[b, :], axis=0).shape}")
                    #print(f"dec test... 2 {tf.expand_dims(predictions[b, :, :], axis=0)}")
                    loss_dec_, size_dec_ = self.loss_function_decoder(tf.expand_dims(tar_id[b,:], axis=0),
                                                                      tf.expand_dims(predictions[b,:,:], axis=0),
                                                                      self.loss_object, self.padding_id)
                    ld += loss_dec_
                    sd += size_dec_

            #loss_enc_, size_enc_ = self.loss_function_encoder(tar_id, predictions, self.loss_object, self.padding_id)
            loss_ = (le+ld)/(se+sd)
            loss_enc += le
            size_enc += se
            loss_dec += ld
            size_dec += sd
            '''
            loss, size = self.loss_function_decoder(tar_id, predictions, self.loss_object, self.padding_id)
            loss_ = loss / size
        gradients = tape.gradient(loss_, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        #return loss_enc, size_enc, loss_dec, size_dec
        return loss, size, loss, size

    @tf.function
    def _distributed_train_step(self, inp_id, tar_id, nm_inp_id, num_aux_tokens):
        if self.strategy is not None:
            loss_enc, size_enc, loss_dec, size_dec = self.strategy.run(self.train_step, args=(inp_id, tar_id, nm_inp_id, num_aux_tokens,))

            # The if else may be totally irrelevant.
            if self.strategy.num_replicas_in_sync > 1:
                loss_enc = tf.reduce_sum(loss_enc.values)
                size_enc = tf.reduce_sum(size_enc.values)
                loss_dec = tf.reduce_sum(loss_dec.values)
                size_dec = tf.reduce_sum(size_dec.values)
            else:
                loss_enc = tf.reduce_sum(loss_enc)
                size_enc = tf.reduce_sum(size_enc)
                loss_dec = tf.reduce_sum(loss_dec)
                size_dec = tf.reduce_sum(size_dec)
        else:
            loss_enc, size_enc, loss_dec, size_dec = self.train_step(inp_id, tar_id, nm_inp_id, num_aux_tokens)

        return loss_enc, size_enc, loss_dec, size_dec

    def train_iteration(self, epoch_start, epoch_end, iteration_counter, save_filepath_train,
                        data_dict, num_aux_tokens, save_end_epoch=True, print_every_iterations=100,
                        save_every_iterations=5000):
        iteration_counter = iteration_counter
        for e in range(epoch_start, epoch_end):
            start = time.time()

            batch = 0
            epoch_loss_total = 0  # sum up all of the losses
            epoch_size_total = 0  # then divide by the total number of losses.
            epoch_loss_enc = 0
            epoch_size_enc = 0
            epoch_loss_dec = 0
            epoch_size_dec = 0
            for (inp_str, inp_id, tar_id, nm_inp_id) in data_dict["train"]:

                iteration_counter += 1  # one iteration is defined to be one batch.

                loss_enc, size_enc, loss_dec, size_dec = self._distributed_train_step(inp_id, tar_id, nm_inp_id, num_aux_tokens) # TODO
                if size_enc+size_dec == 0:
                    print(f"The size is zero, skip the current batch, it will not be counted due to an error!")
                    continue  # start the next batch.

                epoch_loss_total += (loss_enc)#+loss_dec) # (#
                epoch_size_total += (size_enc)#+size_dec) # (#
                epoch_loss_enc += loss_enc
                epoch_size_enc += size_enc
                epoch_loss_dec += loss_dec
                epoch_size_dec += size_dec

                # loss for the current batch.
                #loss_ = (loss_enc+loss_dec) / (size_enc+size_dec)
                loss_ = loss_enc / size_enc
                loss_ = tf.cast(loss_, dtype=tf.dtypes.float32)
                # (decoder).
                loss_dec_ = loss_dec / size_dec
                loss_dec_ = tf.cast(loss_dec_, dtype=tf.dtypes.float32)
                # (encoder)
                loss_enc_ = loss_enc / size_enc
                loss_enc_ = tf.cast(loss_enc_, dtype=tf.dtypes.float32)

                if iteration_counter % print_every_iterations == 0:
                    print(f'Iteration {iteration_counter} Epoch {e+1} Batch {batch+1} Loss {loss_:.4f}'
                          f' Loss(encoder) {loss_enc_:.4f} Loss(decoder) {loss_dec_:.4f}')
                batch += 1

                if (iteration_counter) % save_every_iterations == 0:
                    ckpt_save_path = self.ckpt_manager.save()
                    print(f'Saving checkpoint for iteration {iteration_counter} at {ckpt_save_path}')

                header = True if iteration_counter == 1 else False
                self._save_iteration_results_nm("train", iteration_counter, save_filepath_train, header, loss_, loss_enc_, loss_dec_)

            total_loss = epoch_loss_total / epoch_size_total  # this is the loss of all words divided by the number of words (while eliminating the effect of the padding tokens)
            enc_loss = epoch_loss_enc / epoch_size_enc
            dec_loss = epoch_loss_dec / epoch_loss_dec
            print(f'Epoch {e+1} Loss {total_loss:.4f} Loss(encoder) {enc_loss:.4f} Loss(decoder) {dec_loss:.4f}')
            print(f'Time taken for epoch {e+1}: {time.time() - start:.2f} secs\n')

            header = True if e == 0 else False
            self._save_epoch_results_nm("train", e+1, save_filepath_train, header, total_loss, enc_loss, dec_loss)

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