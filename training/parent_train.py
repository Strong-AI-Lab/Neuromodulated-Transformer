'''
File name: parent_train.py
Author: Kobe Knowles
Date created: 05/07/21
Data last modified: 17/07/21
Python Version: 3.6
Tensorflow version: 2
'''

import tensorflow as tf
import math
import time

#import checkmate
#from checkmate import BestCheckPointSaver

import sys
sys.path.append("..")

from models.AttentionMasks import *

class ParentTrainNL:
    '''
    Class ParentTrainNL
    Description: Parent class that implements basic functions needed for training.
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
                 recent_to_keep=5, load_recent=False, best_to_keep=5, load_best=False):

        assert not(load_recent and load_best), f"load_recent ({load_recent}) and load_best ({load_best}) can't both be set to True!"

        self.model = model
        self.optimizer = optimizer
        self.loss_object = loss_object
        self.loss_function = loss_function
        self.tokenizer = tokenizer

        self.checkpoint_path_recent = checkpoint_path_recent
        self.checkpoint_path_best = checkpoint_path_best
        self.strategy = strategy

        self.padding_id = self.tokenizer.encode_single(pad_token)
        if len(self.padding_id) == 1:
            self.padding_id = self.padding_id[0]
        else:
            raise Exception("The padding token should only have one id! (it hasn't been added to the vocabulary)")

        # the recent checkpoints are required.
        self._create_recent_checkpoint(recent_to_keep)
        if load_recent: self._load_recent_checkpoint()

        #TODO: support for best checkpoint saver to come.
        #if self.checkpoint_path_best != "": self._create_best_checkpoint(best_to_keep)
        #if load_best: self._load_best_checkpoint()

    def _create_recent_checkpoint(self, keep):
        self.ckpt = tf.train.Checkpoint(model=self.model,
                                        optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path_recent, max_to_keep=keep)  # maybe remove max_to_keep?

        #if self.ckpt_manager.latest_checkpoint:
        #    self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        #    print("Latest checkpoint restored.")

    def _load_recent_checkpoint(self):
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("Latest checkpoint restored.")

    def _create_best_checkpoint(self, keep):
        #https://github.com/vonclites/checkmate
        self.best_ckpt_saver = BestCheckpointSaver(save_dir=self.checkpoint_path_best,
                                                   num_to_keep=keep, maximize=True)


    def train_step(self, tar_inp, tar_real, nm_inp, num_aux_tokens):

        nm_mask = create_combined_mask(nm_inp, self.padding_id, num_aux_tokens)
        dec_mask = create_combined_mask(tar_inp, self.padding_id)

        loss, size = 0, 0
        with tf.GradientTape() as tape:
            # TODO: reminder to remove padding_id in below call as it isn't needed? (also num_aux tokens?)
            predictions, _, _ = self.model(tar_inp, nm_inp, training=True, padding_id=self.padding_id,
                                           num_aux_tok=num_aux_tokens, nm_mask=nm_mask, dec_mask=dec_mask) # ret (output, attention weights, nm_output)
            loss, size = self.loss_function(tar_real, predictions, self.loss_object, self.padding_id)

            loss_ = loss/size

        gradients = tape.gradient(loss_, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss, size

    #(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    #tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    #tf.TensorSpec(shape=(None, None), dtype=tf.int64),])

    @tf.function
    def _distributed_train_step(self, tar_inp, tar_real, nm_inp, num_aux_tokens):
        if self.strategy is not None:
            loss, size = self.strategy.run(self.train_step, args=(tar_inp, tar_real, nm_inp, num_aux_tokens,))

            # The if else may be totally irrelevant.
            if self.strategy.num_replicas_in_sync > 1:
                loss = tf.reduce_sum(loss.values)
                size = tf.reduce_sum(size.values)
            else:
                # todo: test below, nan values shouldn't be an issue.
                loss = tf.reduce_sum(loss)
                size = tf.reduce_sum(size)
        else:
            loss, size = self.train_step(tar_inp, tar_real, nm_inp, num_aux_tokens)

        return loss, size

    def train_(self, epoch_start, epoch_end, save_filepath_train, save_filepath_val, data_dict, num_aux_tokens):

        for e in range(epoch_start, epoch_end):
            start = time.time()

            batch = 0
            epoch_loss = 0 # sum up all of the losses
            epoch_size = 0 # then divide by the total number of losses.
            for (tar_inp, tar_real, nm_inp) in data_dict["train"]:

                loss, size = self._distributed_train_step(tar_inp, tar_real, nm_inp, num_aux_tokens)
                if size == 0:
                    print(f"The size is zero, skip the current batch, it will not be counted due to an error!")
                    continue # start the next batch.

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

                if batch % 1 == 0:
                    print(f'Epoch {e+1} Batch {batch} Loss {loss_:.4f} Perplexity {perp:.4f} Bits Per Word (bpw) {bpc:.4f}')
                batch += 1

            total_loss = epoch_loss/epoch_size # this is the loss of all words divided by the number of words (while eliminating the effect of the padding tokens)
            dict__ = self.perplexity_bpc_function(total_loss)
            if "perplexity" in dict_.keys():
                epoch_perp = dict__["perplexity"]
            if "bpc" in dict_.keys():
                epoch_bpc = dict__["bpc"]
            print(
                f'Epoch {e+1} Loss {total_loss:.4f} Perplexity {epoch_perp:.4f} Bits Per Word (bpw) {epoch_bpc:.4f}')
            print(f'Time taken for epoch {e+1}: {time.time() - start:.2f} secs\n')
            # TODO: change from %1 later, just testing to see if works initially.
            if (e+1) % 1 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print(f'Saving checkpoint for epoch {e+1} at {ckpt_save_path}')

            header = True if e == 0 else False
            self._save_epoch_results("train", e+1, save_filepath_train, header, total_loss, epoch_perp, epoch_bpc)

            if "val" in data_dict.keys():
                print(f"Running through the validation set now!")
                self._run_validation(e, save_filepath_val, data_dict["val"], num_aux_tokens) # note e+1 is not correct.

    def _run_validation(self, e, save_filepath, validation, num_aux_tokens):
        start = time.time()
        #batch = 0
        # Still calculate below to get the perplexity and bits per Word scores.
        epoch_loss = 0  # sum up all of the losses
        epoch_size = 0  # then divide by the total number of losses.
        for (tar_inp, tar_real, nm_inp) in validation:
            loss, size = self._distributed_val_step(tar_inp, tar_real, nm_inp, num_aux_tokens)
            epoch_loss += loss
            epoch_size += size

            #loss_ = loss / size
            #loss_ = tf.cast(loss_, dtype=tf.dtypes.float32)
            #dict_ = self.perplexity_bpc_function(loss_)

            #if "perplexity" in dict_.keys():
            #    perp = dict_["perplexity"]
            #if "bpc" in dict_.keys():
            #    bpc = dict_["bpc"]

            # only care about final result.
            #if batch % 1 == 0:
            #    print(
            #        f'Epoch {e+1} Batch {batch} Loss {loss_:.4f} Perplexity {perp:.4f} Bits Per Word (bpc) {bpc:.4f}')
            #batch += 1

        total_loss = epoch_loss / epoch_size  # this is the loss of all words divided by the number of words (while eliminating the effect of the padding tokens)
        dict__ = self.perplexity_bpc_function(total_loss)
        if "perplexity" in dict__.keys():
            epoch_perp = dict__["perplexity"]
        if "bpc" in dict__.keys():
            epoch_bpc = dict__["bpc"]
        print(
            f'Epoch {e+1} Val Loss {total_loss:.4f} Val Perplexity {epoch_perp:.4f} Val Bits Per Word (bpw) {epoch_bpc:.4f}')
        print(f'Time taken for one epoch (val) {e+1}: {time.time() - start:.2f} secs\n')

        header = True if e == 0 else False
        self._save_epoch_results("val", e+1, save_filepath, header, total_loss, epoch_perp, epoch_bpc)

    def val_step(self, tar_inp, tar_real, nm_inp, num_aux_tokens):

        loss, size = 0, 0
        predictions, _, _ = self.model(tar_inp, nm_inp, training=False, padding_id=self.padding_id, num_aux_tok=num_aux_tokens)  # ret (output, attention weights)
        loss, size = self.loss_function(tar_real, predictions, self.loss_object, self.padding_id)

        return loss, size

    @tf.function
    def _distributed_val_step(self, tar_inp, tar_real, nm_inp, num_aux_tokens):
        if self.strategy is not None:
            loss, size = self.strategy.run(self.val_step, args=(tar_inp, tar_real, nm_inp, num_aux_tokens,))

            # The if else may be totally irrelevant.
            if self.strategy.num_replicas_in_sync > 1:
                loss = tf.reduce_sum(loss.values)
                size = tf.reduce_sum(size.values)
            else:
                # todo: test below, nan values shouldn't be an issue.
                loss = tf.reduce_sum(loss)
                size = tf.reduce_sum(size)
        else:
            loss, size = self.val_step(tar_inp, tar_real, nm_inp, num_aux_tokens)

        return loss, size

    # Note: Here I calculate it the same way as in transformer-xl.
    def perplexity_bpc_function(self, loss):
        perplexity = tf.math.exp(tf.reduce_mean(loss))
        bpc = tf.reduce_mean(loss) / tf.constant(math.log(2)) # log is the natural logarithm (i.e. to base e).
        return {
            "perplexity": perplexity,
            "bpc": bpc
        }

    def _save_epoch_results(self, type_, epoch, save_filepath, header, total_loss, epoch_perp, epoch_bpc):

        assert isinstance(type_, str) and isinstance(save_filepath, str)
        file = save_filepath+type_+".txt"
        with open(file, "a") as f:
            #if header and type_ == "train":
            f.write("Epoch Loss Perplexity BPC \n")
            #elif header:
            #    f.write("Epoch Loss Perplexity BPC \n")
            #    # todo below is irrelevant at this stage
            #if type_ == "train":
            f.write(f"{epoch} {total_loss} {epoch_perp} {epoch_bpc} \n")
            #else:
            #    f.write(f"{epoch} {total_loss} {epoch_perp} {epoch_bpc} \n")

    def run_no_train(self, data):
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
        for (tar_inp, tar_real, nm_inp) in data:

            loss, size = self._distributed_val_step(tar_inp, tar_real, nm_inp) # use _distributed_val_step becuase it does exactly what is needed already.
            epoch_loss += loss
            epoch_size += size

            loss_ = loss / size
            loss_ = tf.cast(loss_, dtype=tf.dtypes.float32)
            dict_ = self.perplexity_bpc_function(loss_)

        total_loss = epoch_loss / epoch_size  # this is the loss of all words divided by the number of words (while eliminating the effect of the padding tokens)
        dict__ = self.perplexity_bpc_function(total_loss)
        if "perplexity" in dict_.keys():
            perplexity = dict__["perplexity"]
        if "bpc" in dict_.keys():
            bpc = dict__["bpc"]

        return total_loss, perplexity, bpc

    def generate_natural_language(self, string_input):
        # TODO: implement this later.
        pass

def loss_function_sequence_split(real, pred, loss_object, padding_id):

    # use the mask below to remove influence from padding tokens.
    mask = tf.math.logical_not(tf.math.equal(real, padding_id)) # padding mask?

    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)

    loss_ *= mask

    #return tf.reduce_sum(loss_, axis=1) / tf.reduce_sum(mask, axis=1)
    #return tf.reduce_sum(loss_) / tf.reduce_sum(mask)
    return tf.reduce_sum(loss_), tf.reduce_sum(mask)

class CustomTransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    # https://www.tensorflow.org/text/tutorials/transformer
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomTransformerSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.dtypes.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.sqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.sqrt(self.d_model) * tf.math.minimum(arg1, arg2)