import tensorflow as tf
import math
import time

#import checkmate
#from checkmate import BestCheckPointSaver

import sys
sys.path.append("..")

from models.attention_masks import *

class ParentTrainNL:
    '''
    Class ParentTrainNL
    Description: Parent class that implements basic functions needed for training.
    Input:
        (note: tensorflow models/losses and optimizers are the only that are supported,
            however the only tokenizers that are supported are from huggingface)
        model: The previously initialized model.
        optimizer: The optimizer to use during training.
        loss_object: e.g. SparseCategoricalCrossentropy
        loss_function: Uses the loss_object to get the loss.
        tokenizer: huggingface transformer.
        checkpoint_path: (string) path to the checkpoint folder.
        strategy: (tf.distribute...) Strategy for splitting the training across multiple gpus.
    '''
    def __init__(self, model, optimizer, loss_object, loss_function, tokenizer,
                 checkpoint_recent_path, checkpoint_best_path, strategy, pad_token="<pad>"):

        self.model = model
        self.optimizer = optimizer
        self.loss_object = loss_object
        self.loss_function = loss_function
        self.tokenizer = tokenizer

        self.checkpoint_recent_path = checkpoint_recent_path
        #self.checkpoint_best_path = checkpoint_best_path
        self.strategy = strategy

        self.padding_id = self.tokenizer.encode(pad_token,
                                                add_special_tokens=False,
                                                pad_to_max_length=False,
                                                return_token_type_ids=False,
                                                return_attention_mask=False)
        if len(self.padding_id) == 1:
            self.padding_id = self.padding_id[0]
        else:
            raise Exception("The padding token should only have one id!")

        self._create_recent_checkpoint()
        #self._create_best_checkpoint()

    def _create_recent_checkpoint(self):
        self.ckpt = tf.train.Checkpoint(model=self.model,
                                        optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_recent_path, max_to_keep=5)  # maybe remove max_to_keep?

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("Latest checkpoint restored.")

    def _create_best_checkpoint(self):
        pass


    def train_step(self, tar_inp, tar_real, nm_inp):

        look_ahead_mask = create_combined_mask(tar_inp, padding_id=self.padding_id) # combined mask is padded and look ahead together.
        dec_padding_mask = None # becuase no enc_output as input.
        nm_dec_padding_mask = create_combined_mask(nm_inp, padding_id=self.padding_id) # combined mask overcomes cheating/look ahead problem.

        loss, size = 0, 0
        try:
            with tf.GradientTape() as tape:

                predictions, _ = self.model(tar_inp, nm_inp, True, look_ahead_mask, dec_padding_mask,
                                                  nm_dec_padding_mask, external_memory=False) # ret (output, attention weights)

                loss, size = self.loss_function(tar_real, predictions, self.loss_object, self.padding_id)

                loss_ = loss/size

            gradients = tape.gradient(loss_, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        except AssertionError as ae:
            print(f"Assertion error was caught, meaning that the number of elements in this batch is zero. \n Setting the loss and size to zero...")
        except Exception as e:
            print(f"The error message is:\n {e}")

        return loss, size

    #(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    #tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    #tf.TensorSpec(shape=(None, None), dtype=tf.int64),])

    @tf.function
    def _distributed_train_step(self, tar_inp, tar_real, nm_inp):
        if self.strategy is not None:
            loss, size = self.strategy.run(self.train_step, args=(tar_inp, tar_real, nm_inp,))

            # The if else may be totally irrelevant.
            if self.strategy.num_replicas_in_sync > 1:
                loss = tf.reduce_sum(loss.values)
                size = tf.reduce_sum(size.values)
            else:
                # todo: test below, nan values shouldn't be an issue.
                loss = tf.reduce_sum(loss)
                size = tf.reduce_sum(size)
        else:
            loss, size = self.train_step(tar_inp, tar_real, nm_inp)

        return loss, size

    def train_(self, epoch_start, epoch_end, save_filepath_train, save_filepath_val, data_dict):

        for e in range(epoch_start, epoch_end):
            start = time.time()

            batch = 0
            epoch_loss = 0 # sum up all of the losses
            epoch_size = 0 # then divide by the total number of losses.
            for (tar_inp, tar_real, nm_inp) in data_dict["train"]:

                loss, size = self._distributed_train_step(tar_inp, tar_real, nm_inp)
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
                    print(f'Epoch {e+1} Batch {batch} Loss {loss_:.4f} Perplexity {perp:.4f} Bits Per Word (bpc) {bpc:.4f}')
                batch += 1

            total_loss = epoch_loss/epoch_size # this is the loss of all words divided by the number of words (while eliminating the effect of the padding tokens)
            dict__ = self.perplexity_bpc_function(total_loss)
            if "perplexity" in dict_.keys():
                epoch_perp = dict__["perplexity"]
            if "bpc" in dict_.keys():
                epoch_bpc = dict__["bpc"]
            print(
                f'Epoch {e+1} Loss {total_loss:.4f} Perplexity {epoch_perp:.4f} Bits Per Word (bpc) {epoch_bpc:.4f}')
            print(f'Time taken for epoch {e+1}: {time.time() - start:.2f} secs\n')
            # TODO: change from %1 later, just testing to see if works initially.
            if (e+1) % 1 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print(f'Saving checkpoint for epoch {e+1} at {ckpt_save_path}')

            header = True if e == 0 else False
            self._save_epoch_results("train", e+1, save_filepath_train, header, total_loss, epoch_perp, epoch_bpc)

            if "val" in data_dict.keys():
                print(f"Running through the validation set now!")
                self._run_validation(e, save_filepath_val, data_dict) # note e+1 is not correct.

    def _run_validation(self, e, save_filepath, data_dict):
        start = time.time()
        #batch = 0
        # Still calculate below to get the perplexity and bits per Word scores.
        epoch_loss = 0  # sum up all of the losses
        epoch_size = 0  # then divide by the total number of losses.
        for (tar_inp, tar_real, nm_inp) in data_dict["val"]:
            loss, size = self._distributed_val_step(tar_inp, tar_real, nm_inp)
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
        if "perplexity" in dict_.keys():
            epoch_perp = dict__["perplexity"]
        if "bpc" in dict_.keys():
            epoch_bpc = dict__["bpc"]
        print(
            f'Epoch {e+1} Val Loss {total_loss:.4f} Val Perplexity {epoch_perp:.4f} Val Bits Per Word (bpc) {epoch_bpc:.4f}')
        print(f'Time taken for one epoch (val) {e+1}: {time.time() - start:.2f} secs\n')

        header = True if e == 0 else False
        self._save_epoch_results("val", e+1, save_filepath, header, total_loss, epoch_perp, epoch_bpc)

    def val_step(self, tar_inp, tar_real, nm_inp):
        # equivalent to train_step except the tape

        look_ahead_mask = create_combined_mask(tar_inp, padding_id=self.padding_id)
        dec_padding_mask = None  # becuase no enc_output as input.
        nm_dec_padding_mask = create_combined_mask(nm_inp, padding_id=self.padding_id)

        loss, size = 0, 0
        try:
            predictions, _ = self.model(tar_inp, nm_inp, True, look_ahead_mask, dec_padding_mask,
                                              nm_dec_padding_mask,
                                              external_memory=False)  # ret (output, attention weights)
            loss, size = self.loss_function(tar_real, predictions, self.loss_object, self.padding_id)
        except AssertionError as ae:
            print(f"Assertion error was caught, meaning that the number of elements in this batch is zero. \n Setting the loss and size to zero...")
        except:
            print(f"Error... Continuing... Setting the loss and size to zero for this batch subset.")

        return loss, size

    def _distributed_val_step(self, tar_inp, tar_real, nm_inp):
        if self.strategy is not None:
            loss, size = self.strategy.run(self.val_step, args=(tar_inp, tar_real, nm_inp,))

            if self.strategy.num_replicas_in_sync > 1:
                loss = tf.reduce_sum(loss.values)
                size = tf.reduce_sum(size.values)
            else:
                # todo: test below, nan values shouldn't be an issue.
                loss = tf.reduce_sum(loss)
                size = tf.reduce_sum(size)

            #print(f"Loss: {loss} \n Size: {size}")
        else:
            loss, size = self.val_step(tar_inp, tar_real, nm_inp)

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
            if header and type_ == "train":
                f.write("Epoch Loss Perplexity BPC \n")
            elif header:
                f.write("Epoch Loss Perplexity BPC \n")
                # todo below is irrelevant at this stage
            if type_ == "train":
                f.write(f"{epoch} {total_loss} {epoch_perp} {epoch_bpc} \n")
            else:
                f.write(f"{epoch} {total_loss} {epoch_perp} {epoch_bpc} \n")

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

    # below returns average loss for each individual item in batch, this is equal to the length of the batch size.
    # this is needed to be able to use MEAN in gpu distribution strategy.
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