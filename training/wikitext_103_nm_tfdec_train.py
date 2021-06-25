import tensorflow as tf
import numpy as np
import time
import math

import sys
sys.path.append("..")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

# Own module related imports.
from models.Neuromodulation import *
from models.NMTransformerDec import *
from models.attention_masks import *
from transformers import XLNetTokenizer
from load_datasets.load_wikitext_103 import *
from text_processing.create_tfxl_vocab import get_tfxl_tokenizer, get_xlnet_tokenizer

# torch related imports
#from torch.utils.data.dataset import Dataset
#from torch.utils.data import DataLoader


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

class TrainTFDec_NM:
    '''
    Class: TrainTFDec_NM
    Description: Class that performs training for the tensorflow decoder and neuromodulation network.
    Input:
        (note: below has been replaced with the transformer class itself, already initialized)
        nm_tf_dec_dict: (dict) Contains the following keys with values (key:value_type)
            num_layers: (int)
            d_model: (int)
            num_heads: (int)
            dff: (int)
            ffn_dict_dec: (dict)
            max_seq_len_dec: (int)
            target_vocab_size: (int)
            pe_target: (int)
            rate_dec: (float)
            nm_mha_dec: (bool)
            enc_out: (bool)
            neuromodulation: (bool)
            nm_net_vocab_size: (int)
            pe_nm_net: (int)
            rate_nm_enc: (flaot)
            nm_mha_net: (bool)
            ffn_dict_nm: (dict)
            max_seq_len_nm: (int)
        transformer:
        optimizer:
        loss_object:
        loss_function:
        accuracy_function: - renamed metric function, then removed.
        tokenizer: (note currently only supporting XLNetTokenizer)
        checkpoint_path: (str)
        data_loader_dict: (dict)
        strategy: (tf.distribute.Strategy)
    '''

    # (None, None) (batch, seq_len) -> both can vary hence, set both to None.
    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64)
    ]

    def __init__(self, transformer, max_seq_len_dec, max_seq_len_nm, optimizer, loss_object, loss_function, tokenizer,
                 checkpoint_path, data_dict, strategy):

        self.strategy = strategy

        #self._initialize_network(nm_tf_dec_dict)
        self.transformer = transformer
        self.max_seq_len_dec = max_seq_len_dec
        self.max_seq_len_nm = max_seq_len_nm

        self.optimizer = optimizer # E.g the adam optimizer. # note: needs to be initialzed in the scope of the strategy previously.
        self.loss_object = loss_object # E.g. SparseCategoricalCrossentropy
        self.loss_function = loss_function
        #self.metric_function = metric_function

        self.tokenizer = tokenizer
        self.checkpoint_path = checkpoint_path
        self.data_dict = data_dict

        self._create_checkpoint()

        self.padding_id = tokenizer.encode("<pad>",
                                    add_special_tokens=False,
                                    pad_to_max_length=False,
                                    return_token_type_ids=False,
                                    return_attention_mask=False)
        if len(self.padding_id) == 1:
            self.padding_id = self.padding_id[0]
        else:
            raise Exception("The padding token should only have one id.")

        #checkpoint_path = "./checkpoints/train"

        '''
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_perplexity = tf.keras.metrics.Mean(name="train_perplexity")
        self.train_bpc = tf.keras.metrics.Mean(name="train_bpc")

        self.val_perplexity = tf.keras.metrics.Mean(name="train_perplexity")
        self.val_bpc = tf.keras.metrics.Mean(name="train_bpc")
        '''


    def _create_checkpoint(self):
        self.ckpt = tf.train.Checkpoint(transformer=self.transformer,
                                        optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=10)  # maybe remove max_to_keep

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("Latest checkpoint restored.")

    def _initialize_network(self, _dict):

        self.num_layers = _dict["num_layers"]
        self.d_model = _dict["d_model"]
        self.num_heads = _dict["num_heads"]
        self.dff = _dict["dff"]
        self.max_seq_len_dec = _dict["max_seq_len_dec"]
        self.max_seq_len_nm = _dict["max_seq_len_nm"]
        with self.strategy.scope():
            self.transformer = NMTransformerDec(_dict["num_layers"], _dict["d_model"], _dict["num_heads"], _dict["dff"],
                                            _dict["ffn_dict_dec"], _dict["max_seq_len_dec"], _dict["target_vocab_size"],
                                            _dict["pe_target"], rate_dec=_dict["rate_dec"], nm_mha_dec=_dict["nm_mha_dec"],
                                            enc_out=_dict["enc_out"], neuromodulation=_dict["neuromodulation"],
                                            nm_net_vocab_size=_dict["nm_net_vocab_size"], pe_nm_net=_dict["pe_nm_net"],
                                            rate_nm_enc=_dict["rate_nm_enc"], nm_mha_net=_dict["nm_mha_net"],
                                            ffn_dict_nm=_dict["ffn_dict_nm"], max_seq_len_nm=_dict["max_seq_len_nm"])

    #@tf.function(input_signature=train_step_signature)
    def train_step(self, tar_inp, tar_real, nm_inp):

        #batch_size = tar_inp.shape[0]
        # this means that the batch size is 0 and will cause an error. If so
        #if batch_size == 0: return 0
        #else: self.batch_helper += 1

        look_ahead_mask = create_look_ahead_mask(tar_inp.shape[1])
        dec_padding_mask = create_padding_mask(tar_inp, padding_id=self.padding_id)
        nm_dec_padding_mask = create_padding_mask(nm_inp, padding_id=self.padding_id)
        # TODO: create look_ahead_mask for nm_dec?

        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(tar_inp, nm_inp, True, look_ahead_mask, dec_padding_mask,
                                              nm_dec_padding_mask, external_memory=False) # ret (output, attention weights)
            loss, size = self.loss_function(tar_real, predictions, self.loss_object, self.padding_id)
        #if self.print_loss: print(f"the loss is {loss} \n tar_inp: {tar_inp} \n nm_inp: {nm_inp} \n")
            loss_ = loss/size

        gradients = tape.gradient(loss_, self.transformer.trainable_variables)  # TODO should check trainable variables in my model.
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        #dict_ = self.metric_function(loss)
        #print("LOSS test", loss)
        #v = tf.math.is_nan(loss)

        #if tf.math.is_nan(loss):
        #    loss =
        return loss, size

    def __sum_helper(self, values):

        sum_ = 0
        for i in range(len(values)):
            val = values[i].eval().item()
            print(val)
            if isinstance(val, float) or isinstance(val, int):
                sum_ += val
        return sum_

    @tf.function
    def _distributed_train_step(self, tar_inp, tar_real, nm_inp):
        loss, size = self.strategy.run(self.train_step, args=(tar_inp, tar_real, nm_inp,))
        # loss will be in the form (batch_size, max_seq_len)
        #print(loss)
        #loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
        #size = self.strategy.reduce(tf.distribute.ReduceOp.SUM, size, axis=None)

        if self.strategy.num_replicas_in_sync > 1:
            loss = sum(loss.values)#self.__sum_helper(loss.values)
            size = sum(size.values)#self.__sum_helper(size.values)
        else:
            # todo: extra work needed for below? check if they are in a tuple as well or not and adjust the code appropriately.
            loss = sum(loss.values)
            size = sum(size.values)

        #loss = loss / self.batch_helper
        # reset self.batch_size to zero.
        #self.batch_size = 0
        return loss, size

    def train(self, epochs, save_filepath):

        for e in range(epochs):
            start = time.time()

            batch = 0
            epoch_loss = 0
            epoch_size = 0
            for (tar_inp, tar_real, nm_inp) in self.data_dict["train"]:
                #if batch == 273: self.print_loss = True
                loss, size = self._distributed_train_step(tar_inp, tar_real, nm_inp)
                epoch_loss += loss
                epoch_size += size
                #if math.isnan(loss):
                #    print(f"The current loss is nan becuase of issue with last batch size that doesn't fill"
                #                           f"up all gpu devices with multipleworkers. Skip this epoch.")
                #    continue
                #if batch == 273: self.print_loss = False
                #print(f"The loss here for debugging purposes {loss}")
                #raise Exception("Break here for testing.")
                #self.train_step(nm_inp, tar_inp, tar_real)
                loss_ = loss / size
                loss_ = tf.cast(loss_, dtype=tf.dtypes.float32)
                dict_ = self.metric_function(loss_)
                #self.train_loss.update_state(loss_)

                if "perplexity" in dict_.keys():
                    #self.train_perplexity.update_state(dict_["perplexity"])
                    perp = dict_["perplexity"]
                if "bpc" in dict_.keys():
                    #self.train_bpc.update_state(dict_["bpc"])
                    bpc = dict_["bpc"]

                # TODO: run through validation at the end of every epoch! remember to match in the results clearly.
                if batch % 1 == 0:
                    print(f'Epoch {e+1} Batch {batch} Loss {loss_:.4f} Perplexity {perp:.4f} Bits Per Character (bpc) {bpc:.4f}')
                batch += 1

            total_loss = epoch_loss/epoch_size
            dict__ = self.metric_function(total_loss)
            if "perplexity" in dict_.keys():
                epoch_perp = dict_["perplexity"]
            if "bpc" in dict_.keys():
                epoch_bpc = dict_["bpc"]
            print(
                f'Epoch {e+1} Loss {total_loss:.4f} Perplexity {epoch_perp:.4f} Bits Per Character (bpc) {epoch_bpc:.4f}')
            print(f'Time taken for epoch {e+1}: {time.time() - start:.2f} secs\n')
            # TODO: change from %1 later, just testing to see if works initially.
            if (e+1) % 1 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print(f'Saving checkpoint for epoch {e+1} at {ckpt_save_path}')

            # TODO implement below. Save the results in a text file.
            header = True if e == 0 else False
            self._save_epoch_results("train", e+1, save_filepath, header, total_loss, epoch_perp, epoch_bpc)

            #print(f"Running through the validation set now!")

            #self._run_validation(e, save_filepath) # note e+1 is not correct.

    @tf.function(input_signature=train_step_signature)
    def _val_step(self, nm_inp, tar):
        tar_inp = tar[:, :-1]  # (batch,max_seq_len) # :-1 b/c don't wan't EOS in the input.
        tar_real = tar[:, 1:]  # 1: b/c don't want SOS at the real prediction.

        # TODO: print tar_inp and tar_real to see if the mirrored strategy is still applied to them.

        look_ahead_mask = create_look_ahead_mask(tar_inp.shape[1])
        dec_padding_mask = create_padding_mask(tar_inp, padding_id=5)
        nm_dec_padding_mask = create_padding_mask(nm_inp, padding_id=5)

        predictions, _ = self.transformer(tar_inp, nm_inp, False, look_ahead_mask, dec_padding_mask,
                                              nm_dec_padding_mask,
                                              external_memory=False)  # ret (output, attention weights)
        loss = self.loss_function(tar_real, predictions, self.loss_object, self.padding_id)

        dict_ = self.metric_function(loss)
        if "perplexity" in dict_.keys():
            self.val_perplexity(dict_["perplexity"])
        if "bpc" in dict_.keys():
            self.val_bpc(dict_["bpc"])

    def _run_validation(self, train_epoch, save_filepath):
        self.val_perplexity.reset_states()
        self.val_bpc.reset_states()
        start = time.time()
        for batch, tar in enumerate(self.data_loaders["val"]):

            # currently tar will be in list form. Need to convert to tensor.
            with self.strategy.scope():
                tar_tensor = tf.convert_to_tensor(tar)  # tar is not in tensor format.
                tar_tensor = tf.data.Dataset.from_tensors(tar_tensor)

            # start_ = time.time()
            nm_inp = []
            for i in range(len(tar)):
                nm_inp.append(aux_tok_id + tar[i])
            with self.strategy.scope():
                nm_inp = tf.convert_to_tensor(nm_inp)
                nm_inp = tf.data.Dataset.from_tensors(nm_inp)
            # print(f"The time to create the nm_inp is {time.time() - start_:.2f} seconds.")

            self._val_step(nm_inp, tar_tensor)

        print(f'Perplexity {self.val_perplexity.result():.4f} Bits Per Character (bpc) {self.val_bpc.result():.4f}')
        print(f'Time taken for one validation set run through: {time.time() - start:.2f} secs\n')

        header = True if e == 0 else False
        self._save_epoch_results("train", train_epoch+1, save_filepath, header=header)

        #return {
        #    "perplexity": self.val_perplexity.result(),
        #    "bpc": self.val_bpc.result(),
        #}


    # gets perplexity, bits per character for specified data set.
    def get_metrics(self, input, type_, *nm_auxiliary_tokens):
        # type_ is either "val", "test" (theoretically could be "train"), no training is done here.
        # input is in the shape (batch_size, max_seq_len)
        pass

    # TODO implement this last, not needed currently.
    def evaluate_from_string(self, input, max_output_len=40):
        # input (str).
        pass

    def metric_function(self, loss):
        perplexity = tf.math.exp(tf.reduce_mean(loss))
        bpc = tf.reduce_mean(loss) / tf.constant(math.log(2))
        return {
            "perplexity": perplexity,
            "bpc": bpc
        }

    # type is "train", "val", "test".
    # header means if header is to be appended first.
    def _save_epoch_results(self, type_, epoch, save_filepath, header, total_loss, epoch_perp, epoch_bpc):

        assert isinstance(type_, str) and isinstance(save_filepath, str)
        file = save_filepath+type_+".txt"
        with open(file, "a") as f:
            if header and type_ == "train":
                f.write("Epoch Loss Perplexity BPC \n")
            elif header:
                f.write("Epoch Loss Perplexity BPC \n")
            if type_ == "train":
                f.write(f"{epoch} {total_loss} {epoch_perp} {epoch_bpc} \n")
            else:
                f.write(f"{epoch} {train_perplexity.result()} {train_bpc.result()} \n")


def loss_function(real, pred, loss_object, padding_id, newline_tok=[False, 0]):
    # loss_object should have reduction be set to 'none' so the below works.
    # this is why we reduce_sum the loss as it is in an array and average it by dividying
    # by the sum of the 1 values in the mask (i.e. 1 represents non pad tokens loss)

    # use the mask below to remove influence from padding tokens.
    mask = tf.math.logical_not(tf.math.equal(real, padding_id)) # padding mask?
    # TODO: consider other tokens other than <pad> to do this to.
    mask2 = None
    if newline_tok[0]:
        assert newline_tok[1] != padding_id, f"The new line token id ({newline_tok[1]}) and padding token " \
                                             f"id ({padding_id}) should not be equal."
        mask2 = tf.math.logical_not(tf.math.equal(real, newline_tok[1]))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    if mask2 is not None:
        mask += mask2 # as they don't overlap, this can be done. If they are equal the assert above will go off.
    loss_ *= mask

    # below returns average loss for each individual item in batch, this is equal to the length of the batch size.
    # this is needed to be able to use MEAN in gpu distribution strategy.
    #return tf.reduce_sum(loss_, axis=1) / tf.reduce_sum(mask, axis=1)
    #return tf.reduce_sum(loss_) / tf.reduce_sum(mask)
    return tf.reduce_sum(loss_), tf.reduce_sum(mask)


'''
def main():
    mirrored_strategy = tf.distribute.MirroredStrategy()  # devices=["/gpu:0", "/gpu:1"]
    #learning_rate = CustomTransformerSchedule(d_model)
    #loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
    #                                                            reduction='none')
    # initialize optimizer
    #with mirrored_strategy.scope():
    #    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    #lst_ = [1,2,3]
    #with mirrored_strategy.scope():
    #    data = tf.Variable(1.)#tf.convert_to_tensor(lst_)
    #print(data)

    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    filepath = "/large_data/wikitext-103/wiki.test.tokens"
    max_seq_len, pad_to_max_length = 32, True
    load_data = [True, "/large_data/wikitext-103/test_greedy_msl32.txt"]

    data_test = load_wiki_103(filepath, max_seq_len, tokenizer, pad_to_max_length, load_data=load_data)
    dataloader = DataLoader(data_test, batch_size=4, shuffle=False, num_workers=2, collate_fn=lambda x: x)
    #print(dataloader)

    #for batch, data_ in enumerate(dataloader):
    #    print(f"Batch: {batch} Data: {data_}")
    #    if batch == 8: break

main()
'''
