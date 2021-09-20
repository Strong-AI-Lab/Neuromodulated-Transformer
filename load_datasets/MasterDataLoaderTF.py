'''
File name: MasterDataLoaderTF.py
Author: Kobe Knowles
Date created: 26/08/21
Data last modified: 26/08/21
Python Version: 3.8
Tensorflow version: 2.5
'''

import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
sys.path.append("..")

import tensorflow as tf
import numpy as np
import transformers

import gzip
import json

from text_processing.tokenizer import Tokenizer
import re
import random

from load_datasets.pre_training.load_C4 import *

class MasterDataLoaderTF(object):
    '''

    '''
    def __init__(self, filepaths: dict, seq_len: int, batch_size: int, tokenizer=None,
                 enc_tok="<enc>", dec_tok="<dec>", mlm_tok="<mlm>", lm_tok="<lm>",
                 cls_tok="<cls>", sep_tok="<sep>", mask_tok="<mask>", pad_tok="<pad>",
                 start_tok="<s>", end_tok="</s>", null_tok="<null>",
                 num_reading_strategies=6, pad_to_max_length=True, strategy="random",
                 C4_processed_filepath=""):
        '''
        Function: __init__ \n
        Description: Initialization of class's parameters. \n
        Input:
            filepaths: (dict: key=dataset, value=filepath) Dictionary containing all filepaths to draw from based on said strategy. \n
            seq_len: (int) The length of the sequence to pass into the model. \n
            batch_size: (int) The number of samples in a batch. \n
            tokenizer: (text_processing.tokenizer.Tokenizer) Higher level tokenizer class. \n
            start_tok: (str) The token to append to the start of a sequence (for language modelling tasks). \n
            end_tok: (str) The token to append to the end of a sequence (for language modelling tasks). \n
            pad_tok: (str) The token to apply to match a sequence to the input seq_len. \n
            null_tok: (str) The token that identifies with the null token. \n
            enc_tok: (str) The token that corresponds to the encoder mode. \n
            dec_tok: (str) The token that corresponds to the decoder mode. \n
            num_reading_strategies: (int) The number of reading strategies in the auxiliary tokens. \n
            pad_to_max_length: (bool) True if we pad the input to seq_len, False otherwise.\n
            strategy: (str) The strategy to implement when loading multiple datasets into a tensorflow data format.
        '''

        self.filepaths = filepaths
        self.C4_processed_filepath = C4_processed_filepath
        self.dataLoaders = dict()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.tokenizer = tokenizer

        self.enc_tok = enc_tok
        self.dec_tok = dec_tok
        self.mlm_tok = mlm_tok
        self.lm_tok = lm_tok
        self.cls_tok = cls_tok
        self.sep_tok = sep_tok
        self.mask_tok = mask_tok

        self.start_tok = start_tok
        self.end_tok = end_tok
        self.pad_tok = pad_tok
        self.null_tok = null_tok
        self.num_reading_strategies = num_reading_strategies
        self.pad_to_max_length = pad_to_max_length

        self.strategy = strategy

        for key in self.filepaths.keys():
            if key == "C4_nm_pre_train":
                self.dataLoaders["C4_nm_pre_train"] = C4DataLoader(strategy="train", filepath=self.filepaths["C4_nm_pre_train"], enc_tok=self.enc_tok,
                                                     dec_tok=self.dec_tok, mlm_tok=self.mlm_tok, lm_tok=self.lm_tok,
                                                     start_tok=self.start_tok, end_tok=self.end_tok, cls_tok=self.cls_tok,
                                                     sep_tok=self.sep_tok, mask_tok=self.mask_tok, pad_tok=self.pad_tok,
                                                     seq_len=self.seq_len, pad=False, tokenizer=tokenizer,
                                                     processed_filepath=self.C4_processed_filepath)
            #elif ...

    def pre_train_c4(self):

        break_ = False
        while True:
            counter = 0
            while True:
                if counter == self.batch_size: break
                input_string, input_id, target_input_id, aux_tok_1, aux_tok_2 = self.dataLoaders["C4_nm_pre_train"](mode="enc_dec") # return string form (i.e. in a list ... of the correct size)
                if input_id is None or input_string is None: # if here is reached we are done for the current epoch.
                    break_ = True
                    #print("\nREACH HOPEFULLY NOT\n")
                    break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]
                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len-len(input_id))]
                    target_input_id = target_input_id + [pad_tok_id for _ in range(self.seq_len - len(target_input_id))]

                # get nm_tokens version (i.e. append aux_tokens to the start of input_id) don't need to do it for input_string.
                #nm_input_id = [self.tokenizer.encode_single(aux_tok_1)[0],
                #               self.tokenizer.encode_single(aux_tok_2)[0]] + \
                #              [self.tokenizer.encode_single(self.null_tok)[0] for _ in range(self.num_reading_strategies)] + \
                #              input_id # input_id will be a list.
                null_tok = self.tokenizer.encode_single(self.null_tok)[0]
                nm_input_id = [aux_tok_1, aux_tok_2] + \
                              [null_tok for _ in range(self.num_reading_strategies)] + \
                              input_id # input_id will be a list.

                input_string = tf.cast(tf.convert_to_tensor(np.asarray(input_string)), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(np.asarray(input_id)), dtype=tf.dtypes.int64)
                target_input_id = tf.cast(tf.convert_to_tensor(np.asarray(target_input_id)), dtype=tf.dtypes.int64)
                nm_input_id = tf.cast(tf.convert_to_tensor(np.asarray(nm_input_id)), dtype=tf.dtypes.int64)

                counter += 1
                yield input_string, input_id, target_input_id, nm_input_id
                #yield input_id, target_input_id, nm_input_id
                # need tf.string... here.
            if break_: break # TODO what happens if batch size is not fully complete? Doesn't really matter, the generator will handle it...

    def get_generator(self, type: str, shuffle: bool):
        self.shuffle = shuffle
        generator = None
        if type == "C4_pretrain_enc_dec":
            generator = tf.data.Dataset.from_generator(self.pre_train_c4,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64))
        elif type == "C4_pretrain_enc":
            generator = tf.data.Dataset.from_generator(self.pre_train_c4_enc_only,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64))
        elif type == "C4_pretrain_dec":
            generator = tf.data.Dataset.from_generator(self.pre_train_c4_dec_only,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64))

        return generator

if __name__ == "__main__":

    filepaths = {"C4_nm_pre_train":"/large_data/C4/en/"}
    tok = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
    tokenizer = Tokenizer(tok)
    tokenizer.add_tokens_list(["<enc>", "<dec>", "<mlm>", "<lm>", "<s>", "</s>", "<cls>", "<sep>", "<mask>", "<pad>"])

    dloader = MasterDataLoaderTF(filepaths=filepaths, seq_len=1024, batch_size=4, tokenizer=tokenizer,
                 enc_tok="<enc>", dec_tok="<dec>", mlm_tok="<mlm>", lm_tok="<lm>",
                 cls_tok="<cls>", sep_tok="<sep>", mask_tok="<mask>", pad_tok="<pad>",
                 start_tok="<s>", end_tok="</s>", null_tok="<null>",
                 num_reading_strategies=6, pad_to_max_length=True, strategy="random")

    generator = dloader.get_generator("C4_pretrain", False).batch(4)
    print(f"generator: {generator}")
    batch_ = 1
    print("REACH")
    for (inp_str, inp_id, tar_id, nm_inp_id) in generator:
        print(f"batch: {batch_}")
        #print(f"inp_str: {inp_str.shape} \t inp_str: {inp_str} \n"
        #    f"inp_id.shape: {inp_id.shape} \t inp_id: {inp_id} \n"
        #      f"tar_id.shape: {tar_id.shape} \t tar_id: {tar_id} \n"
        #      f"nm_inp_id.shape: {nm_inp_id.shape} \t nm_inp_id: {nm_inp_id} \n")
        if batch_ == 10: break
        batch_ += 1
