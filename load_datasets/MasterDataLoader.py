'''
File name: MasterDataLoaderTF.py
Author: Kobe Knowles
Date created: 26/08/21
Data last modified: 26/08/21
Python Version: 3.8
Tensorflow version: 2.5
'''

import os
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
sys.path.append("..")

import tensorflow as tf
import numpy as np
import transformers

import gzip
import json

from text_processing.tokenizer import Tokenizer
from transformers import BertTokenizer
import re
import random

from models.config_model import *

from load_datasets.pre_training.loadC4 import *
#from load_datasets.question_answering.load_race import *

class MasterDataLoaderTF(object):
    '''
    '''
    def __init__(self, filepaths: dict, seq_len: int, batch_size: int, tokenizer=None,
                 enc_tok="<enc>", dec_tok="<dec>", mlm_tok="<mlm>", lm_tok="<lm>",
                 cls_tok="<cls>", sep_tok="<sep>", mask_tok="<mask>", pad_tok="<pad>",
                 start_tok="<s>", end_tok="</s>", null_tok="<null>",
                 mqa="<mqa>", pmqa="<pmqa>", bmqa="<bmqa>",
                 peentailment="<peentailment>", pbentailment="<pbentailment>",
                 pcoreference="<pcoreference>", bcoreference="<bcoreference>",
                 psentiment="<psentiment>", pgqa="<pgqa>", psqa="<psqa>", gqa="<gqa>", pbqa="<pbqa>",
                 placeholder="<placeholder>", translation="<translation>",
                 a1="(1)", a2="(2)", a3="(3)", a4="(4)", a5="(5)", a6="(6)", a7="(7)", a8="(8)", a9="(9)",
                 passage="<passage>", p1="<p1>", p2="<p2>", p3="<p3>", p4="<p4>", p5="<p5>", p6="<p6>",
                 p7="<p7>", p8="<p8>", p9="<p9>", hypothesis="<h>", question="<q>", metacognition="<mc>",
                 unk_rs="<unk_rs>", aoint_rs="<aoint_rs>", highlighting_rs="<highlighting_rs>",
                 reread_rs="<reread_rs>", summarize_rs="<summarize_rs>", paraphrase_rs="<paraphrase_rs>",
                 num_reading_strategies=6, pad_to_max_length=True, strategy="random",
                 C4_processed_filepath="", num_aux_toks=3):
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

        self.cls_tok_id = self.tokenizer.encode_single(self.cls_tok)
        if len(self.cls_tok_id) != 1 and (isinstance(self.cls_tok_id, list)):
            raise Exception(
                f"The number of ids the start token is encoded into should be one, got {self.cls_tok_id}!")
        else:
            self.cls_tok_id = self.cls_tok_id[0]

        self.sep_tok = sep_tok
        self.mask_tok = mask_tok

        self.start_tok = start_tok
        self.end_tok = end_tok
        self.pad_tok = pad_tok
        self.null_tok = null_tok

        self.mqa = mqa
        self.pmqa = pmqa
        self.bmqa = bmqa
        self.peentailment = peentailment
        self.pbentailment = pbentailment
        self.pcoreference = pcoreference
        self.bcoreference = bcoreference
        self.psentiment = psentiment
        self.pgqa = pgqa
        self.psqa = psqa
        self.gqa = gqa
        self.pbqa = pbqa

        self.placeholder = placeholder
        self.translation = translation

        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        self.a6 = a6
        self.a7 = a7
        self.a8 = a8
        self.a9 = a9

        self.passage = passage
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5
        self.p6 = p6
        self.p7 = p7
        self.p8 = p8
        self.p9 = p9

        self.hypothesis = hypothesis
        self.question = question
        self.metacognition = metacognition

        self.unk_rs = unk_rs
        self.aoint_rs = aoint_rs
        self.highlighting_rs = highlighting_rs
        self.reread_rs = reread_rs
        self.summarize_rs = summarize_rs
        self.paraphrase_rs =paraphrase_rs

        self.num_reading_strategies = num_reading_strategies
        self.num_aux_toks = num_aux_toks
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
            if break_: break # the generator will handle if the batch isn't fully complete...

    def pre_train_c4_dec_only(self):

        # note: also modified for V4, where nm_id_inp and id_inp are combined into one.
        # input string still doesn't contain the auxiliary tokens, as for UNK_rs they are not needed.
        assert self.num_aux_toks == 3, f"The number of auxiliary tokens (num_aux_toks) should be set to 3, got " \
                                       f"{self.num_aux_toks}!"
        break_ = False
        while True:
            counter = 0
            while True:
                if counter == self.batch_size: break
                input_string, input_id, target_input_id, aux_tok_1, aux_tok_2 = self.dataLoaders["C4_nm_pre_train"](mode="dec")
                #print(f"input_id: {input_id} \n\n target_input_id: {target_input_id}")
                if input_id is None or input_string is None: # if here is reached we are done for the current epoch.
                    break_ = True
                    break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]
                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len-len(input_string))]
                    input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len+self.num_aux_toks-len(input_id))]
                    #target_input_id = [pad_tok_id for _ in range(self.num_aux_toks)] + target_input_id # pad all of the auxiliary losses.
                    # the target doesn't include the auxiliary tokens in prediction. Assume max_seq_len_dec length is the output.
                    target_input_id = target_input_id + [pad_tok_id for _ in range(self.seq_len-len(target_input_id))]

                assert len(input_id) == (self.seq_len+self.num_aux_toks)
                assert len(target_input_id) == self.seq_len
                assert len(input_string) == self.seq_len

                input_string = tf.cast(tf.convert_to_tensor(np.asarray(input_string)), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(np.asarray(input_id)), dtype=tf.dtypes.int64)
                target_input_id = tf.cast(tf.convert_to_tensor(np.asarray(target_input_id)), dtype=tf.dtypes.int64)

                counter += 1
                yield input_string, input_id, target_input_id
                #yield input_id, target_input_id, nm_input_id
                # need tf.string... here.
            if break_: break

    def get_generator(self, type: str, shuffle: bool):
        self.shuffle = shuffle
        self.type = type
        generator = None
        if type == "C4_pretrain_enc_dec":
            generator = tf.data.Dataset.from_generator(self.pre_train_c4,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64))
        elif type == "C4_pretrain_enc":
            raise Exception(f"type C4_pretrain_enc is currently not implemented!")
            generator = tf.data.Dataset.from_generator(self.pre_train_c4_enc_only,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64))
        elif type == "C4_pretrain_dec":
            generator = tf.data.Dataset.from_generator(self.pre_train_c4_dec_only,
                                                       output_types=(tf.dtypes.string, # max_seq_len_dec
                                                                     tf.dtypes.int64, # max_seq_len_nm
                                                                     tf.dtypes.int64)) # max_seq_len_dec

        return generator

if __name__ == "__main__":

    config = V4ConfigMediumSize(strategy=None, batch_size=2, loss_object=None, learning_rate=None)

    filepaths = {"C4_nm_pre_train":"/large_data/C4/en/"}

    dloader = MasterDataLoaderTF(filepaths=filepaths, seq_len=config.max_seq_len_dec, batch_size=config.batch_size, tokenizer=config.tokenizer)

    generator = dloader.get_generator(type="C4_pretrain_dec", shuffle=False).batch(1)

    batch_ = 1
    #print("REACH")
    for (inp_str, inp_id, tar_id) in generator:
        print(f"batch: {batch_}")
        #print(f"inp_str: {inp_str.shape} \t inp_str: {inp_str} \n"
        #        f"inp_id.shape: {inp_id.shape} \t inp_id: {inp_id} \n"
        #        f"tar_id.shape: {tar_id.shape} \t tar_id: {tar_id} \n")
        if batch_ == 5: break
        batch_ += 1
    print(f"batch_ counter: {batch_}")