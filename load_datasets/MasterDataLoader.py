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
from load_datasets.question_answering.loadRACE import *

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

            elif key == "RACE_middle_train":
                self.dataLoaders["RACE_middle_train"] = RACEDataLoader(strategy="train",
                                                                       filepath=self.filepaths["RACE_middle_train"],
                                                                       enc_tok=self.enc_tok, dec_tok=self.dec_tok,
                                                                       mlm_tok=self.mlm_tok, lm_tok=self.lm_tok,
                                                                       start_tok=self.start_tok, end_tok=self.end_tok,
                                                                       cls_tok=self.cls_tok,
                                                                       sep_tok=self.sep_tok, mask_tok=self.mask_tok,
                                                                       pad_tok=self.pad_tok, seq_len=self.seq_len,
                                                                       pad=False,
                                                                       a1=self.a1, a2=self.a2, a3=self.a3, a4=self.a4,
                                                                       a5=self.a5, a6=self.a6, a7=self.a7, a8=self.a8,
                                                                       a9=self.a9,
                                                                       passage=self.passage, p1=self.p1, p2=self.p2,
                                                                       p3=self.p3, p4=self.p4, p5=self.p5, p6=self.p6,
                                                                       p7=self.p7, p8=self.p8, p9=self.p9, mqa=self.mqa,
                                                                       pmqa=self.pmqa, bmqa=self.bmqa,
                                                                       peentailment=self.peentailment,
                                                                       pbentailment=self.pbentailment,
                                                                       pcoreference=self.pcoreference,
                                                                       bcoreference=self.bcoreference,
                                                                       psentiment=self.psentiment,
                                                                       pgqa=self.pgqa, psqa=self.psqa, gqa=self.gqa,
                                                                       pbqa=self.pbqa,
                                                                       placeholder=self.placeholder,
                                                                       translation=self.translation,
                                                                       hypothesis=self.hypothesis,
                                                                       question=self.question,
                                                                       metacognition=self.metacognition,
                                                                       unk_rs=self.unk_rs,
                                                                       aoint_rs=self.aoint_rs,
                                                                       highlighting_rs=highlighting_rs,
                                                                       reread_rs=self.reread_rs,
                                                                       summarize_rs=self.summarize_rs,
                                                                       paraphrase_rs=self.paraphrase_rs,
                                                                       tokenizer=self.tokenizer)
            elif key == "RACE_high_train":
                self.dataLoaders["RACE_high_train"] = RACEDataLoader(strategy="train",
                                                                     filepath=self.filepaths["RACE_high_train"],
                                                                     enc_tok=self.enc_tok, dec_tok=self.dec_tok,
                                                                     mlm_tok=self.mlm_tok, lm_tok=self.lm_tok,
                                                                     start_tok=self.start_tok, end_tok=self.end_tok,
                                                                     cls_tok=self.cls_tok,
                                                                     sep_tok=self.sep_tok, mask_tok=self.mask_tok,
                                                                     pad_tok=self.pad_tok, seq_len=self.seq_len,
                                                                     pad=False,
                                                                     a1=self.a1, a2=self.a2, a3=self.a3, a4=self.a4,
                                                                     a5=self.a5, a6=self.a6, a7=self.a7, a8=self.a8,
                                                                     a9=self.a9,
                                                                     passage=self.passage, p1=self.p1, p2=self.p2,
                                                                     p3=self.p3, p4=self.p4, p5=self.p5, p6=self.p6,
                                                                     p7=self.p7, p8=self.p8, p9=self.p9, mqa=self.mqa,
                                                                     pmqa=self.pmqa, bmqa=self.bmqa,
                                                                     peentailment=self.peentailment,
                                                                     pbentailment=self.pbentailment,
                                                                     pcoreference=self.pcoreference,
                                                                     bcoreference=self.bcoreference,
                                                                     psentiment=self.psentiment,
                                                                     pgqa=self.pgqa, psqa=self.psqa, gqa=self.gqa,
                                                                     pbqa=self.pbqa,
                                                                     placeholder=self.placeholder,
                                                                     translation=self.translation,
                                                                     hypothesis=self.hypothesis, question=self.question,
                                                                     metacognition=self.metacognition,
                                                                     unk_rs=self.unk_rs,
                                                                     aoint_rs=self.aoint_rs,
                                                                     highlighting_rs=highlighting_rs,
                                                                     reread_rs=self.reread_rs,
                                                                     summarize_rs=self.summarize_rs,
                                                                     paraphrase_rs=self.paraphrase_rs,
                                                                     tokenizer=self.tokenizer)
            elif key == "RACE_middle_val":
                self.dataLoaders["RACE_middle_val"] = RACEDataLoader(strategy="val",
                                                                     filepath=self.filepaths["RACE_middle_val"],
                                                                     enc_tok=self.enc_tok, dec_tok=self.dec_tok,
                                                                     mlm_tok=self.mlm_tok, lm_tok=self.lm_tok,
                                                                     start_tok=self.start_tok, end_tok=self.end_tok,
                                                                     cls_tok=self.cls_tok,
                                                                     sep_tok=self.sep_tok, mask_tok=self.mask_tok,
                                                                     pad_tok=self.pad_tok, seq_len=self.seq_len,
                                                                     pad=False,
                                                                     a1=self.a1, a2=self.a2, a3=self.a3, a4=self.a4,
                                                                     a5=self.a5, a6=self.a6, a7=self.a7, a8=self.a8,
                                                                     a9=self.a9,
                                                                     passage=self.passage, p1=self.p1, p2=self.p2,
                                                                     p3=self.p3, p4=self.p4, p5=self.p5, p6=self.p6,
                                                                     p7=self.p7, p8=self.p8, p9=self.p9, mqa=self.mqa,
                                                                     pmqa=self.pmqa, bmqa=self.bmqa,
                                                                     peentailment=self.peentailment,
                                                                     pbentailment=self.pbentailment,
                                                                     pcoreference=self.pcoreference,
                                                                     bcoreference=self.bcoreference,
                                                                     psentiment=self.psentiment,
                                                                     pgqa=self.pgqa, psqa=self.psqa, gqa=self.gqa,
                                                                     pbqa=self.pbqa,
                                                                     placeholder=self.placeholder,
                                                                     translation=self.translation,
                                                                     hypothesis=self.hypothesis, question=self.question,
                                                                     metacognition=self.metacognition,
                                                                     unk_rs=self.unk_rs,
                                                                     aoint_rs=self.aoint_rs,
                                                                     highlighting_rs=highlighting_rs,
                                                                     reread_rs=self.reread_rs,
                                                                     summarize_rs=self.summarize_rs,
                                                                     paraphrase_rs=self.paraphrase_rs,
                                                                     tokenizer=self.tokenizer)
            elif key == "RACE_high_val":
                self.dataLoaders["RACE_high_val"] = RACEDataLoader(strategy="val",
                                                                   filepath=self.filepaths["RACE_high_val"],
                                                                   enc_tok=self.enc_tok, dec_tok=self.dec_tok,
                                                                   mlm_tok=self.mlm_tok, lm_tok=self.lm_tok,
                                                                   start_tok=self.start_tok, end_tok=self.end_tok,
                                                                   cls_tok=self.cls_tok,
                                                                   sep_tok=self.sep_tok, mask_tok=self.mask_tok,
                                                                   pad_tok=self.pad_tok, seq_len=self.seq_len,
                                                                   pad=False,
                                                                   a1=self.a1, a2=self.a2, a3=self.a3, a4=self.a4,
                                                                   a5=self.a5, a6=self.a6, a7=self.a7, a8=self.a8,
                                                                   a9=self.a9,
                                                                   passage=self.passage, p1=self.p1, p2=self.p2,
                                                                   p3=self.p3, p4=self.p4, p5=self.p5, p6=self.p6,
                                                                   p7=self.p7, p8=self.p8, p9=self.p9, mqa=self.mqa,
                                                                   pmqa=self.pmqa, bmqa=self.bmqa,
                                                                   peentailment=self.peentailment,
                                                                   pbentailment=self.pbentailment,
                                                                   pcoreference=self.pcoreference,
                                                                   bcoreference=self.bcoreference,
                                                                   psentiment=self.psentiment,
                                                                   pgqa=self.pgqa, psqa=self.psqa, gqa=self.gqa,
                                                                   pbqa=self.pbqa,
                                                                   placeholder=self.placeholder,
                                                                   translation=self.translation,
                                                                   hypothesis=self.hypothesis, question=self.question,
                                                                   metacognition=self.metacognition, unk_rs=self.unk_rs,
                                                                   aoint_rs=self.aoint_rs,
                                                                   highlighting_rs=highlighting_rs,
                                                                   reread_rs=self.reread_rs,
                                                                   summarize_rs=self.summarize_rs,
                                                                   paraphrase_rs=self.paraphrase_rs,
                                                                   tokenizer=self.tokenizer)
            elif key == "RACE_middle_test":
                self.dataLoaders["RACE_middle_test"] = RACEDataLoader(strategy="test",
                                                                      filepath=self.filepaths["RACE_middle_test"],
                                                                      enc_tok=self.enc_tok, dec_tok=self.dec_tok,
                                                                      mlm_tok=self.mlm_tok, lm_tok=self.lm_tok,
                                                                      start_tok=self.start_tok, end_tok=self.end_tok,
                                                                      cls_tok=self.cls_tok,
                                                                      sep_tok=self.sep_tok, mask_tok=self.mask_tok,
                                                                      pad_tok=self.pad_tok, seq_len=self.seq_len,
                                                                      pad=False,
                                                                      a1=self.a1, a2=self.a2, a3=self.a3, a4=self.a4,
                                                                      a5=self.a5, a6=self.a6, a7=self.a7, a8=self.a8,
                                                                      a9=self.a9,
                                                                      passage=self.passage, p1=self.p1, p2=self.p2,
                                                                      p3=self.p3, p4=self.p4, p5=self.p5, p6=self.p6,
                                                                      p7=self.p7, p8=self.p8, p9=self.p9, mqa=self.mqa,
                                                                      pmqa=self.pmqa, bmqa=self.bmqa,
                                                                      peentailment=self.peentailment,
                                                                      pbentailment=self.pbentailment,
                                                                      pcoreference=self.pcoreference,
                                                                      bcoreference=self.bcoreference,
                                                                      psentiment=self.psentiment,
                                                                      pgqa=self.pgqa, psqa=self.psqa, gqa=self.gqa,
                                                                      pbqa=self.pbqa,
                                                                      placeholder=self.placeholder,
                                                                      translation=self.translation,
                                                                      hypothesis=self.hypothesis,
                                                                      question=self.question,
                                                                      metacognition=self.metacognition,
                                                                      unk_rs=self.unk_rs,
                                                                      aoint_rs=self.aoint_rs,
                                                                      highlighting_rs=highlighting_rs,
                                                                      reread_rs=self.reread_rs,
                                                                      summarize_rs=self.summarize_rs,
                                                                      paraphrase_rs=self.paraphrase_rs,
                                                                      tokenizer=self.tokenizer)
            elif key == "RACE_high_test":
                self.dataLoaders["RACE_high_test"] = RACEDataLoader(strategy="test",
                                                                    filepath=self.filepaths["RACE_high_test"],
                                                                    enc_tok=self.enc_tok, dec_tok=self.dec_tok,
                                                                    mlm_tok=self.mlm_tok, lm_tok=self.lm_tok,
                                                                    start_tok=self.start_tok, end_tok=self.end_tok,
                                                                    cls_tok=self.cls_tok,
                                                                    sep_tok=self.sep_tok, mask_tok=self.mask_tok,
                                                                    pad_tok=self.pad_tok, seq_len=self.seq_len,
                                                                    pad=False,
                                                                    a1=self.a1, a2=self.a2, a3=self.a3, a4=self.a4,
                                                                    a5=self.a5, a6=self.a6, a7=self.a7, a8=self.a8,
                                                                    a9=self.a9,
                                                                    passage=self.passage, p1=self.p1, p2=self.p2,
                                                                    p3=self.p3, p4=self.p4, p5=self.p5, p6=self.p6,
                                                                    p7=self.p7, p8=self.p8, p9=self.p9, mqa=self.mqa,
                                                                    pmqa=self.pmqa, bmqa=self.bmqa,
                                                                    peentailment=self.peentailment,
                                                                    pbentailment=self.pbentailment,
                                                                    pcoreference=self.pcoreference,
                                                                    bcoreference=self.bcoreference,
                                                                    psentiment=self.psentiment,
                                                                    pgqa=self.pgqa, psqa=self.psqa, gqa=self.gqa,
                                                                    pbqa=self.pbqa,
                                                                    placeholder=self.placeholder,
                                                                    translation=self.translation,
                                                                    hypothesis=self.hypothesis, question=self.question,
                                                                    metacognition=self.metacognition,
                                                                    unk_rs=self.unk_rs,
                                                                    aoint_rs=self.aoint_rs,
                                                                    highlighting_rs=highlighting_rs,
                                                                    reread_rs=self.reread_rs,
                                                                    summarize_rs=self.summarize_rs,
                                                                    paraphrase_rs=self.paraphrase_rs,
                                                                    tokenizer=self.tokenizer)

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

    def _combined_race_helper(self):
        if self.type == "RACE_combined_train":
            assert ("RACE_middle_train" in self.dataLoaders.keys() and "RACE_high_train" in self.dataLoaders.keys()) or \
                   ("RACE_middle_train_label" in self.dataLoaders.keys() and "RACE_high_train_label" in self.dataLoaders.keys()), \
                    f"One of RACE_middle_train and/or RACE_high_train is not in the dataLoaders dictionary of dataloaders!" \
                    f"Or alternatively the updated _label at the end of the already mentioned keys."
            key1 = "RACE_middle_train" if "RACE_middle_train" in self.dataLoaders.keys() else "RACE_middle_train_label"
            key2 = "RACE_high_train" if "RACE_high_train" in self.dataLoaders.keys() else "RACE_high_train_label"
            # print(key1)
            # print(key2)

            # depending on the key names
            gen1 = None
            gen2 = None
            if key1 == "RACE_middle_train":
                gen1 = self.dataLoaders["RACE_middle_train"](mode="default_generate")
            else:
                gen1 = self.dataLoaders["RACE_middle_train_label"](mode="default_label")
            if key2 == "RACE_high_train":
                gen2 = self.dataLoaders["RACE_high_train"](mode="default_generate")
            else:
                gen2 = self.dataLoaders["RACE_high_train_label"](mode="default_label")

            stop_gen1 = False
            stop_gen2 = False
            while True:

                input_string, input_id, label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = None, None, None, None, None, None, None
                curr_gen = ''
                stop_all = True
                if random.random() > 0.5:
                    if not stop_gen1:  # do gen1 first.
                        input_string, input_id, label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen1)
                        curr_gen = "gen1"
                        stop_all = False
                    elif not stop_gen2:  # do gen2 if gen1 is finished.
                        curr_gen = "gen2"
                        input_string, input_id, label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen2)
                        stop_all = False
                else:
                    if not stop_gen2:  # do gen2 first.
                        input_string, input_id, label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen2)
                        curr_gen = "gen2"
                        stop_all = False
                    elif not stop_gen1:  # do gen2 if gen1 is finished.
                        input_string, input_id, label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen1)
                        curr_gen = "gen1"
                        stop_all = False

                if stop_all:
                    yield input_string, input_id, label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2  # will be all None
                    break
                if input_string is None and curr_gen == "gen1":  # gen1 is finished, so don't output None and continue, also set stop_gen1 to True so we don't process it again.
                    stop_gen1 = True
                    continue
                if input_string is None and curr_gen == "gen2":
                    stop_gen2 = True
                    continue
                # print(input_string, input_id, label, aux_tok_1, aux_tok_2)
                yield input_string, input_id, label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2

        elif self.type == "RACE_combined_val":
            assert ("RACE_middle_val" in self.dataLoaders.keys() and "RACE_high_val" in self.dataLoaders.keys()) or \
                   (
                               "RACE_middle_val_label" in self.dataLoaders.keys() and "RACE_high_val_label" in self.dataLoaders.keys()), \
                f"One of RACE_middle_val and/or RACE_high_val is not in the dataLoaders dictionary of dataloaders!" \
                f"Or alternatively the updated _label at the end of the already mentioned keys."
            key1 = "RACE_middle_val" if "RACE_middle_val" in self.dataLoaders.keys() else "RACE_middle_val_label"
            key2 = "RACE_high_val" if "RACE_high_val" in self.dataLoaders.keys() else "RACE_high_val_label"

            # depending on the key names
            gen1 = None
            gen2 = None
            if key1 == "RACE_middle_val":
                gen1 = self.dataLoaders["RACE_middle_val"](mode="default_generate")
            else:
                gen1 = self.dataLoaders["RACE_middle_val_label"](mode="default_label")
            if key2 == "RACE_high_val":
                gen2 = self.dataLoaders["RACE_high_val"](mode="default_generate") #TODO: error here gen1 was here, which is obviously incorrect.
            else:
                gen2 = self.dataLoaders["RACE_high_val_label"](mode="default_label")

            stop_gen1 = False
            stop_gen2 = False
            while True:

                input_string, input_id, label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = None, None, None, None, None, None, None
                curr_gen = ''
                stop_all = True
                if random.random() > 0.5:
                    if not stop_gen1:  # do gen1 first.
                        input_string, input_id, label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen1)
                        curr_gen = "gen1"
                        stop_all = False
                    elif not stop_gen2:  # do gen2 if gen1 is finished.
                        curr_gen = "gen2"
                        input_string, input_id, label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen2)
                        stop_all = False
                else:
                    if not stop_gen2:  # do gen2 first.
                        input_string, input_id, label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen2)
                        curr_gen = "gen2"
                        stop_all = False
                    elif not stop_gen1:  # do gen2 if gen1 is finished.
                        input_string, input_id, label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen1)
                        curr_gen = "gen1"
                        stop_all = False

                if stop_all:
                    yield input_string, input_id, label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2  # will be all None
                    break
                if input_string is None and curr_gen == "gen1":  # gen1 is finished, so don't output None and continue, also set stop_gen1 to True so we don't process it again.
                    stop_gen1 = True
                    continue
                if input_string is None and curr_gen == "gen2":
                    stop_gen2 = True
                    continue

                yield input_string, input_id, label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2
        elif self.type == "RACE_combined_test": raise Exception(f"Not implemented (RACE_combined_test)!")

    def get_race_dataloader(self):

        mini_generator = None
        mode_ = None
        if self.type in ["RACE_middle_train", "RACE_middle_val", "RACE_high_train", "RACE_high_val"]:
            mini_generator = self.dataLoaders[self.type](mode="default_generate")
            mode_ = "default_generate"
        elif self.type in ["RACE_middle_test", "RACE_high_test"]:
            mini_generator = self.dataLoaders[self.type](mode="test")
            mode_ = "test"
        else: # type == RACE_combined_train, RACE_combined_val, RACE_combined_test or label version.
            mini_generator = self._combined_race_helper()
            if self.type == "RACE_combined_val" or self.type == "RACE_combined_val_label":
                key1 = "RACE_middle_val" if "RACE_middle_val" in self.dataLoaders.keys() else "RACE_middle_val_label"
                key2 = "RACE_high_val" if "RACE_high_val" in self.dataLoaders.keys() else "RACE_high_val_label"
                if key1 == "RACE_middle_val":
                    mode_ = "default_generate"
                else:
                    mode_ = "default_label"
            elif self.type == "RACE_combined_train" or self.type == "RACE_combined_train_label":
                key1 = "RACE_middle_train" if "RACE_middle_train" in self.dataLoaders.keys() else "RACE_middle_train_label"
                key2 = "RACE_high_train" if "RACE_high_train" in self.dataLoaders.keys() else "RACE_high_train_label"
                if key1 == "RACE_middle_train":
                    mode_ = "default_generate"
                else:
                    mode_ = "default_label"
        if mode_ == "default_generate":
            while True:
                break_ = False
                #try: # Safety so whole training isn't stopped for one error. No error should be reached.
                input_string, input_id, label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(mini_generator)  # label will be a list of one element, the correct answer...
                #except RuntimeError as e:  # stopIteration
                #    print(f"Runtime Error: {e} \n Continuing as per normal as generator has nothing left to generate!")
                #    break_ = True
                #except Exception as e:
                #    print(f"Unknown error with generator, continuing! {e}")
                #    break_ = True

                if break_: break
                if input_string is None or input_id is None: break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]
                    if not isinstance(label, int):  # integer only, not a list of integers. (not of this)
                        label = label + [pad_tok_id for _ in range(self.seq_len - len(label))]  #
                    else: raise Exception(f"label should not be of type int, but is instead a list of integers")

                input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                label_id = tf.cast(tf.convert_to_tensor(label), dtype=tf.dtypes.int64)
                aoint_indices = tf.cast(tf.convert_to_tensor(aoint_indices), dtype=tf.dtypes.int64)
                sample_weights = tf.cast(tf.convert_to_tensor(sample_weights), dtype=tf.dtypes.int64)

                yield input_string, input_id, label_id, aoint_indices, sample_weights
        elif mode_ == "default_label": raise Exception("Not implemented yet (default_label)")
        elif mode_ == "test":
            while True:
                break_ = False
                #try: # Safety so whole training isn't stopped for one error. No error should be reached.
                input_string, input_id, aoint_indices, all_labels, correct_ao, aux_tok_1, aux_tok_2 = next(mini_generator)  # label will be a list of one element, the correct answer...
                #except RuntimeError as e:  # stopIteration
                ##    print(f"Runtime Error: {e} \n Continuing as per normal as generator has nothing left to generate!")
                 #   break_ = True
                #except Exception as e:
                #    print(f"Unknown error with generator, continuing! {e}")
                #    break_ = True

                if break_: break
                if input_string is None or input_id is None: break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]

                input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                aoint_indices = tf.cast(tf.convert_to_tensor(aoint_indices), dtype=tf.dtypes.int64)
                all_labels = tf.cast(tf.convert_to_tensor(np.asarray(all_labels)), dtype=tf.dtypes.string)
                correct_ao = tf.cast(tf.convert_to_tensor(np.asarray(correct_ao)), dtype=tf.dtypes.string)

                yield input_string, input_id, all_labels, correct_ao, aoint_indices
        else: raise Exception(f"Invalid mode_!")

    def get_generator(self, type: str, shuffle: bool, race_label_bool=False):

        self.shuffle = shuffle
        self.type = type

        # race specific variables
        #self.race_label_bool = race_label_bool # True is label only is generated as an answer. False if the whole answer is to be generated.

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
        elif type == "RACE_middle_train" or type == "RACE_middle_val" or \
                type == "RACE_high_train" or type == "RACE_high_val":
            generator = tf.data.Dataset.from_generator(self.get_race_dataloader,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64))
        elif type == "RACE_middle_test" or type == "RACE_high_test":
            generator = tf.data.Dataset.from_generator(self.get_race_dataloader,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.string,
                                                                     tf.dtypes.string,
                                                                     tf.dtypes.int64))
        elif type == "RACE_combined_train" or type == "RACE_combined_val":
            generator = tf.data.Dataset.from_generator(self.get_race_dataloader,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64))
        elif type == "RACE_combined_test":
            generator = tf.data.Dataset.from_generator(self.get_race_dataloader,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.string,
                                                                     tf.dtypes.string,
                                                                     tf.dtypes.int64))

        return generator

if __name__ == "__main__":

    config = V4ConfigMediumSize(strategy=None, batch_size=2, loss_object=None, learning_rate=None)

    #filepaths = {"C4_nm_pre_train":"/large_data/C4/en/"}
    filepaths = {"RACE_high_train": "/large_data/RACE/train/high/",
                 "RACE_high_val": "/large_data/RACE/dev/high/",
                 "RACE_middle_train": "/large_data/RACE/train/middle/",
                 "RACE_middle_val": "/large_data/RACE/dev/middle/"}

    dloader = MasterDataLoaderTF(filepaths=filepaths, seq_len=config.max_seq_len_dec, batch_size=config.batch_size, tokenizer=config.tokenizer)


    #generator = dloader.get_generator(type="C4_pretrain_dec", shuffle=False).batch(1)
    generator = dloader.get_generator("RACE_combined_train", False).batch(2)

    batch_ = 1
    for (input_string, input_id, label_id, aoint_indices, sample_weights) in generator:
        print(f"batch: {batch_}")
        print(f"input_string: {input_string.shape} \t inp_str: {input_string} \n"
              f"input_id.shape: {input_id.shape} \t inp_id: {None} \n"
              f"label_id.shape: {label_id.shape} \t tar_id: {None} \n"
              f"aoint_indices.shape: {aoint_indices.shape} \t aoint_indices: {aoint_indices}"
              f"sample_weights.shape: {sample_weights} \t sample_weights: {sample_weights}")
        if batch_ == 2: break
        batch_ += 1
    print(f"batch_ counter: {batch_}")

    '''
    # this is testing code for c4
    batch_ = 1
    for (inp_str, inp_id, tar_id) in generator:
        print(f"batch: {batch_}")
        print(f"inp_str: {inp_str.shape} \t inp_str: {None} \n"
                f"inp_id.shape: {inp_id.shape} \t inp_id: {None} \n"
                f"tar_id.shape: {tar_id.shape} \t tar_id: {None} \n")
        if batch_ == 5: break
        batch_ += 1
    print(f"batch_ counter: {batch_}")
    '''