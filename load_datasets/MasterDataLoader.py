'''
File name: MasterDataLoaderTF.py
Author: Kobe Knowles
Date created: 26/08/21
Data last modified: 26/08/21
Python Version: 3.8
Tensorflow version: 2.5
'''

#import os
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
from load_datasets.question_answering.loadSQuAD import *
from load_datasets.question_answering.BoolQ import *
from load_datasets.question_answering.NarrativeQA import *
from load_datasets.question_answering.OBQA import *
from load_datasets.question_answering.ARC import *
from load_datasets.question_answering.MCTest import *
from load_datasets.question_answering.CQA import CommonsenseQADataLoader
from load_datasets.question_answering.PIQA import PIQADataLoader
from load_datasets.question_answering.SIQA import SIQADataLoader
from load_datasets.question_answering.WG import WGDataLoader
from load_datasets.question_answering.BoolQCS import BoolQCSDataLoader

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
            elif key == "NarrativeQA_train":
                self.dataLoaders["NarrativeQA_train"] = NarrativeQADataLoader(strategy="train",
                                                                   filepath=self.filepaths["NarrativeQA_train"],
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
                                                                   highlighting_rs=self.highlighting_rs,
                                                                   reread_rs=self.reread_rs,
                                                                   summarize_rs=self.summarize_rs,
                                                                   paraphrase_rs=self.paraphrase_rs,
                                                                   tokenizer=self.tokenizer)
            elif key == "NarrativeQA_val":
                self.dataLoaders["NarrativeQA_val"] = NarrativeQADataLoader(strategy="val",
                                                                   filepath=self.filepaths["NarrativeQA_val"],
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
                                                                   highlighting_rs=self.highlighting_rs,
                                                                   reread_rs=self.reread_rs,
                                                                   summarize_rs=self.summarize_rs,
                                                                   paraphrase_rs=self.paraphrase_rs,
                                                                   tokenizer=self.tokenizer)
            elif key == "NarrativeQA_test":
                self.dataLoaders["NarrativeQA_test"] = NarrativeQADataLoader(strategy="all",
                                                                   filepath=self.filepaths["NarrativeQA_test"],
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
                                                                   highlighting_rs=self.highlighting_rs,
                                                                   reread_rs=self.reread_rs,
                                                                   summarize_rs=self.summarize_rs,
                                                                   paraphrase_rs=self.paraphrase_rs,
                                                                   tokenizer=self.tokenizer)
            elif key == "NarrativeQA_val_test":
                self.dataLoaders["NarrativeQA_val_test"] = NarrativeQADataLoader(strategy="val",
                                                                   filepath=self.filepaths["NarrativeQA_val_test"],
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
                                                                   highlighting_rs=self.highlighting_rs,
                                                                   reread_rs=self.reread_rs,
                                                                   summarize_rs=self.summarize_rs,
                                                                   paraphrase_rs=self.paraphrase_rs,
                                                                   tokenizer=self.tokenizer)
            elif key == "BoolQ_train":
                self.dataLoaders["BoolQ_train"] = BoolQDataLoader(strategy="train",
                                                                   filepath=self.filepaths["BoolQ_train"],
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
                                                                   highlighting_rs=self.highlighting_rs,
                                                                   reread_rs=self.reread_rs,
                                                                   summarize_rs=self.summarize_rs,
                                                                   paraphrase_rs=self.paraphrase_rs,
                                                                   tokenizer=self.tokenizer)
            elif key == "BoolQ_val":
                self.dataLoaders["BoolQ_val"] = BoolQDataLoader(strategy="val",
                                                                   filepath=self.filepaths["BoolQ_val"],
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
                                                                   highlighting_rs=self.highlighting_rs,
                                                                   reread_rs=self.reread_rs,
                                                                   summarize_rs=self.summarize_rs,
                                                                   paraphrase_rs=self.paraphrase_rs,
                                                                   tokenizer=self.tokenizer)
            elif key == "BoolQ_test":
                self.dataLoaders["BoolQ_test"] = BoolQDataLoader(strategy="test",
                                                                   filepath=self.filepaths["BoolQ_test"],
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
                                                                   highlighting_rs=self.highlighting_rs,
                                                                   reread_rs=self.reread_rs,
                                                                   summarize_rs=self.summarize_rs,
                                                                   paraphrase_rs=self.paraphrase_rs,
                                                                   tokenizer=self.tokenizer)
            elif key == "BoolQCS_test":
                self.dataLoaders["BoolQCS_test"] = BoolQCSDataLoader(strategy="test",
                                                                   filepath=self.filepaths["BoolQCS_test"],
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
                                                                   highlighting_rs=self.highlighting_rs,
                                                                   reread_rs=self.reread_rs,
                                                                   summarize_rs=self.summarize_rs,
                                                                   paraphrase_rs=self.paraphrase_rs,
                                                                   tokenizer=self.tokenizer)
            elif key == "BoolQ_test_no_answer":
                self.dataLoaders["BoolQ_test_no_answer"] = BoolQDataLoader(strategy="test_no_answer",
                                                                   filepath=self.filepaths["BoolQ_test_no_answer"],
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
                                                                   highlighting_rs=self.highlighting_rs,
                                                                   reread_rs=self.reread_rs,
                                                                   summarize_rs=self.summarize_rs,
                                                                   paraphrase_rs=self.paraphrase_rs,
                                                                   tokenizer=self.tokenizer)
            elif key == "SQuAD_train_default":
                self.dataLoaders["SQuAD_train_default"] = SQuADDataLoader(strategy="train",
                                                                           filepath=self.filepaths["SQuAD_train_default"],
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
            elif key == "SQuAD_val_default":
                self.dataLoaders["SQuAD_val_default"] = SQuADDataLoader(strategy="val",
                                                                           filepath=self.filepaths["SQuAD_val_default"],
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
            elif key == "SQuAD_test_default":
                self.dataLoaders["SQuAD_test_default"] = SQuADDataLoader(strategy="test",
                                                                           filepath=self.filepaths["SQuAD_test_default"],
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

            elif key == "CQA_train":
                self.dataLoaders["CQA_train"] = CommonsenseQADataLoader(strategy="train",
                                                                       filepath=self.filepaths["CQA_train"],
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
            elif key == "CQA_val":
                self.dataLoaders["CQA_val"] = CommonsenseQADataLoader(strategy="val",
                                                                       filepath=self.filepaths["CQA_val"],
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
            elif key == "CQA_test":
                self.dataLoaders["CQA_test"] = CommonsenseQADataLoader(strategy="test",
                                                                       filepath=self.filepaths["CQA_test"],
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

            elif key == "WG_train":
                self.dataLoaders["WG_train"] = WGDataLoader(strategy="train",
                                                                       filepath=self.filepaths["WG_train"],
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
            elif key == "WG_val":
                self.dataLoaders["WG_val"] = WGDataLoader(strategy="val",
                                                                       filepath=self.filepaths["WG_val"],
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
            elif key == "WG_test":
                self.dataLoaders["WG_test"] = WGDataLoader(strategy="test",
                                                                       filepath=self.filepaths["WG_test"],
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

            elif key == "SIQA_train":
                self.dataLoaders["SIQA_train"] = SIQADataLoader(strategy="train",
                                                                       filepath=self.filepaths["SIQA_train"],
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
            elif key == "SIQA_val":
                self.dataLoaders["SIQA_val"] = SIQADataLoader(strategy="val",
                                                                       filepath=self.filepaths["SIQA_val"],
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
            elif key == "SIQA_test":
                self.dataLoaders["SIQA_test"] = SIQADataLoader(strategy="test",
                                                                       filepath=self.filepaths["SIQA_test"],
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

            elif key == "PIQA_train":
                self.dataLoaders["PIQA_train"] = PIQADataLoader(strategy="train",
                                                                       filepath=self.filepaths["PIQA_train"],
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
            elif key == "PIQA_val":
                self.dataLoaders["PIQA_val"] = PIQADataLoader(strategy="val",
                                                                       filepath=self.filepaths["PIQA_val"],
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
            elif key == "PIQA_test":
                self.dataLoaders["PIQA_test"] = PIQADataLoader(strategy="test",
                                                                       filepath=self.filepaths["PIQA_test"],
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

            elif key == "OBQA_train":
                self.dataLoaders["OBQA_train"] = OpenBookQADataLoader(strategy="train",
                                                                       filepath=self.filepaths["OBQA_train"],
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
            elif key == "OBQA_val":
                self.dataLoaders["OBQA_val"] = OpenBookQADataLoader(strategy="val",
                                                                       filepath=self.filepaths["OBQA_val"],
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
            elif key == "OBQA_test":
                self.dataLoaders["OBQA_test"] = OpenBookQADataLoader(strategy="test",
                                                                       filepath=self.filepaths["OBQA_test"],
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
            elif key == "ARC_test_easy" or key == "ARC_test_challenge" \
                    or key == "ARC_train_easy" or key == "ARC_train_challenge" \
                    or key == "ARC_val_easy" or key == "ARC_val_challenge":
                self.dataLoaders[key] = ARCDataLoader(strategy="doesn't even matter",
                                                       filepath=self.filepaths[key],
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

            elif key == "MCTest_train" or key == "MCTest_val" or key == "MCTest_test":
                self.dataLoaders[key] = MCTestDataLoader(strategy="doesn't even matter",
                                                       filepath=self.filepaths[key],
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
        raise Exception(f"Note: need to modernize this similar to decoder only version.")
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
        if self.type == "RACE_combined_train" or self.type == "RACE_combined_train_label":
            assert ("RACE_middle_train" in self.dataLoaders.keys() and "RACE_high_train" in self.dataLoaders.keys()), \
                    f"One of RACE_middle_train and/or RACE_high_train is not in the dataLoaders dictionary of dataloaders!"
            #key1 = "RACE_middle_train" if "RACE_middle_train" in self.dataLoaders.keys() else "RACE_middle_train_label"
            #key2 = "RACE_high_train" if "RACE_high_train" in self.dataLoaders.keys() else "RACE_high_train_label"
            # print(key1)
            # print(key2)

            # depending on the key names
            gen1 = None
            gen2 = None

            if self.type == "RACE_combined_train":
                gen1 = self.dataLoaders["RACE_middle_train"](mode="default_generate", shuffle=self.shuffle)
                gen2 = self.dataLoaders["RACE_high_train"](mode="default_generate", shuffle=self.shuffle)
            else: # RACE_combined_train_label
                gen1 = self.dataLoaders["RACE_middle_train"](mode="default_label", shuffle=self.shuffle)
                gen2 = self.dataLoaders["RACE_high_train"](mode="default_label", shuffle=self.shuffle)

            '''
            if key1 == "RACE_middle_train":
                gen1 = self.dataLoaders["RACE_middle_train"](mode="default_generate")
            else:
                gen1 = self.dataLoaders["RACE_middle_train_label"](mode="default_label")
            if key2 == "RACE_high_train":
                gen2 = self.dataLoaders["RACE_high_train"](mode="default_generate")
            else:
                gen2 = self.dataLoaders["RACE_high_train_label"](mode="default_label")
            '''

            stop_gen1 = False
            stop_gen2 = False
            while True:

                input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = None, None, None, None, None, None, None, None
                curr_gen = ''
                stop_all = True
                if random.random() > 0.5:
                    if not stop_gen1:  # do gen1 first.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen1)
                        curr_gen = "gen1"
                        stop_all = False
                    elif not stop_gen2:  # do gen2 if gen1 is finished.
                        curr_gen = "gen2"
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen2)
                        stop_all = False
                else:
                    if not stop_gen2:  # do gen2 first.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen2)
                        curr_gen = "gen2"
                        stop_all = False
                    elif not stop_gen1:  # do gen2 if gen1 is finished.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen1)
                        curr_gen = "gen1"
                        stop_all = False

                if stop_all:
                    yield input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2  # will be all None
                    break
                if input_string is None and curr_gen == "gen1":  # gen1 is finished, so don't output None and continue, also set stop_gen1 to True so we don't process it again.
                    stop_gen1 = True
                    continue
                if input_string is None and curr_gen == "gen2":
                    stop_gen2 = True
                    continue
                # print(input_string, input_id, label, aux_tok_1, aux_tok_2)
                yield input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2

        elif self.type == "RACE_combined_val" or self.type == "RACE_combined_val_label":
            assert ("RACE_middle_val" in self.dataLoaders.keys() and "RACE_high_val" in self.dataLoaders.keys()), \
                f"One of RACE_middle_val and/or RACE_high_val is not in the dataLoaders dictionary of dataloaders!"
            #key1 = "RACE_middle_val" if "RACE_middle_val" in self.dataLoaders.keys() else "RACE_middle_val_label"
            #key2 = "RACE_high_val" if "RACE_high_val" in self.dataLoaders.keys() else "RACE_high_val_label"

            # depending on the key names
            gen1 = None
            gen2 = None

            if self.type == "RACE_combined_val":
                gen1 = self.dataLoaders["RACE_middle_val"](mode="default_generate")
                gen2 = self.dataLoaders["RACE_high_val"](mode="default_generate")
            else: # RACE_combined_train_label
                gen1 = self.dataLoaders["RACE_middle_val"](mode="default_label")
                gen2 = self.dataLoaders["RACE_high_val"](mode="default_label")

            '''
            if key1 == "RACE_middle_val":
                gen1 = self.dataLoaders["RACE_middle_val"](mode="default_generate")
            else:
                gen1 = self.dataLoaders["RACE_middle_val_label"](mode="default_label")
            if key2 == "RACE_high_val":
                gen2 = self.dataLoaders["RACE_high_val"](mode="default_generate") #TODO: error here gen1 was here, which is obviously incorrect.
            else:
                gen2 = self.dataLoaders["RACE_high_val_label"](mode="default_label")
            '''

            stop_gen1 = False
            stop_gen2 = False
            while True:

                input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = None, None, None, None, None, None, None, None
                curr_gen = ''
                stop_all = True
                if random.random() > 0.5:
                    if not stop_gen1:  # do gen1 first.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen1)
                        curr_gen = "gen1"
                        stop_all = False
                    elif not stop_gen2:  # do gen2 if gen1 is finished.
                        curr_gen = "gen2"
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen2)
                        stop_all = False
                else:
                    if not stop_gen2:  # do gen2 first.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen2)
                        curr_gen = "gen2"
                        stop_all = False
                    elif not stop_gen1:  # do gen2 if gen1 is finished.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen1)
                        curr_gen = "gen1"
                        stop_all = False

                if stop_all:
                    yield input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2  # will be all None
                    break
                if input_string is None and curr_gen == "gen1":  # gen1 is finished, so don't output None and continue, also set stop_gen1 to True so we don't process it again.
                    stop_gen1 = True
                    continue
                if input_string is None and curr_gen == "gen2":
                    stop_gen2 = True
                    continue

                yield input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2
        elif self.type == "RACE_combined_test": raise Exception(f"Not implemented (RACE_combined_test)!")

    def _combined_race_helper2(self):
        if self.type3 == "RACE_combined_train" or self.type3 == "RACE_combined_train_label":
            assert ("RACE_middle_train" in self.dataLoaders.keys() and "RACE_high_train" in self.dataLoaders.keys()), \
                    f"One of RACE_middle_train and/or RACE_high_train is not in the dataLoaders dictionary of dataloaders!"
            #key1 = "RACE_middle_train" if "RACE_middle_train" in self.dataLoaders.keys() else "RACE_middle_train_label"
            #key2 = "RACE_high_train" if "RACE_high_train" in self.dataLoaders.keys() else "RACE_high_train_label"
            # print(key1)
            # print(key2)

            # depending on the key names
            gen1 = None
            gen2 = None

            if self.type3 == "RACE_combined_train":
                gen1 = self.dataLoaders["RACE_middle_train"](mode="default_generate", shuffle=self.shuffle)
                gen2 = self.dataLoaders["RACE_high_train"](mode="default_generate", shuffle=self.shuffle)
            else: # RACE_combined_train_label
                gen1 = self.dataLoaders["RACE_middle_train"](mode="default_label", shuffle=self.shuffle)
                gen2 = self.dataLoaders["RACE_high_train"](mode="default_label", shuffle=self.shuffle)

            '''
            if key1 == "RACE_middle_train":
                gen1 = self.dataLoaders["RACE_middle_train"](mode="default_generate")
            else:
                gen1 = self.dataLoaders["RACE_middle_train_label"](mode="default_label")
            if key2 == "RACE_high_train":
                gen2 = self.dataLoaders["RACE_high_train"](mode="default_generate")
            else:
                gen2 = self.dataLoaders["RACE_high_train_label"](mode="default_label")
            '''

            stop_gen1 = False
            stop_gen2 = False
            while True:

                input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = None, None, None, None, None, None, None, None
                curr_gen = ''
                stop_all = True
                if random.random() > 0.5:
                    if not stop_gen1:  # do gen1 first.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen1)
                        curr_gen = "gen1"
                        stop_all = False
                    elif not stop_gen2:  # do gen2 if gen1 is finished.
                        curr_gen = "gen2"
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen2)
                        stop_all = False
                else:
                    if not stop_gen2:  # do gen2 first.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen2)
                        curr_gen = "gen2"
                        stop_all = False
                    elif not stop_gen1:  # do gen2 if gen1 is finished.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen1)
                        curr_gen = "gen1"
                        stop_all = False

                if stop_all:
                    yield input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2  # will be all None
                    break
                if input_string is None and curr_gen == "gen1":  # gen1 is finished, so don't output None and continue, also set stop_gen1 to True so we don't process it again.
                    stop_gen1 = True
                    continue
                if input_string is None and curr_gen == "gen2":
                    stop_gen2 = True
                    continue
                # print(input_string, input_id, label, aux_tok_1, aux_tok_2)
                yield input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2

        elif self.type3 == "RACE_combined_val" or self.type3 == "RACE_combined_val_label":
            assert ("RACE_middle_val" in self.dataLoaders.keys() and "RACE_high_val" in self.dataLoaders.keys()), \
                f"One of RACE_middle_val and/or RACE_high_val is not in the dataLoaders dictionary of dataloaders!"
            #key1 = "RACE_middle_val" if "RACE_middle_val" in self.dataLoaders.keys() else "RACE_middle_val_label"
            #key2 = "RACE_high_val" if "RACE_high_val" in self.dataLoaders.keys() else "RACE_high_val_label"

            # depending on the key names
            gen1 = None
            gen2 = None

            if self.type3 == "RACE_combined_val":
                gen1 = self.dataLoaders["RACE_middle_val"](mode="default_generate")
                gen2 = self.dataLoaders["RACE_high_val"](mode="default_generate")
            else: # RACE_combined_train_label
                gen1 = self.dataLoaders["RACE_middle_val"](mode="default_label")
                gen2 = self.dataLoaders["RACE_high_val"](mode="default_label")

            '''
            if key1 == "RACE_middle_val":
                gen1 = self.dataLoaders["RACE_middle_val"](mode="default_generate")
            else:
                gen1 = self.dataLoaders["RACE_middle_val_label"](mode="default_label")
            if key2 == "RACE_high_val":
                gen2 = self.dataLoaders["RACE_high_val"](mode="default_generate") #TODO: error here gen1 was here, which is obviously incorrect.
            else:
                gen2 = self.dataLoaders["RACE_high_val_label"](mode="default_label")
            '''

            stop_gen1 = False
            stop_gen2 = False
            while True:

                input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = None, None, None, None, None, None, None, None
                curr_gen = ''
                stop_all = True
                if random.random() > 0.5:
                    if not stop_gen1:  # do gen1 first.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen1)
                        curr_gen = "gen1"
                        stop_all = False
                    elif not stop_gen2:  # do gen2 if gen1 is finished.
                        curr_gen = "gen2"
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen2)
                        stop_all = False
                else:
                    if not stop_gen2:  # do gen2 first.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen2)
                        curr_gen = "gen2"
                        stop_all = False
                    elif not stop_gen1:  # do gen2 if gen1 is finished.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen1)
                        curr_gen = "gen1"
                        stop_all = False

                if stop_all:
                    yield input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2  # will be all None
                    break
                if input_string is None and curr_gen == "gen1":  # gen1 is finished, so don't output None and continue, also set stop_gen1 to True so we don't process it again.
                    stop_gen1 = True
                    continue
                if input_string is None and curr_gen == "gen2":
                    stop_gen2 = True
                    continue

                yield input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2
        elif self.type3 == "RACE_combined_test": raise Exception(f"Not implemented (RACE_combined_test)!")

    def get_race_dataloader(self):

        mini_generator = None
        mode_ = None
        if self.type in ["RACE_middle_train", "RACE_middle_val", "RACE_high_train", "RACE_high_val"]:
            mini_generator = self.dataLoaders[self.type](mode="default_generate")
            mode_ = "default_generate"
        elif self.type in ["RACE_middle_test", "RACE_high_test"]:
            mini_generator = self.dataLoaders[self.type](mode="test")
            mode_ = "test"
        elif self.type in ["RACE_middle_train_label", "RACE_middle_val_label", "RACE_high_train_label",
                           "RACE_high_val_label"]:
            mini_generator = self.dataLoaders[self.type](mode="default_label")
            mode_ = "default_label"
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
            elif self.type == "RACE_combined_test":
                raise Exception(f"Not implemented yet!")
        if mode_ == "default_generate": # also works for the label version here, hence, why it hasn't been implemented below.
            while True:
                break_ = False
                #try: # Safety so whole training isn't stopped for one error. No error should be reached.
                input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(mini_generator)  # label will be a list of one element, the correct answer...
                #except RuntimeError as e:  # stopIteration
                #    print(f"Runtime Error: {e} \n Continuing as per normal as generator has nothing left to generate!")
                #    break_ = True
                #except Exception as e:
                #    print(f"Unknown error with generator, continuing! {e}")
                #    break_ = True

                if break_: break
                if input_string is None or input_id is None: break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]
                lm_tok_id = self.tokenizer.encode_single(self.lm_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]
                    aux_label = aux_label + [pad_tok_id for _ in range(self.seq_len - len(aux_label))]
                    if not isinstance(label, int):  # integer only, not a list of integers. (not of this)
                        label = label + [pad_tok_id for _ in range(self.seq_len - len(label))]  #
                    else: raise Exception(f"label should not be of type int, but is instead a list of integers")

                if self.override_lm:
                    input_id = [self.cls_tok_id, aux_tok_1, lm_tok_id] + input_id
                else:
                    input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                label_id = tf.cast(tf.convert_to_tensor(label), dtype=tf.dtypes.int64)
                aux_label = tf.cast(tf.convert_to_tensor(aux_label), dtype=tf.dtypes.int64)
                aoint_indices = tf.cast(tf.convert_to_tensor(aoint_indices), dtype=tf.dtypes.int64)
                sample_weights = tf.cast(tf.convert_to_tensor(sample_weights), dtype=tf.dtypes.int64)

                yield input_string, input_id, label_id, aux_label, aoint_indices, sample_weights
        elif mode_ == "default_label": raise Exception("Not implemented yet (default_label)") #TODO
        elif mode_ == "test":
            while True:
                break_ = False
                #try: # Safety so whole training isn't stopped for one error. No error should be reached.
                input_string, input_id, all_labels, correct_ao, aoint_indices, aux_tok_1, aux_tok_2 = next(mini_generator)  # label will be a list of one element, the correct answer...
                #except RuntimeError as e:  # stopIteration
                ##    print(f"Runtime Error: {e} \n Continuing as per normal as generator has nothing left to generate!")
                 #   break_ = True
                #except Exception as e:
                #    print(f"Unknown error with generator, continuing! {e}")
                #    break_ = True

                if break_: break
                if input_string is None or input_id is None: break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]
                lm_tok_id = self.tokenizer.encode_single(self.lm_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]

                if self.override_lm:
                    input_id = [self.cls_tok_id, aux_tok_1, lm_tok_id] + input_id
                else:
                    input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                aoint_indices = tf.cast(tf.convert_to_tensor(aoint_indices), dtype=tf.dtypes.int64)
                all_labels = tf.cast(tf.convert_to_tensor(np.asarray(all_labels)), dtype=tf.dtypes.string)
                correct_ao = tf.cast(tf.convert_to_tensor(np.asarray(correct_ao)), dtype=tf.dtypes.string)

                yield input_string, input_id, all_labels, correct_ao, aoint_indices
        else: raise Exception(f"Invalid mode_!")

    def get_squad_test(self):

        mini_generator = None
        # train and val (technically test as well) are the same, so no need to distinghuish between them
        if self.type == "SQuAD_test_default":
            mini_generator = self.dataLoaders[self.type](mode="test")

        # Shuffling is irrelevant during testing.

        while True:
            #try: # Safety so whole training isn't stopped for one error. No error should be reached.
            input_string, input_id, answers, aux_tok_1, aux_tok_2 = next(mini_generator)

            if input_string is None or input_id is None: break

            pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]

            if self.pad_to_max_length:
                input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]

            # aux_tok_1 represents the mode, i.e. the decoder <dec>
            if self.override_lm:
                input_id = [self.cls_tok_id, aux_tok_1, self.dataLoaders[self.type].lm_tok_id] + input_id
            else:
                input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

            input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
            input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
            answers = tf.cast(tf.convert_to_tensor(answers), dtype=tf.dtypes.string)


            yield input_string, input_id, answers

    def get_squad_train(self):

        mini_generator = None
        # train and val (technically test as well) are the same, so no need to distinghuish between them
        if self.type == "SQuAD_train_default":
            mini_generator = self.dataLoaders[self.type](mode="training")
        elif self.type == "SQuAD_val_default":
            mini_generator = self.dataLoaders[self.type](mode="val")


        if self.shuffle: self.dataLoaders[self.type].shuffle_data()

        while True:
            #try: # Safety so whole training isn't stopped for one error. No error should be reached.
            input_string, input_id, label, aux_label, \
            sample_weights, aux_tok_1, aux_tok_2 = next(mini_generator)

            if input_string is None or input_id is None: break

            pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]

            if self.pad_to_max_length:
                input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]
                aux_label = aux_label + [pad_tok_id for _ in range(self.seq_len - len(aux_label))]
                if not isinstance(label, int):  # integer only, not a list of integers. (not of this)
                    label = label + [pad_tok_id for _ in range(self.seq_len - len(label))]  #
                else: raise Exception(f"label should not be of type int, but is instead a list of integers")

            # aux_tok_1 represents the mode, i.e. the decoder <dec>
            if self.override_lm:
                input_id = [self.cls_tok_id, aux_tok_1, self.dataLoaders[self.type].lm_tok_id] + input_id
            else:
                input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

            input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
            input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
            label_id = tf.cast(tf.convert_to_tensor(label), dtype=tf.dtypes.int64)
            aux_label = tf.cast(tf.convert_to_tensor(aux_label), dtype=tf.dtypes.int64)
            sample_weights = tf.cast(tf.convert_to_tensor(sample_weights), dtype=tf.dtypes.int64)

            yield input_string, input_id, label_id, aux_label, sample_weights

    def get_BoolQ(self):

        mini_generator = None
        # train and val (technically test as well) are the same, so no need to distinghuish between them
        if self.type == "BoolQ_train":
            mini_generator = self.dataLoaders[self.type](mode="train", shuffle=self.shuffle)
        elif self.type == "BoolQ_val":
            mini_generator = self.dataLoaders[self.type](mode="val", shuffle=self.shuffle)
        elif self.type == "BoolQ_test":
            mini_generator = self.dataLoaders[self.type](mode="test", shuffle=self.shuffle)
        elif self.type == "BoolQCS_test":
            mini_generator = self.dataLoaders[self.type](mode="test", shuffle=self.shuffle)
        else: raise Exception("invalid type!")


        if self.type == "BoolQ_train" or self.type == "BoolQ_val":

            while True:

                input_string, input_id, label, aux_label, \
                sample_weights, aux_tok_1, aux_tok_2 = next(mini_generator)

                if input_string is None or input_id is None: break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]
                    aux_label = aux_label + [pad_tok_id for _ in range(self.seq_len - len(aux_label))]
                    if not isinstance(label, int):  # integer only, not a list of integers. (not of this)
                        label = label + [pad_tok_id for _ in range(self.seq_len - len(label))]  #
                    else: raise Exception(f"label should not be of type int, but is instead a list of integers")

                # aux_tok_1 represents the mode, i.e. the decoder <dec>
                if self.override_lm:
                    input_id = [self.cls_tok_id, aux_tok_1, self.dataLoaders[self.type].lm_tok_id] + input_id
                else:
                    input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                label_id = tf.cast(tf.convert_to_tensor(label), dtype=tf.dtypes.int64)
                aux_label = tf.cast(tf.convert_to_tensor(aux_label), dtype=tf.dtypes.int64)
                sample_weights = tf.cast(tf.convert_to_tensor(sample_weights), dtype=tf.dtypes.int64)
                yield input_string, input_id, label_id, aux_label, sample_weights
        elif self.type == "BoolQ_test" or self.type == "BoolQCS_test":
            # with BoolQ_test_no_answer there is no difference between idx and answer, both are strings, the code below
            # handles it...
            while True:

                input_string, input_id, answer, aux_tok_1, aux_tok_2 = next(mini_generator)

                if input_string is None or input_id is None: break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]

                # aux_tok_1 represents the mode, i.e. the decoder <dec>
                if self.override_lm:
                    input_id = [self.cls_tok_id, aux_tok_1, self.dataLoaders[self.type].lm_tok_id] + input_id
                else:
                    input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                answer = tf.cast(tf.convert_to_tensor(answer), dtype=tf.dtypes.string)
                yield input_string, input_id, answer

    def get_NarrativeQA(self):

        mini_generator = None
        # train and val (technically test as well) are the same, so no need to distinghuish between them
        if self.type == "NarrativeQA_train":
            mini_generator = self.dataLoaders[self.type](mode="train", shuffle=self.shuffle)
        elif self.type == "NarrativeQA_val":
            mini_generator = self.dataLoaders[self.type](mode="val", shuffle=self.shuffle)
        elif self.type == "NarrativeQA_test":
            mini_generator = self.dataLoaders[self.type](mode="test", shuffle=self.shuffle)
        elif self.type == "NarrativeQA_val_test":
            mini_generator = self.dataLoaders[self.type](mode="val_test", shuffle=self.shuffle)
        else:
            raise Exception("invalid type!")

        if self.type == "NarrativeQA_train" or self.type == "NarrativeQA_val":

            while True:
                # try: # Safety so whole training isn't stopped for one error. No error should be reached.
                input_string, input_id, label, aux_label, \
                sample_weights, aux_tok_1, aux_tok_2 = next(mini_generator)

                if input_string is None or input_id is None: break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]
                    aux_label = aux_label + [pad_tok_id for _ in range(self.seq_len - len(aux_label))]
                    if not isinstance(label, int):  # integer only, not a list of integers. (not of this)
                        label = label + [pad_tok_id for _ in range(self.seq_len - len(label))]  #
                    else:
                        raise Exception(f"label should not be of type int, but is instead a list of integers")

                # aux_tok_1 represents the mode, i.e. the decoder <dec>
                if self.override_lm:
                    input_id = [self.cls_tok_id, aux_tok_1, self.dataLoaders[self.type].lm_tok_id] + input_id
                else:
                    input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                label_id = tf.cast(tf.convert_to_tensor(label), dtype=tf.dtypes.int64)
                aux_label = tf.cast(tf.convert_to_tensor(aux_label), dtype=tf.dtypes.int64)
                sample_weights = tf.cast(tf.convert_to_tensor(sample_weights), dtype=tf.dtypes.int64)
                yield input_string, input_id, label_id, aux_label, sample_weights

        elif self.type == "NarrativeQA_test" or self.type == "NarrativeQA_val_test":

            while True:
                # try: # Safety so whole training isn't stopped for one error. No error should be reached.
                input_string, input_id, answers, aux_tok_1, aux_tok_2 = next(mini_generator)
                if input_string is None or input_id is None: break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]

                # aux_tok_1 represents the mode, i.e. the decoder <dec>
                if self.override_lm:
                    input_id = [self.cls_tok_id, aux_tok_1, self.dataLoaders[self.type].lm_tok_id] + input_id
                else:
                    input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                answers = tf.cast(tf.convert_to_tensor(answers), dtype=tf.dtypes.string)

                yield input_string, input_id, answers

    def get_WG(self):

        mini_generator = None
        # train and val (technically test as well) are the same, so no need to distinghuish between them
        if self.type == "WG_train":
            mini_generator = self.dataLoaders[self.type](mode="train_generate", shuffle=self.shuffle)
        elif self.type == "WG_train_label":
            mini_generator = self.dataLoaders["WG_train"](mode="train_label", shuffle=self.shuffle)
        elif self.type == "WG_val":
            mini_generator = self.dataLoaders[self.type](mode="val_generate", shuffle=self.shuffle)
        elif self.type == "WG_val_label":
            mini_generator = self.dataLoaders["WG_val"](mode="val_label", shuffle=self.shuffle)
        elif self.type == "WG_test":
            mini_generator = self.dataLoaders[self.type](mode="test", shuffle=self.shuffle)
        else:
            raise Exception("invalid type!")

        if self.type == "WG_train_label" or self.type == "WG_val_label":

            while True:
                # try: # Safety so whole training isn't stopped for one error. No error should be reached.
                input_string, input_id, label, aux_label, aoint_indices, \
                sample_weights, aux_tok_1, aux_tok_2 = next(mini_generator)

                if input_string is None or input_id is None: break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]
                lm_tok_id = self.tokenizer.encode_single(self.lm_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]
                    aux_label = aux_label + [pad_tok_id for _ in range(self.seq_len - len(aux_label))]
                    if not isinstance(label, int):  # integer only, not a list of integers. (not of this)
                        label = label + [pad_tok_id for _ in range(self.seq_len - len(label))]  #
                    else:
                        raise Exception(f"label should not be of type int, but is instead a list of integers")

                # aux_tok_1 represents the mode, i.e. the decoder <dec>
                if self.override_lm:
                    input_id = [self.cls_tok_id, aux_tok_1, lm_tok_id] + input_id
                else:
                    input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                label_id = tf.cast(tf.convert_to_tensor(label), dtype=tf.dtypes.int64)
                aux_label = tf.cast(tf.convert_to_tensor(aux_label), dtype=tf.dtypes.int64)
                aoint_indices = tf.cast(tf.convert_to_tensor(aoint_indices), dtype=tf.dtypes.int64)
                sample_weights = tf.cast(tf.convert_to_tensor(sample_weights), dtype=tf.dtypes.int64)
                yield input_string, input_id, label_id, aux_label, aoint_indices, sample_weights
        elif self.type == "WG_test":
            while True:
                # try: # Safety so whole training isn't stopped for one error. No error should be reached.
                input_string, input_id, all_labels, correct_ao, aoint_indices, aux_tok_1, aux_tok_2 = next(mini_generator)

                if input_string is None or input_id is None: break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]
                lm_tok_id = self.tokenizer.encode_single(self.lm_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]

                # aux_tok_1 represents the mode, i.e. the decoder <dec>
                if self.override_lm:
                    input_id = [self.cls_tok_id, aux_tok_1, lm_tok_id] + input_id
                else:
                    input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                all_labels = tf.cast(tf.convert_to_tensor(all_labels), dtype=tf.dtypes.string)
                correct_ao = tf.cast(tf.convert_to_tensor(correct_ao), dtype=tf.dtypes.string)
                aoint_indices = tf.cast(tf.convert_to_tensor(aoint_indices), dtype=tf.dtypes.int64)
                yield input_string, input_id, all_labels, correct_ao, aoint_indices

    def get_SIQA(self):

        mini_generator = None
        # train and val (technically test as well) are the same, so no need to distinghuish between them
        if self.type == "SIQA_train":
            mini_generator = self.dataLoaders[self.type](mode="train_generate", shuffle=self.shuffle)
        elif self.type == "SIQA_train_label":
            mini_generator = self.dataLoaders["SIQA_train"](mode="train_label", shuffle=self.shuffle)
        elif self.type == "SIQA_val":
            mini_generator = self.dataLoaders[self.type](mode="val_generate", shuffle=self.shuffle)
        elif self.type == "SIQA_val_label":
            mini_generator = self.dataLoaders["SIQA_val"](mode="val_label", shuffle=self.shuffle)
        elif self.type == "SIQA_test":
            mini_generator = self.dataLoaders[self.type](mode="test", shuffle=self.shuffle)
        else:
            raise Exception("invalid type!")

        if self.type == "SIQA_train_label" or self.type == "SIQA_val_label":

            while True:
                # try: # Safety so whole training isn't stopped for one error. No error should be reached.
                input_string, input_id, label, aux_label, aoint_indices, \
                sample_weights, aux_tok_1, aux_tok_2 = next(mini_generator)

                if input_string is None or input_id is None: break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]
                lm_tok_id = self.tokenizer.encode_single(self.lm_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]
                    aux_label = aux_label + [pad_tok_id for _ in range(self.seq_len - len(aux_label))]
                    if not isinstance(label, int):  # integer only, not a list of integers. (not of this)
                        label = label + [pad_tok_id for _ in range(self.seq_len - len(label))]  #
                    else:
                        raise Exception(f"label should not be of type int, but is instead a list of integers")

                # aux_tok_1 represents the mode, i.e. the decoder <dec>
                if self.override_lm:
                    input_id = [self.cls_tok_id, aux_tok_1, lm_tok_id] + input_id
                else:
                    input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                label_id = tf.cast(tf.convert_to_tensor(label), dtype=tf.dtypes.int64)
                aux_label = tf.cast(tf.convert_to_tensor(aux_label), dtype=tf.dtypes.int64)
                aoint_indices = tf.cast(tf.convert_to_tensor(aoint_indices), dtype=tf.dtypes.int64)
                sample_weights = tf.cast(tf.convert_to_tensor(sample_weights), dtype=tf.dtypes.int64)
                yield input_string, input_id, label_id, aux_label, aoint_indices, sample_weights
        elif self.type == "SIQA_test":
            while True:
                # try: # Safety so whole training isn't stopped for one error. No error should be reached.
                input_string, input_id, all_labels, correct_ao, aoint_indices, aux_tok_1, aux_tok_2 = next(mini_generator)

                if input_string is None or input_id is None: break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]
                lm_tok_id = self.tokenizer.encode_single(self.lm_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]

                # aux_tok_1 represents the mode, i.e. the decoder <dec>
                if self.override_lm:
                    input_id = [self.cls_tok_id, aux_tok_1, lm_tok_id] + input_id
                else:
                    input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                all_labels = tf.cast(tf.convert_to_tensor(all_labels), dtype=tf.dtypes.string)
                correct_ao = tf.cast(tf.convert_to_tensor(correct_ao), dtype=tf.dtypes.string)
                aoint_indices = tf.cast(tf.convert_to_tensor(aoint_indices), dtype=tf.dtypes.int64)
                yield input_string, input_id, all_labels, correct_ao, aoint_indices

    def get_PIQA(self):

        mini_generator = None
        # train and val (technically test as well) are the same, so no need to distinghuish between them
        if self.type == "PIQA_train":
            mini_generator = self.dataLoaders[self.type](mode="train_generate", shuffle=self.shuffle)
        elif self.type == "PIQA_train_label":
            mini_generator = self.dataLoaders["PIQA_train"](mode="train_label", shuffle=self.shuffle)
        elif self.type == "PIQA_val":
            mini_generator = self.dataLoaders[self.type](mode="val_generate", shuffle=self.shuffle)
        elif self.type == "PIQA_val_label":
            mini_generator = self.dataLoaders["PIQA_val"](mode="val_label", shuffle=self.shuffle)
        elif self.type == "PIQA_test":
            mini_generator = self.dataLoaders[self.type](mode="test", shuffle=self.shuffle)
        else:
            raise Exception("invalid type!")

        if self.type == "PIQA_train_label" or self.type == "PIQA_val_label":

            while True:
                # try: # Safety so whole training isn't stopped for one error. No error should be reached.
                input_string, input_id, label, aux_label, aoint_indices, \
                sample_weights, aux_tok_1, aux_tok_2 = next(mini_generator)

                if input_string is None or input_id is None: break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]
                lm_tok_id = self.tokenizer.encode_single(self.lm_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]
                    aux_label = aux_label + [pad_tok_id for _ in range(self.seq_len - len(aux_label))]
                    if not isinstance(label, int):  # integer only, not a list of integers. (not of this)
                        label = label + [pad_tok_id for _ in range(self.seq_len - len(label))]  #
                    else:
                        raise Exception(f"label should not be of type int, but is instead a list of integers")

                # aux_tok_1 represents the mode, i.e. the decoder <dec>
                if self.override_lm:
                    input_id = [self.cls_tok_id, aux_tok_1, lm_tok_id] + input_id
                else:
                    input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                label_id = tf.cast(tf.convert_to_tensor(label), dtype=tf.dtypes.int64)
                aux_label = tf.cast(tf.convert_to_tensor(aux_label), dtype=tf.dtypes.int64)
                aoint_indices = tf.cast(tf.convert_to_tensor(aoint_indices), dtype=tf.dtypes.int64)
                sample_weights = tf.cast(tf.convert_to_tensor(sample_weights), dtype=tf.dtypes.int64)
                yield input_string, input_id, label_id, aux_label, aoint_indices, sample_weights
        elif self.type == "PIQA_test":
            while True:
                # try: # Safety so whole training isn't stopped for one error. No error should be reached.
                input_string, input_id, all_labels, correct_ao, aoint_indices, aux_tok_1, aux_tok_2 = next(mini_generator)

                if input_string is None or input_id is None: break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]
                lm_tok_id = self.tokenizer.encode_single(self.lm_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]

                # aux_tok_1 represents the mode, i.e. the decoder <dec>
                if self.override_lm:
                    input_id = [self.cls_tok_id, aux_tok_1, lm_tok_id] + input_id
                else:
                    input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                all_labels = tf.cast(tf.convert_to_tensor(all_labels), dtype=tf.dtypes.string)
                correct_ao = tf.cast(tf.convert_to_tensor(correct_ao), dtype=tf.dtypes.string)
                aoint_indices = tf.cast(tf.convert_to_tensor(aoint_indices), dtype=tf.dtypes.int64)
                yield input_string, input_id, all_labels, correct_ao, aoint_indices

    def get_CQA(self):

        mini_generator = None
        # train and val (technically test as well) are the same, so no need to distinghuish between them
        if self.type == "CQA_train":
            mini_generator = self.dataLoaders[self.type](mode="train_generate", shuffle=self.shuffle)
        elif self.type == "CQA_train_label":
            mini_generator = self.dataLoaders["CQA_train"](mode="train_label", shuffle=self.shuffle)
        elif self.type == "CQA_val":
            mini_generator = self.dataLoaders[self.type](mode="val_generate", shuffle=self.shuffle)
        elif self.type == "CQA_val_label":
            mini_generator = self.dataLoaders["CQA_val"](mode="val_label", shuffle=self.shuffle)
        elif self.type == "CQA_test":
            mini_generator = self.dataLoaders[self.type](mode="test", shuffle=self.shuffle)
        else:
            raise Exception("invalid type!")

        if self.type == "CQA_train_label" or self.type == "CQA_val_label":

            while True:
                # try: # Safety so whole training isn't stopped for one error. No error should be reached.
                input_string, input_id, label, aux_label, aoint_indices, \
                sample_weights, aux_tok_1, aux_tok_2 = next(mini_generator)

                if input_string is None or input_id is None: break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]
                lm_tok_id = self.tokenizer.encode_single(self.lm_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]
                    aux_label = aux_label + [pad_tok_id for _ in range(self.seq_len - len(aux_label))]
                    if not isinstance(label, int):  # integer only, not a list of integers. (not of this)
                        label = label + [pad_tok_id for _ in range(self.seq_len - len(label))]  #
                    else:
                        raise Exception(f"label should not be of type int, but is instead a list of integers")

                # aux_tok_1 represents the mode, i.e. the decoder <dec>
                if self.override_lm:
                    input_id = [self.cls_tok_id, aux_tok_1, lm_tok_id] + input_id
                else:
                    input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                label_id = tf.cast(tf.convert_to_tensor(label), dtype=tf.dtypes.int64)
                aux_label = tf.cast(tf.convert_to_tensor(aux_label), dtype=tf.dtypes.int64)
                aoint_indices = tf.cast(tf.convert_to_tensor(aoint_indices), dtype=tf.dtypes.int64)
                sample_weights = tf.cast(tf.convert_to_tensor(sample_weights), dtype=tf.dtypes.int64)
                yield input_string, input_id, label_id, aux_label, aoint_indices, sample_weights
        elif self.type == "CQA_test":

            while True:
                # try: # Safety so whole training isn't stopped for one error. No error should be reached.
                input_string, input_id, all_labels, correct_ao, aoint_indices, aux_tok_1, aux_tok_2 = next(mini_generator)

                if input_string is None or input_id is None: break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]
                lm_tok_id = self.tokenizer.encode_single(self.lm_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]

                # aux_tok_1 represents the mode, i.e. the decoder <dec>
                if self.override_lm:
                    input_id = [self.cls_tok_id, aux_tok_1, lm_tok_id] + input_id
                else:
                    input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                all_labels = tf.cast(tf.convert_to_tensor(all_labels), dtype=tf.dtypes.string)
                correct_ao = tf.cast(tf.convert_to_tensor(correct_ao), dtype=tf.dtypes.string)
                aoint_indices = tf.cast(tf.convert_to_tensor(aoint_indices), dtype=tf.dtypes.int64)
                yield input_string, input_id, all_labels, correct_ao, aoint_indices

    def get_OBQA(self):

        mini_generator = None
        # train and val (technically test as well) are the same, so no need to distinghuish between them
        if self.type == "OBQA_train":
            mini_generator = self.dataLoaders[self.type](mode="train_generate", shuffle=self.shuffle)
        elif self.type == "OBQA_train_label":
            mini_generator = self.dataLoaders["OBQA_train"](mode="train_label", shuffle=self.shuffle)
        elif self.type == "OBQA_val":
            mini_generator = self.dataLoaders[self.type](mode="val_generate", shuffle=self.shuffle)
        elif self.type == "OBQA_val_label":
            mini_generator = self.dataLoaders["OBQA_val"](mode="val_label", shuffle=self.shuffle)
        elif self.type == "OBQA_test":
            mini_generator = self.dataLoaders[self.type](mode="test", shuffle=self.shuffle)
        else:
            raise Exception("invalid type!")

        if self.type == "OBQA_train_label" or self.type == "OBQA_val_label":

            while True:
                # try: # Safety so whole training isn't stopped for one error. No error should be reached.
                input_string, input_id, label, aux_label, aoint_indices, \
                sample_weights, aux_tok_1, aux_tok_2 = next(mini_generator)

                if input_string is None or input_id is None: break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]
                lm_tok_id = self.tokenizer.encode_single(self.lm_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]
                    aux_label = aux_label + [pad_tok_id for _ in range(self.seq_len - len(aux_label))]
                    if not isinstance(label, int):  # integer only, not a list of integers. (not of this)
                        label = label + [pad_tok_id for _ in range(self.seq_len - len(label))]  #
                    else:
                        raise Exception(f"label should not be of type int, but is instead a list of integers")

                # aux_tok_1 represents the mode, i.e. the decoder <dec>
                if self.override_lm:
                    input_id = [self.cls_tok_id, aux_tok_1, lm_tok_id] + input_id
                else:
                    input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                label_id = tf.cast(tf.convert_to_tensor(label), dtype=tf.dtypes.int64)
                aux_label = tf.cast(tf.convert_to_tensor(aux_label), dtype=tf.dtypes.int64)
                aoint_indices = tf.cast(tf.convert_to_tensor(aoint_indices), dtype=tf.dtypes.int64)
                sample_weights = tf.cast(tf.convert_to_tensor(sample_weights), dtype=tf.dtypes.int64)
                yield input_string, input_id, label_id, aux_label, aoint_indices, sample_weights
        elif self.type == "OBQA_test":

            while True:
                # try: # Safety so whole training isn't stopped for one error. No error should be reached.
                input_string, input_id, all_labels, correct_ao, aoint_indices, aux_tok_1, aux_tok_2 = next(mini_generator)

                if input_string is None or input_id is None: break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]
                lm_tok_id = self.tokenizer.encode_single(self.lm_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]

                # aux_tok_1 represents the mode, i.e. the decoder <dec>
                if self.override_lm:
                    input_id = [self.cls_tok_id, aux_tok_1, lm_tok_id] + input_id
                else:
                    input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                all_labels = tf.cast(tf.convert_to_tensor(all_labels), dtype=tf.dtypes.string)
                correct_ao = tf.cast(tf.convert_to_tensor(correct_ao), dtype=tf.dtypes.string)
                aoint_indices = tf.cast(tf.convert_to_tensor(aoint_indices), dtype=tf.dtypes.int64)
                yield input_string, input_id, all_labels, correct_ao, aoint_indices

    def _combined_ARC_helper(self):

        if self.type == "ARC_train_label_combined" or self.type == "ARC_train_combined":
            assert ("ARC_train_easy" in self.dataLoaders.keys() and "ARC_train_challenge" in self.dataLoaders.keys()), \
                    f"One of ARC_train_easy and/or ARC_train_challenge is not in the dataLoaders dictionary of dataloaders!"

            # depending on the key names
            gen1 = None
            gen2 = None

            if self.type == "ARC_train_combined":
                raise Exception(f"Currently not supported: ARC_train_combined")
                gen1 = self.dataLoaders["ARC_train_easy"](mode="train_generate", shuffle=self.shuffle)
                gen2 = self.dataLoaders["ARC_train_challenge"](mode="train_generate", shuffle=self.shuffle)
            elif self.type == "ARC_train_label_combined":
                gen1 = self.dataLoaders["ARC_train_easy"](mode="train_label", shuffle=self.shuffle)
                gen2 = self.dataLoaders["ARC_train_challenge"](mode="train_label", shuffle=self.shuffle)

            stop_gen1 = False
            stop_gen2 = False
            while True:

                input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = None, None, None, None, None, None, None, None
                curr_gen = ''
                stop_all = True
                if random.random() > 0.5:
                    if not stop_gen1:  # do gen1 first.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen1)
                        curr_gen = "gen1"
                        stop_all = False
                    elif not stop_gen2:  # do gen2 if gen1 is finished.
                        curr_gen = "gen2"
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen2)
                        stop_all = False
                else:
                    if not stop_gen2:  # do gen2 first.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen2)
                        curr_gen = "gen2"
                        stop_all = False
                    elif not stop_gen1:  # do gen1 if gen2 is finished.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen1)
                        curr_gen = "gen1"
                        stop_all = False

                if stop_all:
                    yield input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2  # will be all None
                    break
                if input_string is None and curr_gen == "gen1":  # gen1 is finished, so don't output None and continue, also set stop_gen1 to True so we don't process it again.
                    stop_gen1 = True
                    continue
                if input_string is None and curr_gen == "gen2":
                    stop_gen2 = True
                    continue
                # print(input_string, input_id, label, aux_tok_1, aux_tok_2)
                yield input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2

        elif self.type == "ARC_val_label_combined" or self.type == "ARC_val_combined":
            assert ("ARC_val_easy" in self.dataLoaders.keys() and "ARC_val_challenge" in self.dataLoaders.keys()), \
                f"One of ARC_val_easy and/or ARC_val_challenge is not in the dataLoaders dictionary of dataloaders!"
            # depending on the key names

            gen1 = None
            gen2 = None

            if self.type == "ARC_val_combined":
                raise Exception(f"Currently not supported: ARC_val_combined")
                gen1 = self.dataLoaders["ARC_val_easy"](mode="val_generate", shuffle=self.shuffle)
                gen2 = self.dataLoaders["ARC_val_challenge"](mode="val_generate", shuffle=self.shuffle)
            elif self.type == "ARC_val_label_combined":
                gen1 = self.dataLoaders["ARC_val_easy"](mode="val_label", shuffle=self.shuffle)
                gen2 = self.dataLoaders["ARC_val_challenge"](mode="val_label", shuffle=self.shuffle)

            stop_gen1 = False
            stop_gen2 = False

            while True:

                input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = None, None, None, None, None, None, None, None
                curr_gen = ''
                stop_all = True
                if random.random() > 0.5:
                    if not stop_gen1:  # do gen1 first.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen1)
                        curr_gen = "gen1"
                        stop_all = False
                    elif not stop_gen2:  # do gen2 if gen1 is finished.
                        curr_gen = "gen2"
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen2)
                        stop_all = False
                else:
                    if not stop_gen2:  # do gen2 first.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen2)
                        curr_gen = "gen2"
                        stop_all = False

                    elif not stop_gen1:  # do gen1 if gen2 is finished.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen1)
                        curr_gen = "gen1"
                        stop_all = False

                if stop_all:
                    yield input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2  # will be all None
                    break

                if input_string is None and curr_gen == "gen1":  # gen1 is finished, so don't output None and continue, also set stop_gen1 to True so we don't process it again.
                    stop_gen1 = True
                    continue

                if input_string is None and curr_gen == "gen2":
                    stop_gen2 = True
                    continue

                # print(input_string, input_id, label, aux_tok_1, aux_tok_2)

                yield input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2
        else: raise Exception(f"Invalid type: {self.type}")
        #elif self.type == "ARC_combined_test": raise Exception(f"Not implemented (RACE_combined_test)!")

    def _combined_ARC_helper2(self):

        if self.type2 == "ARC_train_label_combined" or self.type2 == "ARC_train_combined":
            assert ("ARC_train_easy" in self.dataLoaders.keys() and "ARC_train_challenge" in self.dataLoaders.keys()), \
                    f"One of ARC_train_easy and/or ARC_train_challenge is not in the dataLoaders dictionary of dataloaders!"

            # depending on the key names
            gen1 = None
            gen2 = None

            if self.type2 == "ARC_train_combined":
                raise Exception(f"Currently not supported: ARC_train_combined")
                gen1 = self.dataLoaders["ARC_train_easy"](mode="train_generate", shuffle=self.shuffle)
                gen2 = self.dataLoaders["ARC_train_challenge"](mode="train_generate", shuffle=self.shuffle)
            elif self.type2 == "ARC_train_label_combined":
                gen1 = self.dataLoaders["ARC_train_easy"](mode="train_label", shuffle=self.shuffle)
                gen2 = self.dataLoaders["ARC_train_challenge"](mode="train_label", shuffle=self.shuffle)

            stop_gen1 = False
            stop_gen2 = False
            while True:

                input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = None, None, None, None, None, None, None, None
                curr_gen = ''
                stop_all = True
                if random.random() > 0.5:
                    if not stop_gen1:  # do gen1 first.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen1)
                        curr_gen = "gen1"
                        stop_all = False
                    elif not stop_gen2:  # do gen2 if gen1 is finished.
                        curr_gen = "gen2"
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen2)
                        stop_all = False
                else:
                    if not stop_gen2:  # do gen2 first.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen2)
                        curr_gen = "gen2"
                        stop_all = False
                    elif not stop_gen1:  # do gen1 if gen2 is finished.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen1)
                        curr_gen = "gen1"
                        stop_all = False

                if stop_all:
                    yield input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2  # will be all None
                    break
                if input_string is None and curr_gen == "gen1":  # gen1 is finished, so don't output None and continue, also set stop_gen1 to True so we don't process it again.
                    stop_gen1 = True
                    continue
                if input_string is None and curr_gen == "gen2":
                    stop_gen2 = True
                    continue
                # print(input_string, input_id, label, aux_tok_1, aux_tok_2)
                yield input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2

        elif self.type2 == "ARC_val_label_combined" or self.type2 == "ARC_val_combined":
            assert ("ARC_val_easy" in self.dataLoaders.keys() and "ARC_val_challenge" in self.dataLoaders.keys()), \
                f"One of ARC_val_easy and/or ARC_val_challenge is not in the dataLoaders dictionary of dataloaders!"
            # depending on the key names

            gen1 = None
            gen2 = None

            if self.type2 == "ARC_val_combined":
                raise Exception(f"Currently not supported: ARC_val_combined")
                gen1 = self.dataLoaders["ARC_val_easy"](mode="val_generate", shuffle=self.shuffle)
                gen2 = self.dataLoaders["ARC_val_challenge"](mode="val_generate", shuffle=self.shuffle)
            elif self.type2 == "ARC_val_label_combined":
                gen1 = self.dataLoaders["ARC_val_easy"](mode="val_label", shuffle=self.shuffle)
                gen2 = self.dataLoaders["ARC_val_challenge"](mode="val_label", shuffle=self.shuffle)

            stop_gen1 = False
            stop_gen2 = False

            while True:

                input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = None, None, None, None, None, None, None, None
                curr_gen = ''
                stop_all = True
                if random.random() > 0.5:
                    if not stop_gen1:  # do gen1 first.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen1)
                        curr_gen = "gen1"
                        stop_all = False
                    elif not stop_gen2:  # do gen2 if gen1 is finished.
                        curr_gen = "gen2"
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen2)
                        stop_all = False
                else:
                    if not stop_gen2:  # do gen2 first.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen2)
                        curr_gen = "gen2"
                        stop_all = False

                    elif not stop_gen1:  # do gen1 if gen2 is finished.
                        input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = next(gen1)
                        curr_gen = "gen1"
                        stop_all = False

                if stop_all:
                    yield input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2  # will be all None
                    break

                if input_string is None and curr_gen == "gen1":  # gen1 is finished, so don't output None and continue, also set stop_gen1 to True so we don't process it again.
                    stop_gen1 = True
                    continue

                if input_string is None and curr_gen == "gen2":
                    stop_gen2 = True
                    continue

                # print(input_string, input_id, label, aux_tok_1, aux_tok_2)

                yield input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2
        else: raise Exception(f"Invalid type: {self.type2}")
        #elif self.type == "ARC_combined_test": raise Exception(f"Not implemented (RACE_combined_test)!")

    def get_ARC(self):
        '''
        elif key == "ARC_test_easy" or key == "ARC_test_challenge" \
                    or key == "ARC_train_easy" or key == "ARC_train_challenge" \
                    or key == "ARC_val_easy" or key == "ARC_val_challenge":
        '''
        mini_generator = None
        # train and val (technically test as well) are the same, so no need to distinghuish between them
        if self.type == "ARC_train_label_combined" or self.type == "ARC_val_label_combined":
            mini_generator = self._combined_ARC_helper()
        elif self.type == "ARC_test_easy":
            mini_generator = self.dataLoaders[self.type](mode="test", shuffle=self.shuffle)
        elif self.type == "ARC_test_challenge":
            mini_generator = self.dataLoaders[self.type](mode="test", shuffle=self.shuffle)
        else:
            raise Exception("invalid type!")

        if self.type == "ARC_train_label_combined" or self.type == "ARC_val_label_combined":

            while True:
                # try: # Safety so whole training isn't stopped for one error. No error should be reached.
                input_string, input_id, label, aux_label, aoint_indices, \
                sample_weights, aux_tok_1, aux_tok_2 = next(mini_generator)

                if input_string is None or input_id is None: break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]
                lm_tok_id = self.tokenizer.encode_single(self.lm_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]
                    aux_label = aux_label + [pad_tok_id for _ in range(self.seq_len - len(aux_label))]
                    if not isinstance(label, int):  # integer only, not a list of integers. (not of this)
                        label = label + [pad_tok_id for _ in range(self.seq_len - len(label))]  #
                    else:
                        raise Exception(f"label should not be of type int, but is instead a list of integers")

                # aux_tok_1 represents the mode, i.e. the decoder <dec>
                if self.override_lm:
                    input_id = [self.cls_tok_id, aux_tok_1, lm_tok_id] + input_id
                else:
                    input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                label_id = tf.cast(tf.convert_to_tensor(label), dtype=tf.dtypes.int64)
                aux_label = tf.cast(tf.convert_to_tensor(aux_label), dtype=tf.dtypes.int64)
                aoint_indices = tf.cast(tf.convert_to_tensor(aoint_indices), dtype=tf.dtypes.int64)
                sample_weights = tf.cast(tf.convert_to_tensor(sample_weights), dtype=tf.dtypes.int64)
                yield input_string, input_id, label_id, aux_label, aoint_indices, sample_weights
        elif self.type == "ARC_test_easy" or self.type == "ARC_test_challenge":

            while True:
                # try: # Safety so whole training isn't stopped for one error. No error should be reached.
                input_string, input_id, all_labels, correct_ao, aoint_indices, aux_tok_1, aux_tok_2 = next(
                    mini_generator)

                if input_string is None or input_id is None: break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]
                lm_tok_id = self.tokenizer.encode_single(self.lm_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]

                # aux_tok_1 represents the mode, i.e. the decoder <dec>
                if self.override_lm:
                    input_id = [self.cls_tok_id, aux_tok_1, lm_tok_id] + input_id
                else:
                    input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                all_labels = tf.cast(tf.convert_to_tensor(all_labels), dtype=tf.dtypes.string)
                correct_ao = tf.cast(tf.convert_to_tensor(correct_ao), dtype=tf.dtypes.string)
                aoint_indices = tf.cast(tf.convert_to_tensor(aoint_indices), dtype=tf.dtypes.int64)
                yield input_string, input_id, all_labels, correct_ao, aoint_indices

    def get_MCTest(self):

        mini_generator = None
        if self.type == "MCTest_train":
            mini_generator = self.dataLoaders[self.type](mode="train_generate", shuffle=self.shuffle)
        elif self.type == "MCTest_train_label":
            mini_generator = self.dataLoaders["MCTest_train"](mode="train_label", shuffle=self.shuffle)
        elif self.type == "MCTest_val":
            mini_generator = self.dataLoaders[self.type](mode="val_generate", shuffle=self.shuffle)
        elif self.type == "MCTest_val_label":
            mini_generator = self.dataLoaders["MCTest_val"](mode="val_label", shuffle=self.shuffle)
        elif self.type == "MCTest_test":
            mini_generator = self.dataLoaders[self.type](mode="test", shuffle=self.shuffle)
        else:
            raise Exception("invalid type!")

        if self.type == "MCTest_train_label" or self.type == "MCTest_val_label":

            while True:
                # try: # Safety so whole training isn't stopped for one error. No error should be reached.
                input_string, input_id, label, aux_label, aoint_indices, \
                sample_weights, aux_tok_1, aux_tok_2 = next(mini_generator)

                if input_string is None or input_id is None: break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]
                lm_tok_id = self.tokenizer.encode_single(self.lm_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]
                    aux_label = aux_label + [pad_tok_id for _ in range(self.seq_len - len(aux_label))]
                    if not isinstance(label, int):  # integer only, not a list of integers. (not of this)
                        label = label + [pad_tok_id for _ in range(self.seq_len - len(label))]  #
                    else:
                        raise Exception(f"label should not be of type int, but is instead a list of integers")

                # aux_tok_1 represents the mode, i.e. the decoder <dec>
                if self.override_lm:
                    input_id = [self.cls_tok_id, aux_tok_1, lm_tok_id] + input_id
                else:
                    input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                label_id = tf.cast(tf.convert_to_tensor(label), dtype=tf.dtypes.int64)
                aux_label = tf.cast(tf.convert_to_tensor(aux_label), dtype=tf.dtypes.int64)
                aoint_indices = tf.cast(tf.convert_to_tensor(aoint_indices), dtype=tf.dtypes.int64)
                sample_weights = tf.cast(tf.convert_to_tensor(sample_weights), dtype=tf.dtypes.int64)
                yield input_string, input_id, label_id, aux_label, aoint_indices, sample_weights

        elif self.type == "MCTest_test":

            while True:
                # try: # Safety so whole training isn't stopped for one error. No error should be reached.
                input_string, input_id, all_labels, correct_ao, aoint_indices, aux_tok_1, aux_tok_2 = next(mini_generator)

                if input_string is None or input_id is None: break

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]
                lm_tok_id = self.tokenizer.encode_single(self.lm_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]

                # aux_tok_1 represents the mode, i.e. the decoder <dec>
                if self.override_lm:
                    input_id = [self.cls_tok_id, aux_tok_1, lm_tok_id] + input_id
                else:
                    input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                all_labels = tf.cast(tf.convert_to_tensor(all_labels), dtype=tf.dtypes.string)
                correct_ao = tf.cast(tf.convert_to_tensor(correct_ao), dtype=tf.dtypes.string)
                aoint_indices = tf.cast(tf.convert_to_tensor(aoint_indices), dtype=tf.dtypes.int64)
                yield input_string, input_id, all_labels, correct_ao, aoint_indices

    def get_generator(self, type: str, shuffle: bool, override_lm: bool):

        self.shuffle = shuffle
        self.type = type
        self.override_lm = override_lm

        # race specific variables
        #self.race_label_bool = race_label_bool # True is label only is generated as an answer. False if the whole answer is to be generated.
        #input_string, input_id, label_id, aux_label, sample_weights
        generator = None
        if type == "C4_pretrain_enc_dec":
            generator = tf.data.Dataset.from_generator(self.pre_train_c4,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64))
        elif type == "NarrativeQA_train" or type == "NarrativeQA_val":
            generator = tf.data.Dataset.from_generator(self.get_NarrativeQA,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64))
        elif type == "NarrativeQA_test" or type == "NarrativeQA_val_test":
            generator = tf.data.Dataset.from_generator(self.get_NarrativeQA,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.string))
        elif type == "BoolQ_train" or type == "BoolQ_val":
            generator = tf.data.Dataset.from_generator(self.get_BoolQ,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64))
        elif type == "BoolQ_test":
            generator = tf.data.Dataset.from_generator(self.get_BoolQ,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.string))
        elif type == "BoolQCS_test":
            generator = tf.data.Dataset.from_generator(self.get_BoolQ,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.string))
        elif type == "BoolQ_test_no_answer":
            generator = tf.data.Dataset.from_generator(self.get_BoolQ,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.string))
        elif type == "SQuAD_train_default":
            generator = tf.data.Dataset.from_generator(self.get_squad_train,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64))
        elif type == "SQuAD_val_default":
            generator = tf.data.Dataset.from_generator(self.get_squad_train,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64))
        elif type == "SQuAD_test_default":
            generator = tf.data.Dataset.from_generator(self.get_squad_test,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.string))
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
                type == "RACE_high_train" or type == "RACE_high_val" or \
                type == "RACE_middle_train_label" or type == "RACE_middle_val_label" or \
                type == "RACE_high_train_label" or type == "RACE_high_val_label":
            generator = tf.data.Dataset.from_generator(self.get_race_dataloader,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
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
        elif type == "RACE_combined_train" or type == "RACE_combined_val" or \
                type == "RACE_combined_train_label" or type == "RACE_combined_val_label":
            generator = tf.data.Dataset.from_generator(self.get_race_dataloader,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
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
        elif type == "OBQA_train_label" or type == "OBQA_val_label":
            generator = tf.data.Dataset.from_generator(self.get_OBQA,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64))
        elif type == "OBQA_test":
            generator = tf.data.Dataset.from_generator(self.get_OBQA,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.string,
                                                                     tf.dtypes.string,
                                                                     tf.dtypes.int64))
        elif type == "CQA_train_label" or type == "CQA_val_label":
            generator = tf.data.Dataset.from_generator(self.get_CQA,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64))
        elif type == "CQA_test":
            generator = tf.data.Dataset.from_generator(self.get_CQA,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.string,
                                                                     tf.dtypes.string,
                                                                     tf.dtypes.int64))

        elif type == "WG_train_label" or type == "WG_val_label":
            generator = tf.data.Dataset.from_generator(self.get_WG,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64))
        elif type == "WG_test":
            generator = tf.data.Dataset.from_generator(self.get_WG,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.string,
                                                                     tf.dtypes.string,
                                                                     tf.dtypes.int64))

        elif type == "SIQA_train_label" or type == "SIQA_val_label":
            generator = tf.data.Dataset.from_generator(self.get_SIQA,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64))
        elif type == "SIQA_test":
            generator = tf.data.Dataset.from_generator(self.get_SIQA,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.string,
                                                                     tf.dtypes.string,
                                                                     tf.dtypes.int64))

        elif type == "PIQA_train_label" or type == "PIQA_val_label":
            generator = tf.data.Dataset.from_generator(self.get_PIQA,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64))
        elif type == "PIQA_test":
            generator = tf.data.Dataset.from_generator(self.get_PIQA,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.string,
                                                                     tf.dtypes.string,
                                                                     tf.dtypes.int64))
        #MCTest_train
        elif type == "MCTest_train_label" or type == "MCTest_val_label":
            generator = tf.data.Dataset.from_generator(self.get_MCTest,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64))
        elif type == "MCTest_test":
            generator = tf.data.Dataset.from_generator(self.get_MCTest,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.string,
                                                                     tf.dtypes.string,
                                                                     tf.dtypes.int64))

        elif type == "ARC_train_label_combined" or type == "ARC_val_label_combined":
            generator = tf.data.Dataset.from_generator(self.get_ARC,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64))

        elif type == "ARC_test_easy" or type == "ARC_test_challenge": # do challenge and easy separately.
            generator = tf.data.Dataset.from_generator(self.get_ARC,
                                                       output_types=(tf.dtypes.string,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.string,
                                                                     tf.dtypes.string,
                                                                     tf.dtypes.int64))

        return generator

    def _get_dataloader_RACE(self):
        return self._combined_race_helper2()
    def _get_dataloader_OBQA(self):
        return self.dataLoaders["OBQA_train"](mode="train_label", shuffle=self.shuffle)
    def _get_dataloader_ARC(self):
        return self._combined_ARC_helper2()
    def _get_dataloader_MCTest(self):
        return self.dataLoaders["MCTest_train"](mode="train_label", shuffle=self.shuffle)
    def _get_dataloader_SQuADv2(self):
        return self.dataLoaders["SQuAD_train_default"](mode="training")
    def _get_dataloader_NarrativeQA(self):
        return self.dataLoaders["NarrativeQA_train"](mode="train", shuffle=self.shuffle)
    def _get_dataloader_BoolQ(self):
        return self.dataLoaders["BoolQ_train"](mode="train", shuffle=self.shuffle)

    def getGeneralTrainer(self):
        ## get all data loaders...
        # get a dict of all data loaders here...
        # partly hardcoded.

        dataLoaderDict = {}
        dataLoaderDict_number = {}
        for i, dset in enumerate(self.seed_datasets):
            if dset == "RACE":
                self.type3 = "RACE_combined_train_label"
                dataLoaderDict[dset] = self._get_dataloader_RACE()
                dataLoaderDict_number[i+1] = dset
            elif dset == "OBQA":
                dataLoaderDict[dset] = self._get_dataloader_OBQA()
                dataLoaderDict_number[i+1] = dset
            elif dset == "ARC":
                self.type2 = "ARC_train_label_combined"
                dataLoaderDict[dset] = self._get_dataloader_ARC()
                dataLoaderDict_number[i+1] = dset
            elif dset == "MCTest":
                dataLoaderDict[dset] = self._get_dataloader_MCTest()
                dataLoaderDict_number[i+1] = dset
            elif dset == "SQuADv2":
                dataLoaderDict[dset] = self._get_dataloader_SQuADv2()
                dataLoaderDict_number[i+1] = dset
            elif dset == "NarrativeQA":
                dataLoaderDict[dset] = self._get_dataloader_NarrativeQA()
                dataLoaderDict_number[i+1] = dset
            elif dset == "BoolQ":
                dataLoaderDict[dset] = self._get_dataloader_BoolQ()
                dataLoaderDict_number[i+1] = dset

        while True:

            num = random.randint(1, self.num_seed_datasets)
            #print(f"num: {num}")
            if num in [1,2,3,4]: # RACE, OBQA, ARC, MCTest
                #print(f"\n\n{len(dataLoaderDict[dataLoaderDict_number[num]])}\n\n")

                input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = \
                    next(dataLoaderDict[dataLoaderDict_number[num]])

                # here we reset the dataloader, get the next item, and continue as normal.
                if input_string is None or input_id is None:
                    print(f"\nResetting data loader: {dataLoaderDict_number[num]}\n")
                    if num == 1: dataLoaderDict[dataLoaderDict_number[num]] = self._get_dataloader_RACE()
                    elif num == 2: dataLoaderDict[dataLoaderDict_number[num]] = self._get_dataloader_OBQA()
                    elif num == 3: dataLoaderDict[dataLoaderDict_number[num]] = self._get_dataloader_ARC()
                    elif num == 4: dataLoaderDict[dataLoaderDict_number[num]] = self._get_dataloader_MCTest()
                    input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 = \
                        next(dataLoaderDict[dataLoaderDict_number[num]])

                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]
                lm_tok_id = self.tokenizer.encode_single(self.lm_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]
                    aux_label = aux_label + [pad_tok_id for _ in range(self.seq_len - len(aux_label))]
                    if not isinstance(label, int):  # integer only, not a list of integers. (not of this)
                        label = label + [pad_tok_id for _ in range(self.seq_len - len(label))]  #
                    else:
                        raise Exception(f"label should not be of type int, but is instead a list of integers")

                # aux_tok_1 represents the mode, i.e. the decoder <dec>
                input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                label_id = tf.cast(tf.convert_to_tensor(label), dtype=tf.dtypes.int64)
                aux_label = tf.cast(tf.convert_to_tensor(aux_label), dtype=tf.dtypes.int64)
                aoint_indices = tf.cast(tf.convert_to_tensor(aoint_indices), dtype=tf.dtypes.int64)
                sample_weights = tf.cast(tf.convert_to_tensor(sample_weights), dtype=tf.dtypes.int64)
                yield input_string, input_id, label_id, aux_label, aoint_indices, sample_weights
            elif num in [5,6,7]: # SQuADv2, NarrativeQA, BoolQ
                input_string, input_id, label, aux_label, sample_weights, aux_tok_1, aux_tok_2 = \
                    next(dataLoaderDict[dataLoaderDict_number[num]])

                aoint_indices = [0,0] # set to nothing, because the generator needs all outputs to have the same number
                # of items.

                if input_string is None or input_id is None:
                    print(f"\nResetting data loader: {dataLoaderDict_number[num]}\n")
                    if num == 5: dataLoaderDict[dataLoaderDict_number[num]] = self._get_dataloader_SQuADv2()
                    elif num == 6: dataLoaderDict[dataLoaderDict_number[num]] = self._get_dataloader_NarrativeQA()
                    elif num == 7: dataLoaderDict[dataLoaderDict_number[num]] = self._get_dataloader_BoolQ()
                    input_string, input_id, label, aux_label, sample_weights, aux_tok_1, aux_tok_2 = \
                        next(dataLoaderDict[dataLoaderDict_number[num]])


                pad_tok_id = self.tokenizer.encode_single(self.pad_tok)[0]

                if self.pad_to_max_length:
                    input_string = input_string + [self.pad_tok for _ in range(self.seq_len - len(input_string))]
                    input_id = input_id + [pad_tok_id for _ in range(self.seq_len - len(input_id))]
                    aux_label = aux_label + [pad_tok_id for _ in range(self.seq_len - len(aux_label))]
                    if not isinstance(label, int):  # integer only, not a list of integers. (not of this)
                        label = label + [pad_tok_id for _ in range(self.seq_len - len(label))]  #
                    else:
                        raise Exception(f"label should not be of type int, but is instead a list of integers")

                input_id = [self.cls_tok_id, aux_tok_1, aux_tok_2] + input_id

                input_string = tf.cast(tf.convert_to_tensor(input_string), dtype=tf.dtypes.string)
                input_id = tf.cast(tf.convert_to_tensor(input_id), dtype=tf.dtypes.int64)
                label_id = tf.cast(tf.convert_to_tensor(label), dtype=tf.dtypes.int64)
                aux_label = tf.cast(tf.convert_to_tensor(aux_label), dtype=tf.dtypes.int64)
                aoint_indices = tf.cast(tf.convert_to_tensor(aoint_indices), dtype=tf.dtypes.int64)
                sample_weights = tf.cast(tf.convert_to_tensor(sample_weights), dtype=tf.dtypes.int64)
                yield input_string, input_id, label_id, aux_label, aoint_indices, sample_weights



            # input_string, input_id, label, aux_label, aoint_indices, sample_weights, aux_tok_1, aux_tok_2 # MQA
            # input_string, input_id, label, aux_label, sample_weights, aux_tok_1, aux_tok_2 # GQA

            # return types.
            #tf.dtypes.string,tf.dtypes.int64,tf.dtypes.int64,tf.dtypes.int64, tf.dtypes.int64,tf.dtypes.int64

    def general_generator_text_to_text_format(self, seed_datasets=["RACE", "OBQA", "ARC", "MCTest",
                                                                   "SQuADv2", "NarrativeQA", "BoolQ"],
                                              min_train_size=1400, batch_size=32, shuffle=True):
        self.seed_datasets = seed_datasets
        self.min_train_size = 1400 # This will be the number of instances from each training set...
        self.batch_size = batch_size
        self.num_seed_datasets = len(seed_datasets)
        self.shuffle = True

        # NOTE: this generator goes on infinitely, need to stop manually, ctrl + C in linux terminal.
        generator = tf.data.Dataset.from_generator(self.getGeneralTrainer,
                                                   output_types=(tf.dtypes.string,
                                                                 tf.dtypes.int64,
                                                                 tf.dtypes.int64,
                                                                 tf.dtypes.int64,
                                                                 tf.dtypes.int64,
                                                                 tf.dtypes.int64))

        return generator


if __name__ == "__main__":


    config = V4ConfigMediumSize(strategy=None, batch_size=2, loss_object=None, learning_rate=None, gpt2_117=True,
                                tokenizer="gpt2")

    '''
    filepaths = {"OBQA_train": "/large_data/OpenBookQA-V1-Sep2018/Data/Main/train.jsonl",
                 "OBQA_val": "/large_data/OpenBookQA-V1-Sep2018/Data/Main/dev.jsonl",
                 "OBQA_test": "/large_data/OpenBookQA-V1-Sep2018/Data/Main/test.jsonl"}
    dloader = MasterDataLoaderTF(filepaths=filepaths, seq_len=config.max_seq_len_dec,
                                 batch_size=config.batch_size, tokenizer=config.tokenizer)
    generator = dloader.get_generator("OBQA_train_label", False, override_lm=True).batch(1)
    '''

    '''
    filepaths = {"ARC_test_easy": "/large_data/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Test.jsonl",
                 "ARC_test_challene": "/large_data/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Test.jsonl",
                 "ARC_train_easy": "/large_data/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Train.jsonl",
                 "ARC_train_challenge":"/large_data/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Train.jsonl",
                 "ARC_val_easy": "/large_data/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Dev.jsonl",
                 "ARC_val_challenge":"/large_data/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Dev.jsonl"}
    dloader = MasterDataLoaderTF(filepaths=filepaths, seq_len=config.max_seq_len_dec,
                                 batch_size=config.batch_size, tokenizer=config.tokenizer)

    #generator = dloader.get_generator("ARC_train_label_combined", True, override_lm=True).batch(1)
    generator = dloader.get_generator("ARC_test_easy", True, override_lm=True).batch(1)
    '''
    #batch_ = 1

    #filepaths = {"MCTest_train": "/large_data/MCTest/",
    #             "MCTest_val": "/large_data/MCTest/",
    #             "MCTest_test": "/large_data/MCTest/"}
    #dloader = MasterDataLoaderTF(filepaths=filepaths, seq_len=config.max_seq_len_dec,
    #                             batch_size=config.batch_size, tokenizer=config.tokenizer)
    #generator = dloader.get_generator("MCTest_val_label", False, override_lm=True).batch(1)
    #generator = dloader.get_generator("MCTest_test", False, override_lm=True).batch(1)

    '''
    for (input_string, input_id, all_labels, correct_ao, aoint_indices) in generator: # for test.
        print(f"batch: {batch_}")
        # input_string, input_id, label_id, aux_label, aoint_indices, sample_weights
        print(f"input_string: {input_string.shape} \t inp_str: {input_string} \n"
              f"input_id.shape: {input_id.shape} \t inp_id: {input_id} \n"
              f"all_labels: {all_labels}\n"
              f"correct_ao: {correct_ao}\n"
              f"aoint_indices: {aoint_indices}")
        if batch_ == 1: break
        batch_ += 1
    print(f"batch_ counter: {batch_}")
    '''

    filepaths = {"NarrativeQA_train": "/large_data/NarrativeQA/narrativeqa-master/",
                "SQuAD_train_default": "/large_data/SQuAD 2.0/train-v2.0.json",
                "RACE_high_train": "/large_data/RACE/train/high/",
                "RACE_middle_train": "/large_data/RACE/train/middle/",
                "BoolQ_train": "/large_data/BoolQ/train.jsonl",
                "OBQA_train": "/large_data/OpenBookQA-V1-Sep2018/Data/Main/train.jsonl",
                "ARC_train_easy": "/large_data/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Train.jsonl",
                "ARC_train_challenge": "/large_data/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Train.jsonl",
                "MCTest_train": "/large_data/MCTest/"}

    dloader_train = MasterDataLoaderTF(filepaths=filepaths, seq_len=config.max_seq_len_dec,
    batch_size = config.batch_size, tokenizer = config.tokenizer)
    # generator_train = dloader_train.get_generator(type="NarrativeQA_train", shuffle=True, override_lm=True).batch(config.batch_size)
    generator = dloader_train.general_generator_text_to_text_format(seed_datasets=["RACE", "OBQA","ARC", "MCTest",
                                                                           "SQuADv2", "NarrativeQA", "BoolQ"],
    min_train_size = 1400, batch_size = config.batch_size, shuffle = True).batch(32)

    # for train_label
    batch_ = 1
    for (input_string, input_id, label_id, aux_label, aoint_indices, sample_weights) in generator:
        print(f"batch: {batch_}")
        # input_string, input_id, label_id, aux_label, aoint_indices, sample_weights
        #print(f"input_string: {input_string.shape} \t inp_str: {input_string} \n"
        #      f"input_id.shape: {input_id.shape} \t inp_id: {input_id} \n"
        #      f"answers.shape: {label_id.shape} \t answers: {label_id} \n"
        #      f"aux_label.shape: {aux_label.shape}\t aux_label: {aux_label}\n"
        #      f"aoint_indices: {aoint_indices}\tsample_weights: {sample_weights}")
        if batch_ == 100000: break
        batch_ += 1
    print(f"batch_ counter: {batch_}")


    '''
    filepaths = {"SQuAD_train_default": "/large_data/SQuAD 2.0/train-v2.0.json",
                 "SQuAD_test_default": "/large_data/SQuAD 2.0/dev-v2.0.json"}
    dloader = MasterDataLoaderTF(filepaths=filepaths, seq_len=config.max_seq_len_dec,
                                 batch_size=config.batch_size, tokenizer=config.tokenizer)
    generator = dloader.get_generator("SQuAD_test_default", True, override_lm=True).batch(3)

    batch_ = 1
    for (input_string, input_id, answers) in generator:
        print(f"batch: {batch_}")
        print(f"input_string: {input_string.shape} \t inp_str: {input_string} \n"
              f"input_id.shape: {input_id.shape} \t inp_id: {input_id} \n"
              f"answers.shape: {answers.shape} \t answers: {answers} \n")
        if batch_ == 3: break
        batch_ += 1
    print(f"batch_ counter: {batch_}")
    '''
    '''
    filepaths = {"BoolQ_train":"/large_data/BoolQ/train.jsonl",
                 "BoolQ_val":"/large_data/BoolQ/dev.jsonl"}
    dloader = MasterDataLoaderTF(filepaths=filepaths, seq_len=config.max_seq_len_dec, batch_size=config.batch_size,
                                 tokenizer=config.tokenizer)
    generator = dloader.get_generator(type="BoolQ_train", shuffle=True, override_lm=True).batch(1)
    '''
    '''
    filepaths = {"NarrativeQA_train": "/large_data/NarrativeQA/narrativeqa-master/",
                 "NarrativeQA_val": "/large_data/NarrativeQA/narrativeqa-master/",
                 "NarrativeQA_test": "/large_data/NarrativeQA/narrativeqa-master/"}
    dloader = MasterDataLoaderTF(filepaths=filepaths, seq_len=config.max_seq_len_dec, batch_size=config.batch_size,
                                 tokenizer=config.tokenizer)
    generator = dloader.get_generator(type="NarrativeQA_test", shuffle=True, override_lm=True).batch(1)

    
    batch_ = 1
    for (input_string, input_id, answer) in generator:
        print(f"batch: {batch_}")
        print(f"input_string: {input_string.shape} \t inp_str: {input_string} \n"
              f"input_id.shape: {input_id.shape} \t inp_id: {input_id} \n"
              #f"label_id.shape: {label_id.shape} \t tar_id: {label_id} \n" 
              #f"aux_label.shape: {aux_label.shape} \t aux_label: {aux_label}"
              #f"sample_weights.shape: {sample_weights} \t sample_weights: {sample_weights}")
              f"answer: {answer}")
        if batch_ == 4: break
        batch_ += 1
    print(f"batch_ counter: {batch_}")

    '''
    #config = V4ConfigMediumSize(strategy=None, batch_size=2, loss_object=None, learning_rate=None, gpt2_117=True,
    #                            tokenizer="gpt2")


    '''
    #filepaths = {"C4_nm_pre_train":"/large_data/C4/en/"}
    filepaths = {"RACE_high_train": "/large_data/RACE/train/high/",
                 "RACE_high_val": "/large_data/RACE/dev/high/",
                 "RACE_middle_train": "/large_data/RACE/train/middle/",
                 "RACE_middle_val": "/large_data/RACE/dev/middle/"}

    dloader = MasterDataLoaderTF(filepaths=filepaths, seq_len=config.max_seq_len_dec,
                                 batch_size=config.batch_size, tokenizer=config.tokenizer)

  
    #generator = dloader.get_generator(type="C4_pretrain_dec", shuffle=False).batch(1)
    generator = dloader.get_generator("RACE_combined_train_label", False, override_lm=True).batch(2)

    batch_ = 1
    for (input_string, input_id, label_id, aux_loss, aoint_indices, sample_weights) in generator:
        print(f"batch: {batch_}")
        #print(f"input_string: {input_string.shape} \t inp_str: {input_string} \n"
        #      f"input_id.shape: {input_id.shape} \t inp_id: {None} \n"
        #      f"label_id.shape: {label_id.shape} \t tar_id: {label_id} \n"
        #      f"aoint_indices.shape: {aoint_indices.shape} \t aoint_indices: {aoint_indices}"
        #      f"sample_weights.shape: {sample_weights} \t sample_weights: {sample_weights}")
        if batch_ == 2: break
        batch_ += 1
    print(f"batch_ counter: {batch_}")
    '''
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