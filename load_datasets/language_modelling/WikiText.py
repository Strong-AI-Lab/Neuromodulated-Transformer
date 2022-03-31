import os
from os import listdir
from os.path import isfile, join

import random
import gzip
import json
import re
import sys
import regex
import copy

import pandas as pd

sys.path.append("../..")


from models.config_model import *

class WikiTextDataLoader:
    '''
    The implementation of a dataloader for the WikiText dataset to be utilzed by another higher level data loader.
    '''

    def __init__(self, strategy: str, filepath: str, enc_tok: str, dec_tok: str, mlm_tok: str, lm_tok: str,
                 start_tok: str, end_tok: str, cls_tok: str, sep_tok: str, mask_tok: str, pad_tok: str,
                 seq_len: int, pad: bool,
                 a1: str, a2: str, a3: str, a4: str, a5: str, a6: str, a7: str, a8: str, a9: str,
                 passage: str, p1: str, p2: str, p3: str, p4: str, p5: str, p6: str, p7: str, p8: str, p9: str,
                 mqa: str, pmqa: str, bmqa: str, peentailment: str, pbentailment: str,
                 pcoreference: str, bcoreference: str, psentiment: str, pgqa: str, psqa: str, gqa: str, pbqa: str,
                 placeholder: str, translation: str, hypothesis: str, question: str, metacognition: str,
                 unk_rs: str, aoint_rs: str, highlighting_rs: str, reread_rs: str, summarize_rs: str,
                 paraphrase_rs: str, tokenizer=None, sliding_window=32,
                 aux_toks=[""], shuffle=False):

        self.sliding_window = sliding_window
        self.aux_toks = aux_toks
        self.shuffle = shuffle

        self.data = None

        self.filepath = filepath # path to the file containing the data.
        self.strategy = strategy  # train, validation, test -- train and validation will be the same.
        assert self.strategy in ["default", "gpt2-remove"], f"The strategy {self.strategy} is not supported!"

        self.enc_tok = enc_tok
        self.dec_tok = dec_tok
        self.mlm_tok = mlm_tok
        self.lm_tok = lm_tok
        self.start_tok = start_tok
        self.end_tok = end_tok
        self.cls_tok = cls_tok
        self.sep_tok = sep_tok
        self.mask_tok = mask_tok
        self.pad_tok = pad_tok

        self.pad = pad
        self.seq_len = seq_len

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

        self.unk_rs = unk_rs
        self.aoint_rs = aoint_rs
        self.highlighting_rs = highlighting_rs
        self.reread_rs = reread_rs
        self.summarize_rs = summarize_rs
        self.paraphrase_rs = paraphrase_rs

        self.tokenizer = tokenizer

        if tokenizer is not None:
            ###### <s> ######
            self.start_tok_id = self.tokenizer.encode_single(self.start_tok)
            if len(self.start_tok_id) != 1 and (isinstance(self.start_tok_id, list)):
                raise Exception(
                    f"The number of ids the start token is encoded into should be one, got {self.start_tok_id}!")
            else:
                self.start_tok_id = self.start_tok_id[0]
            print(f"The start token id is: {self.start_tok_id}")

            ###### </s> ######
            self.end_tok_id = self.tokenizer.encode_single(self.end_tok)
            if len(self.end_tok_id) != 1 and (isinstance(self.end_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.end_tok_id)}!")
            else:
                self.end_tok_id = self.end_tok_id[0]
            print(f"The end token id is: {self.end_tok_id}")

            ###### <cls> ######
            self.cls_tok_id = self.tokenizer.encode_single(self.cls_tok)
            if len(self.cls_tok_id) != 1 and (isinstance(self.cls_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.cls_tok_id)}!")
            else:
                self.cls_tok_id = self.cls_tok_id[0]
            print(f"The cls token id is: {self.cls_tok_id}")

            ###### <sep> ######
            self.sep_tok_id = self.tokenizer.encode_single(self.sep_tok)
            if len(self.sep_tok_id) != 1 and (isinstance(self.sep_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.sep_tok_id)}!")
            else:
                self.sep_tok_id = self.sep_tok_id[0]
            print(f"The sep token id is: {self.sep_tok_id}")

            ###### <mask> ######
            self.mask_tok_id = self.tokenizer.encode_single(self.mask_tok)
            if len(self.mask_tok_id) != 1 and (isinstance(self.mask_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.mask_tok_id)}!")
            else:
                self.mask_tok_id = self.mask_tok_id[0]
            print(f"The mask token id is: {self.mask_tok_id}")

            ###### <enc> ######
            self.enc_tok_id = self.tokenizer.encode_single(self.enc_tok)
            if len(self.enc_tok_id) != 1 and (isinstance(self.enc_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.enc_tok_id)}!")
            else:
                self.enc_tok_id = self.enc_tok_id[0]
            print(f"The enc token id is: {self.enc_tok_id}")

            ###### <dec> ######
            self.dec_tok_id = self.tokenizer.encode_single(self.dec_tok)
            if len(self.dec_tok_id) != 1 and (isinstance(self.dec_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.dec_tok_id)}!")
            else:
                self.dec_tok_id = self.dec_tok_id[0]
            print(f"The dec token id is: {self.dec_tok_id}")

            ###### <mlm> ######
            self.mlm_tok_id = self.tokenizer.encode_single(self.mlm_tok)
            if len(self.mlm_tok_id) != 1 and (isinstance(self.mlm_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.mlm_tok_id)}!")
            else:
                self.mlm_tok_id = self.mlm_tok_id[0]
            print(f"The mlm token id is: {self.mlm_tok_id}")

            ###### <lm> ######
            self.lm_tok_id = self.tokenizer.encode_single(self.lm_tok)
            if len(self.lm_tok_id) != 1 and (isinstance(self.lm_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.lm_tok_id)}!")
            else:
                self.lm_tok_id = self.lm_tok_id[0]
            print(f"The lm token id is: {self.lm_tok_id}")

            ###### <pad> ######
            self.pad_tok_id = self.tokenizer.encode_single(self.pad_tok)
            if len(self.pad_tok_id) != 1 and (isinstance(self.pad_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.pad_tok_id)}!")
            else:
                self.pad_tok_id = self.pad_tok_id[0]
            print(f"The pad token id is: {self.pad_tok_id}")

            ###### <a1> ######
            self.a1_tok_id = self.tokenizer.encode_single(self.a1)
            if len(self.a1_tok_id) != 1 and (isinstance(self.a1_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.a1_tok_id)}!")
            else:
                self.a1_tok_id = self.a1_tok_id[0]
            print(f"The a1 token id is: {self.a1_tok_id}")

            ###### <a2> ######
            self.a2_tok_id = self.tokenizer.encode_single(self.a2)
            if len(self.a2_tok_id) != 1 and (isinstance(self.a2_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.a2_tok_id)}!")
            else:
                self.a2_tok_id = self.a2_tok_id[0]
            print(f"The a2 token id is: {self.a2_tok_id}")

            ###### <a3> ######
            self.a3_tok_id = self.tokenizer.encode_single(self.a3)
            if len(self.a3_tok_id) != 1 and (isinstance(self.a3_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.a3_tok_id)}!")
            else:
                self.a3_tok_id = self.a3_tok_id[0]
            print(f"The a3 token id is: {self.a3_tok_id}")

            ###### <a4> ######
            self.a4_tok_id = self.tokenizer.encode_single(self.a4)
            if len(self.a4_tok_id) != 1 and (isinstance(self.a4_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.a4_tok_id)}!")
            else:
                self.a4_tok_id = self.a4_tok_id[0]
            print(f"The a4 token id is: {self.a4_tok_id}")

            ###### <a5> ######
            self.a5_tok_id = self.tokenizer.encode_single(self.a5)
            if len(self.a5_tok_id) != 1 and (isinstance(self.a5_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.a5_tok_id)}!")
            else:
                self.a5_tok_id = self.a5_tok_id[0]
            print(f"The a5 token id is: {self.a5_tok_id}")

            ###### <a6> ######
            self.a6_tok_id = self.tokenizer.encode_single(self.a6)
            if len(self.a6_tok_id) != 1 and (isinstance(self.a6_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.a6_tok_id)}!")
            else:
                self.a6_tok_id = self.a6_tok_id[0]
            print(f"The a6 token id is: {self.a6_tok_id}")

            ###### <a7> ######
            self.a7_tok_id = self.tokenizer.encode_single(self.a7)
            if len(self.a7_tok_id) != 1 and (isinstance(self.a7_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.a7_tok_id)}!")
            else:
                self.a7_tok_id = self.a7_tok_id[0]
            print(f"The a7 token id is: {self.a7_tok_id}")

            ###### <a8> ######
            self.a8_tok_id = self.tokenizer.encode_single(self.a8)
            if len(self.a8_tok_id) != 1 and (isinstance(self.a8_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.a8_tok_id)}!")
            else:
                self.a8_tok_id = self.a8_tok_id[0]
            print(f"The a8 token id is: {self.a8_tok_id}")

            ###### <a9> ######
            self.a9_tok_id = self.tokenizer.encode_single(self.a9)
            if len(self.a9_tok_id) != 1 and (isinstance(self.a9_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.a9_tok_id)}!")
            else:
                self.a9_tok_id = self.a9_tok_id[0]
            print(f"The a9 token id is: {self.a9_tok_id}")

            ###### <passage> ######
            self.passage_tok_id = self.tokenizer.encode_single(self.passage)
            if len(self.passage_tok_id) != 1 and (isinstance(self.passage_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.passage_tok_id)}!")
            else:
                self.passage_tok_id = self.passage_tok_id[0]
            print(f"The passage token id is: {self.passage_tok_id}")

            ###### <p1> ######
            self.p1_tok_id = self.tokenizer.encode_single(self.p1)
            if len(self.p1_tok_id) != 1 and (isinstance(self.p1_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.p1_tok_id)}!")
            else:
                self.p1_tok_id = self.p1_tok_id[0]
            print(f"The p1 token id is: {self.p1_tok_id}")

            ###### <p2> ######
            self.p2_tok_id = self.tokenizer.encode_single(self.p2)
            if len(self.p2_tok_id) != 1 and (isinstance(self.p2_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.p2_tok_id)}!")
            else:
                self.p2_tok_id = self.p2_tok_id[0]
            print(f"The p2 token id is: {self.p2_tok_id}")

            ###### <p3> ######
            self.p3_tok_id = self.tokenizer.encode_single(self.p3)
            if len(self.p3_tok_id) != 1 and (isinstance(self.p3_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.p3_tok_id)}!")
            else:
                self.p3_tok_id = self.p3_tok_id[0]
            print(f"The p3 token id is: {self.p3_tok_id}")

            ###### <p4> ######
            self.p4_tok_id = self.tokenizer.encode_single(self.p4)
            if len(self.p4_tok_id) != 1 and (isinstance(self.p4_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.p4_tok_id)}!")
            else:
                self.p4_tok_id = self.p4_tok_id[0]
            print(f"The p4 token id is: {self.p4_tok_id}")

            ###### <p5> ######
            self.p5_tok_id = self.tokenizer.encode_single(self.p5)
            if len(self.p5_tok_id) != 1 and (isinstance(self.p5_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.p5_tok_id)}!")
            else:
                self.p5_tok_id = self.p5_tok_id[0]
            print(f"The p5 token id is: {self.p5_tok_id}")

            ###### <p6> ######
            self.p6_tok_id = self.tokenizer.encode_single(self.p6)
            if len(self.p6_tok_id) != 1 and (isinstance(self.p6_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.p6_tok_id)}!")
            else:
                self.p6_tok_id = self.p6_tok_id[0]
            print(f"The p6 token id is: {self.p6_tok_id}")

            ###### <p7> ######
            self.p7_tok_id = self.tokenizer.encode_single(self.p7)
            if len(self.p7_tok_id) != 1 and (isinstance(self.p7_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.p7_tok_id)}!")
            else:
                self.p7_tok_id = self.p7_tok_id[0]
            print(f"The p7 token id is: {self.p7_tok_id}")

            ###### <p8> ######
            self.p8_tok_id = self.tokenizer.encode_single(self.p8)
            if len(self.p8_tok_id) != 1 and (isinstance(self.p8_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.p8_tok_id)}!")
            else:
                self.p8_tok_id = self.p8_tok_id[0]
            print(f"The p8 token id is: {self.p8_tok_id}")

            ###### <p9> ######
            self.p9_tok_id = self.tokenizer.encode_single(self.p9)
            if len(self.p9_tok_id) != 1 and (isinstance(self.p9_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.p9_tok_id)}!")
            else:
                self.p9_tok_id = self.p9_tok_id[0]
            print(f"The p9 token id is: {self.p9_tok_id}")

            ###### <h> ######
            self.hypothesis_tok_id = self.tokenizer.encode_single(self.hypothesis)
            if len(self.hypothesis_tok_id) != 1 and (isinstance(self.hypothesis_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.hypothesis_tok_id)}!")
            else:
                self.hypothesis_tok_id = self.hypothesis_tok_id[0]
            print(f"The h token id is: {self.hypothesis_tok_id}")

            ###### <q> ######
            self.question_tok_id = self.tokenizer.encode_single(self.question)
            if len(self.question_tok_id) != 1 and (isinstance(self.question_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.question_tok_id)}!")
            else:
                self.question_tok_id = self.question_tok_id[0]
            print(f"The q token id is: {self.question_tok_id}")

            ###### <mqa> ######
            self.mqa_tok_id = self.tokenizer.encode_single(self.mqa)
            if len(self.mqa_tok_id) != 1 and (isinstance(self.mqa_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.mqa_tok_id)}!")
            else:
                self.mqa_tok_id = self.mqa_tok_id[0]
            print(f"The mqa token id is: {self.mqa_tok_id}")

            ###### <pmqa> ######
            self.pmqa_tok_id = self.tokenizer.encode_single(self.pmqa)
            if len(self.pmqa_tok_id) != 1 and (isinstance(self.pmqa_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.pmqa_tok_id)}!")
            else:
                self.pmqa_tok_id = self.pmqa_tok_id[0]
            print(f"The pmqa token id is: {self.pmqa_tok_id}")

            ###### <bmqa> ######
            self.bmqa_tok_id = self.tokenizer.encode_single(self.bmqa)
            if len(self.bmqa_tok_id) != 1 and (isinstance(self.bmqa_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.bmqa_tok_id)}!")
            else:
                self.bmqa_tok_id = self.bmqa_tok_id[0]
            print(f"The bmqa token id is: {self.bmqa_tok_id}")

            ###### <peentailment> ######
            self.peentailment_tok_id = self.tokenizer.encode_single(self.peentailment)
            if len(self.peentailment_tok_id) != 1 and (isinstance(self.peentailment_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.peentailment_tok_id)}!")
            else:
                self.peentailment_tok_id = self.peentailment_tok_id[0]
            print(f"The peentailment token id is: {self.peentailment_tok_id}")

            ###### <pbentailment> ######
            self.pbentailment_tok_id = self.tokenizer.encode_single(self.pbentailment)
            if len(self.pbentailment_tok_id) != 1 and (isinstance(self.pbentailment_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.pbentailment_tok_id)}!")
            else:
                self.pbentailment_tok_id = self.pbentailment_tok_id[0]
            print(f"The pbentailment token id is: {self.pbentailment_tok_id}")

            ###### <pcoreference> ######
            self.pcoreference_tok_id = self.tokenizer.encode_single(self.pcoreference)
            if len(self.pcoreference_tok_id) != 1 and (isinstance(self.pcoreference_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.pcoreference_tok_id)}!")
            else:
                self.pcoreference_tok_id = self.pcoreference_tok_id[0]
            print(f"The pcoreference token id is: {self.pcoreference_tok_id}")

            ###### <bcoreference> ######
            self.bcoreference_tok_id = self.tokenizer.encode_single(self.bcoreference)
            if len(self.bcoreference_tok_id) != 1 and (isinstance(self.bcoreference_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.bcoreference_tok_id)}!")
            else:
                self.bcoreference_tok_id = self.bcoreference_tok_id[0]
            print(f"The pcoreference token id is: {self.bcoreference_tok_id}")

            ###### <psentiment> ######
            self.psentiment_tok_id = self.tokenizer.encode_single(self.psentiment)
            if len(self.psentiment_tok_id) != 1 and (isinstance(self.psentiment_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.psentiment_tok_id)}!")
            else:
                self.psentiment_tok_id = self.psentiment_tok_id[0]
            print(f"The psentiment token id is: {self.psentiment_tok_id}")

            ###### <pgqa> ######
            self.pgqa_tok_id = self.tokenizer.encode_single(self.pgqa)
            if len(self.pgqa_tok_id) != 1 and (isinstance(self.pgqa_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.pgqa_tok_id)}!")
            else:
                self.pgqa_tok_id = self.pgqa_tok_id[0]
            print(f"The pgqa token id is: {self.pgqa_tok_id}")

            ###### <psqa> ######
            self.psqa_tok_id = self.tokenizer.encode_single(self.psqa)
            if len(self.psqa_tok_id) != 1 and (isinstance(self.psqa_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.psqa_tok_id)}!")
            else:
                self.psqa_tok_id = self.psqa_tok_id[0]
            print(f"The psqa token id is: {self.psqa_tok_id}")

            ###### <gqa> ######
            self.gqa_tok_id = self.tokenizer.encode_single(self.gqa)
            if len(self.gqa_tok_id) != 1 and (isinstance(self.gqa_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.gqa_tok_id)}!")
            else:
                self.gqa_tok_id = self.gqa_tok_id[0]
            print(f"The gqa token id is: {self.gqa_tok_id}")

            ###### <pgqa> ######
            self.pbqa_tok_id = self.tokenizer.encode_single(self.pbqa)
            if len(self.pbqa_tok_id) != 1 and (isinstance(self.pbqa_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.pbqa_tok_id)}!")
            else:
                self.pbqa_tok_id = self.pbqa_tok_id[0]
            print(f"The pbqa token id is: {self.pbqa_tok_id}")

            ###### <placeholder> ######
            self.placeholder_tok_id = self.tokenizer.encode_single(self.placeholder)
            if len(self.placeholder_tok_id) != 1 and (isinstance(self.placeholder_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.placeholder_tok_id)}!")
            else:
                self.placeholder_tok_id = self.placeholder_tok_id[0]
            print(f"The placeholder token id is: {self.placeholder_tok_id}")

            ###### <translation> ######
            self.translation_tok_id = self.tokenizer.encode_single(self.translation)
            if len(self.translation_tok_id) != 1 and (isinstance(self.translation_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.translation_tok_id)}!")
            else:
                self.translation_tok_id = self.translation_tok_id[0]
            print(f"The translation token id is: {self.translation_tok_id}")

            ###### <metacognition> ######
            self.metacognition_tok_id = self.tokenizer.encode_single(self.metacognition)
            if len(self.metacognition_tok_id) != 1 and (isinstance(self.metacognition_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.metacognition_tok_id)}!")
            else:
                self.metacognition_tok_id = self.metacognition_tok_id[0]
            print(f"The metacognition token id is: {self.metacognition_tok_id}")

            ###### <unk_rs> ######
            self.unk_rs_tok_id = self.tokenizer.encode_single(self.unk_rs)
            if len(self.unk_rs_tok_id) != 1 and (isinstance(self.unk_rs_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.unk_rs_tok_id)}!")
            else:
                self.unk_rs_tok_id = self.unk_rs_tok_id[0]
            print(f"The unk_rs token id is: {self.unk_rs_tok_id}")

            ###### <aoint_rs> ######
            self.aoint_rs_tok_id = self.tokenizer.encode_single(self.aoint_rs)
            if len(self.aoint_rs_tok_id) != 1 and (isinstance(self.aoint_rs_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.aoint_rs_tok_id)}!")
            else:
                self.aoint_rs_tok_id = self.aoint_rs_tok_id[0]
            print(f"The aoint_rs token id is: {self.aoint_rs_tok_id}")

            ###### <highlighting_rs> ######
            self.highlighting_rs_tok_id = self.tokenizer.encode_single(self.highlighting_rs)
            if len(self.highlighting_rs_tok_id) != 1 and (isinstance(self.highlighting_rs_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.highlighting_rs_tok_id)}!")
            else:
                self.highlighting_rs_tok_id = self.highlighting_rs_tok_id[0]
            print(f"The highlighting_rs token id is: {self.highlighting_rs_tok_id}")

            ###### <reread_rs> ######
            self.reread_rs_tok_id = self.tokenizer.encode_single(self.reread_rs)
            if len(self.reread_rs_tok_id) != 1 and (isinstance(self.reread_rs_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.reread_rs_tok_id)}!")
            else:
                self.reread_rs_tok_id = self.reread_rs_tok_id[0]
            print(f"The reread_rs token id is: {self.reread_rs_tok_id}")

            ###### <summarize_rs> ######
            self.summarize_rs_tok_id = self.tokenizer.encode_single(self.summarize_rs)
            if len(self.summarize_rs_tok_id) != 1 and (isinstance(self.summarize_rs_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.summarize_rs_tok_id)}!")
            else:
                self.summarize_rs_tok_id = self.summarize_rs_tok_id[0]
            print(f"The summarize_rs token id is: {self.summarize_rs_tok_id}")

            ###### <paraphrase_rs> ######
            self.paraphrase_rs_tok_id = self.tokenizer.encode_single(self.paraphrase_rs)
            if len(self.paraphrase_rs_tok_id) != 1 and (isinstance(self.paraphrase_rs_tok_id, list)):
                raise Exception(
                    f"The number of ids the end token is encoded into should be one, got {len(self.paraphrase_rs_tok_id)}!")
            else:
                self.paraphrase_rs_tok_id = self.paraphrase_rs_tok_id[0]
            print(f"The paraphrase_rs token id is: {self.paraphrase_rs_tok_id}")

        self.strategy = strategy
        assert self.strategy in ["default", "gpt2-remove"], f"The strategy {self.strategy} is not supported!"

        # self.main_heading_pattern = "= [^=]*[^=] = \n"
        self.main_heading_pattern = "= [^=]*[^=] = \n"
        self.any_heading_pattern = "= [= ]*[^=]*[^=] [= ]*="

        self.data_dict = {}  # each article heading will be put here with the associated input.

        self._process_raw_file()

    def _process_raw_file(self):

        data = None
        with open(self.filepath, "r") as f:
            data = f.readlines()

        if self.strategy == "default":

            counter = 0
            main_heading = None
            document_content = ""
            for i in range(len(data)):
                if re.search(self.main_heading_pattern, data[i]): # if a heading do the following.
                    # below if tests for the main heading.
                    if document_content == "": # append the heading here as well as we want it in the input as well.
                        counter += 1
                        if counter > 1: raise Exception(f"counter should never reach a value greater than 1!")
                        document_content = re.sub("\n$", str(self.end_tok), data[i])
                    else:
                        assert main_heading is not None, f"Error in the execution of the code, main_heading should not be None here."
                        self.data_dict[main_heading] = document_content
                        document_content = re.sub("\n$", str(self.end_tok), data[i]) # document content is set to the main heading.
                    main_heading = re.sub("\n$", "", data[i])  # for the heading \n is not wanted.
                    continue
                elif data[i] == " \n": continue # just skip this as it is pointless.
                else: # the content of the heading...
                    document_content += re.sub("\n$", str(self.end_tok), data[i])
            self.data_dict[main_heading] = document_content
            #print(f"self.data_dict[main_heading]: {self.data_dict[main_heading]}")
            #print(f"\nmain_heading: {main_heading}\n")

        elif self.strategy == "gpt2-remove":

            main_heading = None
            document_content = ""
            for i in range(len(data)):
                if re.search(self.main_heading_pattern, data[i]):  # if a heading do the following.
                    # below if tests for the main heading.
                    if document_content != "":
                        self.data_dict[main_heading] = document_content
                        document_content = ""
                    main_heading = re.sub("\n$", "", data[i])  # for the heading \n is not wanted.
                    continue
                elif data[i] == " \n":
                    continue  # just skip this as it is pointless.
                else:  # the content of the heading...
                    txt = data[i]
                    txt = re.sub(self.any_heading_pattern, "", txt) # removes all headings from the context.
                    txt = re.sub("\n", "", txt) # remove all end of lines.
                    txt = re.sub("<unk>", "", txt) # remove all <unk> tokens.
                    document_content += txt
                    #document_content += re.sub("\n$", str(self.end_tok), data[i])
            self.data_dict[main_heading] = document_content
            #print(f"self.data_dict[main_heading]: {self.data_dict[main_heading]}")
            #print(f"\nmain_heading: {main_heading}\n")

        else:
            raise Exception(f"Invalid strategy: {self.strategy}")

    def default_tok_strategy_iterator(self):

        nm_aux_tok = []
        for word in self.aux_tok:
            w = self.tokenizer.encode_single(word)
            if len(w) > 1: raise Exception(f"{word} should only have on id associated with it in the tokenizer. Something went wrong!")
            else: w = w[0]
            nm_aux_tok.append(w)

        keys = list(self.data_dict.keys())
        #print(keys)
        if self.shuffle:
            random.shuffle(keys) # shuffles inplace.

        for i, key in enumerate(keys):

            article_ids = self.tokenizer.encode_single(self.data_dict[key])
            #article_ids = article_ids["input_ids"]
            start_index = 0
            end_index = self.max_seq_len
            tar_inp = None
            nm_tar_inp = None
            tar_real = None
            article_len = len(article_ids)
            while True:
                if start_index >= article_len-1: break # breaking condition for the article.

                if end_index < article_len-1:
                    tar_inp = article_ids[start_index:end_index]
                    tar_real = article_ids[start_index+1:end_index+1]
                else:
                    tar_inp = article_ids[start_index:article_len-1] # -1 so there is an output token for the target.
                    tar_real = article_ids[start_index+1:article_len] # i.e. shifted to the right.

                if self.pad:
                    tar_inp = tar_inp + [self.pad_tok_id for _ in range(self.max_seq_len-len(tar_inp))]
                    tar_real = tar_real + [self.pad_tok_id for _ in range(self.max_seq_len-len(tar_real))]

                nm_inp = nm_aux_tok + tar_inp

                #tar_inp = tf.cast(tf.convert_to_tensor(np.asarray(tar_inp)), dtype=tf.dtypes.int64)
                tar_real = tf.cast(tf.convert_to_tensor(np.asarray(tar_real)), dtype=tf.dtypes.int64)
                nm_inp = tf.cast(tf.convert_to_tensor(np.asarray(nm_inp)), dtype=tf.dtypes.int64)

                start_index = end_index
                end_index += self.max_seq_len

                #yield tar_inp, tar_real, nm_inp
                yield nm_inp, tar_real

    def sliding_window_article_iterator(self):

        nm_aux_tok = []
        for word in self.aux_tok:
            w = self.tokenizer.encode_single(word)
            if len(w) == 1: w = w[0]
            else: raise Exception(f"{word} should only have on id associated with it in the tokenizer. Something went wrong!")
            nm_aux_tok.append(w)

        keys = list(self.data_dict.keys())
        #print(keys)
        if self.shuffle:
            random.shuffle(keys) # shuffles inplace.

        for i, key in enumerate(keys):

            #article_ids = self.tokenizer.encode_single(self.data_dict[key])
            article_ids = self.tokenizer.encode_single_id_string_max_seq_len(
                self.data_dict[key], max_seq_len=10000000)[0]#[1:-2] # 0 is ids, 1 is strings. # no <s> or </s>
            #article_ids = article_ids["input_ids"]
            start_index = 0
            end_index = self.max_seq_len
            tar_inp = None
            nm_tar_inp = None
            tar_real = None
            article_len = len(article_ids)
            counter = 0
            while True:

                if start_index >= article_len-1: break  # breaking condition for the article.
                # article_len -1 because want tar_real to be </s> at the very least if start_index is article_len-2
                if end_index < article_len-1:
                    tar_inp = article_ids[start_index:end_index]
                    tar_real = article_ids[start_index+1:end_index+1]
                else:
                    tar_inp = article_ids[start_index:article_len-1] # -1 so there is an output token for the target.
                    tar_real = article_ids[start_index+1:article_len] # i.e. shifted to the right.

                if self.pad:
                    tar_inp = tar_inp + [self.pad_tok_id for _ in range(self.max_seq_len-len(tar_inp))]
                    tar_real = tar_real + [self.pad_tok_id for _ in range(self.max_seq_len-len(tar_real))]
                    #tar_inp = [self.pad_tok_id for _ in range(self.max_seq_len-len(tar_inp))] + tar_inp
                    #tar_real = [self.pad_tok_id for _ in range(self.max_seq_len-len(tar_real))] + tar_real

                nm_inp = nm_aux_tok + tar_inp

                #tar_inp = tf.cast(tf.convert_to_tensor(np.asarray(tar_inp)), dtype=tf.dtypes.int64)
                tar_real = tf.cast(tf.convert_to_tensor(np.asarray(tar_real)), dtype=tf.dtypes.int64)
                nm_inp = tf.cast(tf.convert_to_tensor(np.asarray(nm_inp)), dtype=tf.dtypes.int64)

                start_index += self.sliding_window
                end_index += self.sliding_window

                if counter == 0:
                    counter += 1
                    yield nm_inp, tar_real, tf.ones((1), dtype=tf.dtypes.int64)
                    # 1 indicates that we are at the first sentence in the article. (we should calculate the loss for all tokens)
                else:
                    yield nm_inp, tar_real, tf.zeros((1), dtype=tf.dtypes.int64)
                    # 0 indicates that we are not at the first sentence in the article. (we should calculate the loss only for the last sliding_window tokens)


    def call__(self):

        sliding_window_size = self.sliding_window
        nm_aux_tok_ = self.aux_toks
        shuffle = self.shuffle

        nm_aux_tok = []
        for word in nm_aux_tok_:
            w = self.tokenizer.encode_single(word)
            if len(w) == 1:
                w = w[0]
            else:
                raise Exception(
                    f"{word} should only have on id associated with it in the tokenizer. Something went wrong!")
            nm_aux_tok.append(w)

        keys = list(self.data_dict.keys())
        # print(keys)
        if shuffle:
            random.shuffle(keys)  # shuffles inplace.

        for i, key in enumerate(keys):

            # article_ids = self.tokenizer.encode_single(self.data_dict[key])
            article_ids = self.tokenizer.encode_single_id_string_max_seq_len(
                self.data_dict[key], max_seq_len=10000000)[0]  # [1:-2] # 0 is ids, 1 is strings. # no <s> or </s>
            # article_ids = article_ids["input_ids"]
            start_index = 0
            end_index = self.seq_len
            tar_inp = None
            nm_tar_inp = None
            tar_real = None
            article_len = len(article_ids)
            counter = 0
            while True:

                if start_index >= article_len - 1: break  # breaking condition for the article.
                # article_len -1 because want tar_real to be </s> at the very least if start_index is article_len-2
                if end_index < article_len - 1:
                    tar_inp = article_ids[start_index:end_index]
                    tar_real = article_ids[start_index + 1:end_index + 1]
                else:
                    tar_inp = article_ids[start_index:article_len - 1]  # -1 so there is an output token for the target.
                    tar_real = article_ids[start_index + 1:article_len]  # i.e. shifted to the right.

                if self.pad:
                    tar_inp = tar_inp + [self.pad_tok_id for _ in range(self.seq_len - len(tar_inp))]
                    tar_real = tar_real + [self.pad_tok_id for _ in range(self.seq_len - len(tar_real))]
                    # tar_inp = [self.pad_tok_id for _ in range(self.max_seq_len-len(tar_inp))] + tar_inp
                    # tar_real = [self.pad_tok_id for _ in range(self.max_seq_len-len(tar_real))] + tar_real

                nm_inp = nm_aux_tok + tar_inp

                # tar_inp = tf.cast(tf.convert_to_tensor(np.asarray(tar_inp)), dtype=tf.dtypes.int64)
                tar_real = tf.cast(tf.convert_to_tensor(np.asarray(tar_real)), dtype=tf.dtypes.int64)
                nm_inp = tf.cast(tf.convert_to_tensor(np.asarray(nm_inp)), dtype=tf.dtypes.int64)

                start_index += sliding_window_size
                end_index += sliding_window_size

                if counter == 0:
                    counter += 1
                    yield nm_inp, tar_real, tf.ones((1), dtype=tf.dtypes.int64)
                    # 1 indicates that we are at the first sentence in the article. (we should calculate the loss for all tokens)
                else:
                    yield nm_inp, tar_real, tf.zeros((1), dtype=tf.dtypes.int64)
                    # 0 indicates that we are not at the first sentence in the article. (we should calculate the loss only for the last sliding_window tokens)

if __name__ == "__main__":

    config = V4ConfigMediumSize(strategy=None, batch_size=2, loss_object=None, learning_rate=None, gpt2_117=True,
                                tokenizer="gpt2", vocab_filepath="/data/kkno604/Neuromodulated-Transformer/vocabulary/vocab1.txt")

    enc_tok = "<enc>"
    dec_tok = "<dec>"
    mlm_tok = "<mlm>"
    lm_tok = "<lm>"
    cls_tok = "<cls>"
    sep_tok = "<sep>"
    mask_tok = "<mask>"
    pad_tok = "<pad>"
    start_tok = "<s>"
    end_tok = "</s>"
    null_tok = "<null>"
    mqa = "<mqa>"
    pmqa = "<pmqa>"
    bmqa = "<bmqa>"
    peentailment = "<peentailment>"
    pbentailment = "<pbentailment>"
    pcoreference = "<pcoreference>"
    bcoreference = "<bcoreference>"
    psentiment = "<psentiment>"
    pgqa = "<pgqa>"
    psqa = "<psqa>"
    gqa = "<gqa>"
    pbqa = "<pbqa>"
    placeholder = "<placeholder>"
    translation = "<translation>"
    a1 = "(1)"
    a2 = "(2)"
    a3 = "(3)"
    a4 = "(4)"
    a5 = "(5)"
    a6 = "(6)"
    a7 = "(7)"
    a8 = "(8)"
    a9 = "(9)"
    passage = "<passage>"
    p1 = "<p1>"
    p2 = "<p2>"
    p3 = "<p3>"
    p4 = "<p4>"
    p5 = "<p5>"
    p6 = "<p6>"
    p7 = "<p7>"
    p8 = "<p8>"
    p9 = "<p9>"
    hypothesis = "<h>"
    question = "<q>"
    metacognition = "<mc>"
    unk_rs = "<unk_rs>"
    aoint_rs = "<aoint_rs>"
    highlighting_rs = "<highlighting_rs>"
    reread_rs = "<reread_rs>"
    summarize_rs = "<summarize_rs>"
    paraphrase_rs = "<paraphrase_rs>"
    num_reading_strategies = 6
    pad_to_max_length = True
    strategy = "random"
    C4_processed_filepath = ""
    num_aux_toks = 3

    aux_toks = ["<cls>", "<dec>", "<lm>"]

    dloader = WikiTextDataLoader(strategy="gpt2-remove",
                             filepath="/large_data/wikitext-103/wiki.test.tokens",
                             enc_tok=enc_tok, dec_tok=dec_tok,
                             mlm_tok=mlm_tok, lm_tok=lm_tok,
                             start_tok=start_tok, end_tok=end_tok,
                             cls_tok=cls_tok,
                             sep_tok=sep_tok, mask_tok=mask_tok,
                             pad_tok=pad_tok, seq_len=768,
                             pad=True,
                             a1=a1, a2=a2, a3=a3, a4=a4,
                             a5=a5, a6=a6, a7=a7, a8=a8,
                             a9=a9,
                             passage=passage, p1=p1, p2=p2,
                             p3=p3, p4=p4, p5=p5, p6=p6,
                             p7=p7, p8=p8, p9=p9, mqa=mqa,
                             pmqa=pmqa, bmqa=bmqa,
                             peentailment=peentailment,
                             pbentailment=pbentailment,
                             pcoreference=pcoreference,
                             bcoreference=bcoreference,
                             psentiment=psentiment,
                             pgqa=pgqa, psqa=psqa, gqa=gqa,
                             pbqa=pbqa,
                             placeholder=placeholder,
                             translation=translation,
                             hypothesis=hypothesis, question=question,
                             metacognition=metacognition,
                             unk_rs=unk_rs,
                             aoint_rs=aoint_rs,
                             highlighting_rs=highlighting_rs,
                             reread_rs=reread_rs,
                             summarize_rs=summarize_rs,
                             paraphrase_rs=paraphrase_rs,
                             #tokenizer=None)
                             tokenizer=config.tokenizer,
                             sliding_window=config.max_seq_len_dec,
                             aux_toks=aux_toks,
                             shuffle=False)



    generator = tf.data.Dataset.from_generator(dloader.call__,
                                               output_types=(tf.dtypes.int64,
                                                             tf.dtypes.int64,
                                                             tf.dtypes.int64)).batch(config.batch_size)
    #mini_gen = dloader("test", False)
    for i, j, k in generator:
        print(i, j, k)
        break







