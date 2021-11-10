'''
File name: load_C4.py
Author: Kobe Knowles
Date created: 20/08/21
Data last modified: 9/11/21
Python Version: 3.8
Tensorflow version: 2.5
'''

import os
import random

import gzip
import json
import re

import sys

import regex

sys.path.append("../..")

from transformers import TransfoXLTokenizer
from text_processing.tokenizer import Tokenizer

class C4DataLoader:
    '''
    '''
    def __init__(self, strategy: str, filepath: str, enc_tok: str, dec_tok: str, mlm_tok: str,
                 lm_tok: str, start_tok: str, end_tok: str, cls_tok: str, sep_tok: str, mask_tok: str, pad_tok: str,
                 seq_len: int, pad: bool, tokenizer=None, processed_filepath=""):
        '''
        Function: __init__ \n
        Description: Initialize the parameters of the C4DataLoader class. \n
        Input:
            strategy: (str) The strategy to use to load the data. \n
            filapath: (str) Path to en folder containing all .gz files. \n
            enc_tok: (str) Token that represents the encoder. \n
            dec_tok: (str) Token that represents the decoder. \n
            mlm_tok: (str) Auxiliary token representation for the (dynamic) masked language modelling task. \n
            lm_tok: (str) Auxiliary token representation for the language modelling task. \n
            start_tok: (str) The start token to append at the beginning of a document. \n
            end_tok: (str) The end token to append to the end of a document. \n
            sep_tok: (str) The seperator token to be used by the encoder in masked language modelling. \n
            mask_tok: (str) The token to mask masked out tokens in masked language modelling. \n
            pad_tok: (str) The token that represents the pad token. \n
            seq_len: (int) Number of tokens in an individual sample. \n
            pad: (bool) True if we pad to the seq_len if required; False otherwise.
            tokenizer: (Tokenizer) The tokenizer to be used by the data loader. \n
            processed_filepaths: (str) String that points to a text file that has stored all files already processed fully.
        '''

        self.data_counter = 0 # counter for how much we are through the data.
        self.strategy = strategy

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

        self.tokenizer = tokenizer

        if tokenizer is not None:
            ###### <s> ######
            self.start_tok_id = self.tokenizer.encode_single(self.start_tok)
            if len(self.start_tok_id) != 1 and (isinstance(self.start_tok_id, list)):
                raise Exception(f"The number of ids the start token is encoded into should be one, got {self.start_tok_id}!")
            else:
                self.start_tok_id = self.start_tok_id[0]
            print(f"The start token id is: {self.start_tok_id}")

            ###### </s> ######
            self.end_tok_id = self.tokenizer.encode_single(self.end_tok)
            if len(self.end_tok_id) != 1 and (isinstance(self.end_tok_id, list)):
                raise Exception(f"The number of ids the end token is encoded into should be one, got {len(self.end_tok_id)}!")
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

        self.pad = pad # TODO: support not shown below...
        self.seq_len = seq_len

        self.filepath = filepath # "test/.../"
        self.filepaths_list_fixed = os.listdir(self.filepath)
        #print(len(self.filepaths_list_fixed))
        if processed_filepath == "":
            if strategy == "all": pass # do nothing.
            elif strategy == "train":
                reg = re.compile(r".*validation.*")
                self.filepaths_list_fixed = [path for path in self.filepaths_list_fixed if not reg.match(path)]
            elif strategy == "val" or strategy == "validation":
                reg = re.compile(r".*train.*")
                self.filepaths_list_fixed = [path for path in self.filepaths_list_fixed if not reg.match(path)]
        else:
            remove = self._open_file_helper(processed_filepath) # already processed.
            self.filepaths_list_fixed = [item for item in self.filepaths_list_fixed if not (item in remove)]

        self.filepaths_list = self.filepaths_list_fixed
        #print(len(self.filepaths_list_fixed))
        #print(f"\nfilepaths_list {self.filepaths_list}\n")

        self.current_file = self.filepaths_list.pop()
        self._load_gz_file(self.filepath+self.current_file)
        #print(len(self.data))
        self.load_data()

    def _save_file_helper(self, file):
        with open(self.filepath+"../processed.txt", "a") as f:
            f.write(f"{file} \n")

    def _open_file_helper(self, file):
        file_to_remove = None
        with open(self.filepath+"../processed.txt", "r") as f:
            file_to_remove = f.readlines()
        file_to_remove = [item.strip(" \n") for item in file_to_remove]
        return file_to_remove

    def load_data(self):

        if self.data_counter >= len(self.data):
            self._save_file_helper(self.current_file)
            if len(self.filepaths_list) == 0:
                self.data = None
                return None # stops any more processing.
            self.current_file = self.filepaths_list.pop()
            self._load_gz_file(self.filepath+self.current_file) # updates self.data...
            self.data_counter = 0

        #self.data_id_string = self.tokenizer.encode_single_id_string(self.data[self.data_counter]["text"]) # ([id, ...], [string, ...])
        self.data_id_string = self.tokenizer.encode_single_id_string_max_seq_len(self.data[self.data_counter]["text"], max_seq_len=10000000)  # ([id, ...], [string, ...])
        #print(f"\n\n\nlength of the data id string: {len(self.data_id_string[0])}\n\n\n")
        self.start_index = 0
        self.end_index = self.seq_len
        self.data_counter += 1

    def _load_gz_file(self, filepath):

        print(f"\nLoading data at {filepath}!\n")
        self.data = []
        data = None
        with gzip.open(filepath, 'rb') as f:
            data = f.read().decode('utf-8').split("\n")
            #data = f.read()
        if data is not None:
            for item in data:
                try:
                    loaded = json.loads(item)
                    self.data.append(loaded)
                except:
                    continue # handles any errors in loading by skipping to the next item.


    def _get_decoder_lm(self):

        input_string = None
        input_id = None
        target_input_id = None

        text_len = len(self.data_id_string[0])  # 0 or 1 doesn't matter.

        if self.start_index >= text_len-1:
            self.load_data()
            if self.data is None: return None, None, None

        text_len = len(self.data_id_string[0])  # 0 or 1 doesn't matter.

        if self.end_index < text_len-1:
            input_string = self.data_id_string[1][self.start_index:self.end_index]
            input_id = self.data_id_string[0][self.start_index:self.end_index]
            target_input_id = self.data_id_string[0][self.start_index+1:self.end_index+1]
        else:
            input_string = self.data_id_string[1][self.start_index:text_len-1]
            input_id = self.data_id_string[0][self.start_index:text_len-1]
            target_input_id = self.data_id_string[0][self.start_index+1:text_len] # i.e. shifted to the right.

        #self.start_index = self.end_index
        self.start_index += self.seq_len
        self.end_index += self.seq_len
        assert len(target_input_id) == len(input_id), f" text_len: {text_len} \ntarget_len: {len(target_input_id)} " \
                                                      f"\ninput_id_len: {len(input_id)} \n start_index: {self.start_index}" \
                                                      f"\nend_index: {self.end_index}"
        return input_string, input_id, target_input_id

    def _get_encoder_mlm(self):
        input_string = None
        input_id = None
        target_input_id = None
        text_len = len(self.data_id_string[0])  # 0 or 1 doesn't matter.

        if self.start_index >= text_len:
            self.load_data()
            if self.data is None: return None, None, None

        # article_len -1 because want tar_real to be </s> at the very least if start_index is article_len-2
        if self.end_index < text_len:
            input_string = self.data_id_string[1][self.start_index:self.end_index]
            input_id = self.data_id_string[0][self.start_index:self.end_index]
            #target_input_id = self.data_id_string[0][self.start_index:self.end_index]
        else:
            input_string = self.data_id_string[1][self.start_index:text_len]
            input_id = self.data_id_string[0][self.start_index:text_len]
            #target_input_id = self.data_id_string[0][self.start_index:text_len]  # i.e. shifted to the right.

        target_input_id = [self.pad_tok_id for _ in range(len(input_id))]

        mask_prob = [random.random() for _ in range(len(input_id))]
        for i, prob in enumerate(mask_prob):
            if prob <= 0.15: # i.e. mask 15% of tokens:
                prob2 = random.random()
                if prob2 <= 0.8:
                    #print(f"input_id pos {i} (mask replacement): {input_id[i]}")
                    target_input_id[i] = input_id[i]
                    input_id[i] = self.mask_tok_id
                elif prob2 > 0.8 and prob2 <= 0.9: # here replace with a random token.
                    #print(f"replaced with a random token at position {i}")
                    input_id[i] = random.randint(0,self.tokenizer.get_vocab_size()-1)
                #else: # leave unchanged, just commented out as nothing is needed to be done here.

        self.start_index = self.end_index
        self.end_index += self.seq_len

        return input_string, input_id, target_input_id

    def __call__(self, mode: str):
        if mode == "enc_dec":
            a = random.random()  # 0 or 1
            if a <= 0.87:  # corresponds to the encoder 87% of the time because 15% are maked and the output, need to weight with the decoder for balanced labels...
                input_string, input_id, target_input_id = self._get_encoder_mlm()
                return input_string, input_id, target_input_id, self.enc_tok_id, self.mlm_tok_id
            else:  # corresponds to the decoder
                input_string, input_id, target_input_id = self._get_decoder_lm()
                return input_string, input_id, target_input_id, self.dec_tok_id, self.lm_tok_id
        elif mode == "enc":
            input_string, input_id, target_input_id = self._get_encoder_mlm()
            return input_string, input_id, target_input_id, self.enc_tok_id, self.mlm_tok_id
        elif mode == "dec":
            input_string, input_id, target_input_id = self._get_decoder_lm()
            return input_string, input_id, target_input_id, self.dec_tok_id, self.lm_tok_id

if __name__ == "__main__":
    tok = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
    tokenizer = Tokenizer(tok)
    tokenizer.add_tokens_list(["<enc>", "<dec>", "<mlm>", "<lm>", "<s>", "</s>", "<cls>", "<sep>", "<mask>", "<pad>"])
    c4 = C4DataLoader(strategy="train", filepath="/large_data/C4/en/", enc_tok="<enc>", dec_tok="<dec>",
                      mlm_tok="<mlm>", lm_tok="<lm>", start_tok="<s>", end_tok="</s>", cls_tok="<cls>",
                      sep_tok="<sep>", mask_tok="<mask>", pad_tok="<pad>", seq_len=16, pad=False, tokenizer=tokenizer,
                      processed_filepath="")
                      #processed_filepath="/large_data/C4/en/../processed.txt")
    for i in range(5):
        input_string, input_id, target_input_id, aux_pos_1, aux_pos_2 = c4()
        print(f"input_string \n {input_string} \n"
              f"input_id \n {input_id} \n"
              f"target_input_id \n {target_input_id} \n"
              f"aux_pos_1 \n {aux_pos_1} \n"
              f"aux_pos_2 \n {aux_pos_2} \n")