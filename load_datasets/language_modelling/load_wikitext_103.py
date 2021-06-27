'''
Author: Kobe Knowles
'''

import tensorflow as tf
import numpy as np
import os
from pathlib import Path
import re
import copy
import random
import math
import json
from transformers import XLNetTokenizer

#from torch.utils.data.dataset import Dataset
#from torch.utils.data import DataLoader
#import matplotlib.pyplot as plt
#import pandas as pd

#import numpy as np
#np.set_printoptions(precision=4)

import sys
sys.path.append("..") # only helps when importing from another folder...

#from text_processing.tokenization import *
from text_processing.create_tfxl_vocab import *


class load_Wikitext_tf:

    __data = []

    def __init__(self, filepath, max_seq_len, tokenizer=None, pad_to_max_length=True, strategy="greedy",
                 start_token="<s>", end_token="</s>", pad_token="<pad>", new_line_token="<nline>", load_data=[False, ""]):

        self.filepath = filepath
        self.max_seq_len = max_seq_len # what to pad to.
        self.tokenizer = tokenizer

        self.strategy = strategy
        assert self.strategy in ["greedy","document_oriented", "document_oriented_no_new_line",
                                 "include_headings"], f"Invalid strategy: {self.strategy}."

        self.heading_pattern = ' [= ]+[^=]*[^=] [= ]+\n'

        self.start_token = start_token
        stok = tokenizer.encode(self.start_token,
                                        add_special_tokens=False,
                                        pad_to_max_length=False,
                                        return_token_type_ids=False,
                                        return_attention_mask=False)
        print(f"The start token id is {stok}")

        self.end_token = end_token
        etok = tokenizer.encode(self.end_token,
                                add_special_tokens=False,
                                pad_to_max_length=False,
                                return_token_type_ids=False,
                                return_attention_mask=False)
        print(f"The end token id is {etok}")

        self.pad_token = pad_token
        ptok = tokenizer.encode(self.pad_token,
                                add_special_tokens=False,
                                pad_to_max_length=False,
                                return_token_type_ids=False,
                                return_attention_mask=False)
        print(f"The pad token id is {ptok}")

        self.new_line_token = new_line_token
        ntok = tokenizer.encode(self.new_line_token,
                                add_special_tokens=False,
                                pad_to_max_length=False,
                                return_token_type_ids=False,
                                return_attention_mask=False)
        print(f"The new line token id is {ntok}")

        if not load_data[0]:
            self.data = []
            self._load() # self.data becomes ["row 1", "row 2", ...]

            if tokenizer is None:
                raise Exception("A tokenizer is required as input.")
            else:
                self._tokenize()
                if pad_to_max_length:
                    self._pad()
                else:
                    print("Note: That this function is currently untested and may not work! (def _split()")
                    self._split() # this splits sequences larger than the max_seq_len into multiple samples.
            self.__data = self.data
        else:
            self._read_json(load_data[1])

    def get_data_nm(self, eos_id=None, pad_id=None, *nm_auxiliary_tokens):

        print(f"Note the requirement here that there is only one eos token in each row. \n"
              f"If not then there may be a ramdom pad token in the middle of the row. \n")

        if eos_id is None or pad_id is None:
            raise Exception(f"One of eos_id: {eos_id}, pad_id: {pad_id} is None and shouldn't be.")

        aux_tok_id = [id_ for id_ in nm_auxiliary_tokens]
        tar_inp = []
        tar_real = []
        nm_inp = []
        for i, row in enumerate(self.__data):
            # replace eos id with pad, as this is never passed as input. -- this was old and decided to remove.
            # inp_ = [pad_id if x == eos_id else x for x in row]
            inp_ = row
            tar_inp.append(inp_)

            # add auxililary tokens to the input.
            nm_inp.append(aux_tok_id + inp_) # this is just the input with aux_token ids appended at the beginning.

            # as the target this essentially shifts right by one and appends a pad token to the end.
            # this is just a shift left by 1, then append the pad token at the end.
            tar_real.append(row[1:]+[pad_id]) # adding pad_id keeps the sequence length matching the inp_ dimension.

        tar_inp = tf.cast(tf.convert_to_tensor(np.asarray(tar_inp)), dtype=tf.dtypes.int64)
        tar_real = tf.cast(tf.convert_to_tensor(np.asarray(tar_real)), dtype=tf.dtypes.int64)
        nm_inp = tf.cast(tf.convert_to_tensor(np.asarray(nm_inp)), dtype=tf.dtypes.int64)
        return tar_inp, tar_real, nm_inp

    def __len__(self):
        return len(self.__data)

    def _load(self):
        data = None
        with open(self.filepath, "r") as f:
            data = f.readlines()

        if self.strategy == "greedy":
            for i in range(len(data)):
                if re.match(self.heading_pattern, data[i]):
                    continue
                elif data[i] != " \n":
                    t = re.sub("\n$", str(self.end_token), data[i])
                    self.data.append(str(self.start_token) + " " + t)
        elif self.strategy == "document_oriented":
            prev_head = False
            document_ = ""
            len_nline_tok = len(self.new_line_token)
            for i in range(len(data)):
                if re.match(self.heading_pattern, data[i]):
                    prev_head = True
                    if document_ != "": # i.e. we are entering a new batch item (or sample).
                        self.data.append(document_[:-len_nline_tok] + " " + str(self.end_token)) # [:-len_nline_tok] this removes the <nline> token and replaces it with the end token.
                    t = re.sub("\n$", str(self.new_line_token), data[i])  # replace \n with <nline> token.
                    document_ = "" # no matter what we reset this here.
                elif data[i] != " \n":
                    if prev_head:
                        data[i] = str(self.start_token)+ " " + data[i]  # represents the start of an article or document...
                        prev_head = False
                    t = re.sub("\n$", str(self.new_line_token), data[i]) # replace \n with <nline> token.
                    document_ += (" " + t) # Extra space at start of sequence is removed with tokenizer (or can do manually with .strip())
            # last append to data.
            if document_ != "":
                self.data.append(document_[:-len_nline_tok] + " " + str(self.end_token)) # necessary to perform one at the end.
        elif self.strategy == "document_oriented_no_new_line":
            prev_head = False
            document_ = ""
            len_nline_tok = len(self.new_line_token)
            for i in range(len(data)):
                if re.match(self.heading_pattern, data[i]):
                    prev_head = True
                    if document_ != "":
                        self.data.append(document_[:-len_nline_tok] + " " + str(self.end_token))
                    document_ = ""
                elif data[i] != " \n":
                    if prev_head:
                        data[i] = str(self.start_token)+ " " + data[i]  # represents the start of an article or document...
                        prev_head = False
                    t = re.sub("\n$", str(self.end_token), data[i])
                    document_ += (" " + t)
            # last append to data.
            if document_ != "":
                self.data.append(document_[:-len_nline_tok] + " " + str(self.end_token))
        elif self.strategy == "include_headings":
            for i in range(len(data)):
                if data[i] == " \n":
                    continue # skip this line as it is irrelevant.
                t = re.sub("\n$", str(self.end_token), data[i])

                self.data.append(t)
        else:
            raise Exception(f"Invalid strategy! {self.strategy}")


    def _tokenize(self):
        self.data = self.tokenizer.batch_encode_plus(self.data,
                                                       add_special_tokens=False,
                                                       pad_to_max_length=False,
                                                       return_token_type_ids=False,
                                                       return_attention_mask=False)["input_ids"]

    def _pad(self):
        # get pad token.
        pad_token_id = tokenizer.encode(self.pad_token,
                                        add_special_tokens=False,
                                        pad_to_max_length=False,
                                        return_token_type_ids=False,
                                        return_attention_mask=False)
        if len(pad_token_id) > 1:
            raise Exception("pad_token is not part of the tokenizer's vocabulary!")
        elif len(pad_token_id) == 1:
            pad_token_id = pad_token_id[0]
        else:
            raise Exception("Something really went wrong if here is reached!")


        # no matter what the strategy, it is already done previously.
        if self.strategy != "include_headings":
            temp_data = []
            for row in self.data:
                if len(row) <= self.max_seq_len:
                    temp_data.append(row + [pad_token_id for _ in range(self.max_seq_len-len(row))])
                else: # len(row) > self.max_seq_len
                    temp_data.append(row[:self.max_seq_len])
                    # row[self.max_seq_len-1] as the last element is to be used as context in the next sample.
                    # The recursion below is spliting the samples based on the maximum sequence length and creating new examples by splitting existing.
                    temp_data = self._recursion_pad(temp_data, row[self.max_seq_len:], row[self.max_seq_len-1], pad_token_id)
                    #temp_data.append(row)
            self.data = temp_data
        else:
            combine = []
            # efficiency isn't really important here, once this is done once it can just be saved to a file and read again.
            for row in self.data:
                combine += row # essentially flattens the input.
            temp_list = []
            min_counter = 0
            assert len(combine) >= self.max_seq_len, "Something went horribly wrong..."
            max_counter = self.max_seq_len
            while True:
                assert max_counter-min_counter == self.max_seq_len, "The difference between the min counter and max counter is not equal to the max_seq_len"
                if max_counter > len(combine):
                    comb = combine[min_counter:len(combine)]
                    temp_list.append(comb+[pad_token_id for _ in range(self.max_seq_len-len(comb))])
                    break

                temp_list.append(combine[min_counter:max_counter])
                # Why should I minus 1 here?
                min_counter = max_counter
                max_counter += self.max_seq_len
            self.data = temp_list

    def _recursion_pad(self, temp_data, row, prev_token, pad_token_id):
        # prev_token will be an integer
        # prev_token is used for additional context and won't be counted twice in training as targets start at index 1, not 0.
        if len(row) <= self.max_seq_len-1: # -1 becuase of additional prev_token.
            temp_data.append([prev_token] + row + [pad_token_id for _ in range(self.max_seq_len-1 - len(row))])
        else:
            temp_data.append([prev_token] + row[:self.max_seq_len-1])
            temp_data = self._recursion_pad(temp_data, row[self.max_seq_len-1:], row[self.max_seq_len-2], pad_token_id) # -2 here is correct.
        return temp_data

    def _split(self):
        # basically _pad but with no padding.
        temp_data = []
        for row in self.data:
            if len(row) <= self.max_seq_len:
                temp_data.append(row)
            else:  # len(row) > self.max_seq_len
                temp_data.append(row[:self.max_seq_len])
                # row[self.max_seq_len-1] as the last element is to be used as context in the next sample.
                # The recursion below is spliting the samples based on the maximum sequence length and creating new examples by splitting existing.
                temp_data = self._recursion_split(temp_data, row[self.max_seq_len:], row[self.max_seq_len - 1])
                # temp_data.append(row)
        self.data = temp_data

    def _recursion_split(self, temp_data, row, prev_token, pad_token_id):
        # prev_token will be an integer
        # prev_token is used for additional context and won't be counted twice in training as targets start at index 1, not 0.
        if len(row) <= self.max_seq_len-1: # -1 becuase of additional prev_token.
            temp_data.append([prev_token] + row)
        else:
            temp_data.append([prev_token] + row[:self.max_seq_len-1])
            temp_data = self._recursion_pad(temp_data, row[self.max_seq_len-1:], row[self.max_seq_len-2]) # -2 here is correct.
        return temp_data

    def save_json(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.__data, f)
            print("Save successful at " + str(filepath))

    def _read_json(self, filepath):
        with open(filepath, "r") as f:
            self.__data = json.load(f)
            self.data = self.__data
            print("Load successful at " + str(filepath))

if __name__ == "__main__":

    max_seq_len = 512
    tokenizer = get_tfxl_tokenizer()

    folderpath = "/large_data/wikitext-103"


    wikitext_val_data = load_Wikitext_tf(folderpath + "/wiki.valid.tokens", max_seq_len=max_seq_len, tokenizer=tokenizer,
                                      pad_to_max_length=True, strategy="include_headings", start_token="<s>", end_token="</s>",
                                      pad_token="<pad>", new_line_token="<nline>", load_data=[False, ""])
    wikitext_val_data.save_json("/large_data/wikitext-103/val_include_headings_mlen512_tfxltok.txt")

    wikitext_test_data = load_Wikitext_tf(folderpath + "/wiki.test.tokens", max_seq_len=max_seq_len, tokenizer=tokenizer,
                                      pad_to_max_length=True, strategy="include_headings", start_token="<s>", end_token="</s>",
                                      pad_token="<pad>", new_line_token="<nline>", load_data=[False, ""])
    wikitext_test_data.save_json("/large_data/wikitext-103/test_include_headings_mlen512_tfxltok.txt")

    wikitext_train_data = load_Wikitext_tf(folderpath+"/wiki.train.tokens", max_seq_len=max_seq_len, tokenizer=tokenizer,
                                  pad_to_max_length=True, strategy="include_headings", start_token="<s>", end_token="</s>",
                                  pad_token="<pad>", new_line_token="<nline>", load_data=[False, ""])
    wikitext_train_data.save_json("/large_data/wikitext-103/train_include_headings_mlen512_tfxltok.txt")







