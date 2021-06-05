'''
Author: Kobe Knowles
'''

#import tensorflow as tf
import os
from pathlib import Path
import re
import copy
import random
import math
import json
from transformers import XLNetTokenizer
#import matplotlib.pyplot as plt
#import pandas as pd

#import numpy as np
#np.set_printoptions(precision=4)

import sys
sys.path.append("..") # only helps when importing from another folder...

from text_processing.tokenization import *


#data_files = tf.data.Dataset.list_files(str(path_to_wikitext_103_files/'*/*'))

class Wikitext_103_Loader(object):
    def __init__(self, batch_size, batch_strategy, tokenizer,
                 folderpath="../datasets/wikitext-103-v1/wikitext-103",
                 max_seq_len=32, load_tokenized_tokens=[False, 'path_to_folder/',
                                                        'train.txt', 'val.txt', 'test.txt'],
                 start_token="<s>", end_token="</s>", pad_token="<pad>"):

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token

        '''
        batch_strategy:
            "greedy" refers to a simple batching strategy that combines inputs where possible.
                i.e. combine muliple if no [END].
            "2D_batching": i.e document level.
        '''
        self.batch_strategy = batch_strategy

        self.folderpath = folderpath

        self.heading_pattern = ' [= ]+[^=]*[^=] [= ]+\n'

        self.train_examples = []
        self.val_examples = []
        self.test_examples = []

        if not load_tokenized_tokens[0]:
            self._load_train()
            self._load_validation()
            self._load_test()

            self._tokenize_all("train")
            self._tokenize_all("val")
            self._tokenize_all("test")
        else:
            z = load_tokenized_tokens
            self._read_json(z[1], z[2], z[3], z[4])

    def _load_train(self):
        data = None
        with open(self.folderpath + "/wiki.train.tokens", "r") as f:
            data = f.readlines()

        for i in range(len(data)):
            if re.match(self.heading_pattern, data[i]):  # re.match returns True or None.
                # uncomment below to store headings.
                #train_headings.append(train_data[i].strip("\n").strip())
                #print("Heading:", data[i])
                pass
            elif data[i] != " \n":
                data[i] = data[i].lower()
                data[i] = "<s> "+ data[i] # represents the start of an article or document...
                t = re.sub("\n$", "</s>", data[i])
                #print(t)
                self.train_examples.append(t.strip())


    def _load_validation(self):
        data = None
        with open(self.folderpath + "/wiki.valid.tokens", "r") as f:
            data = f.readlines()
        #print(data[0:100])
        for i in range(len(data)):
            if re.match(self.heading_pattern, data[i]):  # re.match returns True or None.
                # uncomment below to store headings.
                # train_headings.append(train_data[i].strip("\n").strip())
                #print("Heading:", data[i])
                pass
            elif data[i] != " \n":
                data[i] = data[i].lower()
                data[i] = "<s> " + data[i] # represents the start of an article or document...
                t = re.sub("\n$", "</s>", data[i])
                self.val_examples.append(t.strip())

    def _load_test(self):
        data = None
        with open(self.folderpath + "/wiki.test.tokens", "r") as f:
            data = f.readlines()

        for i in range(len(data)):
            if re.match(self.heading_pattern, data[i]):  # re.match returns True or None.
                # uncomment below to store headings.
                # train_headings.append(train_data[i].strip("\n").strip())
                # print("Heading:", data[i])
                pass
            elif data[i] != " \n":
                data[i] = data[i].lower()
                data[i] = "<s> " + data[i].strip()  # represents the start of an article or document...
                t = re.sub("\n$", "</s>", data[i])
                self.test_examples.append(t.strip())

    def _tokenize_all(self, type_):

        if type_ == "train":
            self.train_examples = self.tokenizer.batch_encode_plus(self.train_examples,
                                                                      add_special_tokens=False,
                                                                      pad_to_max_length=False,
                                                                      return_token_type_ids=False,
                                                                      return_attention_mask=False)["input_ids"]
            # (batch_size(here this is the len of the dataset, seq_len (varies))
            # [[90, 79, 132, 79, 94, 9, 56, 851, 13, 87, 79, 309, 76, 6740, 376, 39, 88, 4091, 15],
            # [71, 3640, 564, 86, 71, 336, 755, 77, 71, 50, 3719, 4, 0, 0, 0, 0, 0, 0, 0]]
            if self.batch_strategy == "greedy":
                self._greedy_batch("train")
            elif self.batch_strategy == "2D_batching":
                self._2D_batch("train")

        elif type_ == "val":
            self.val_examples = self.tokenizer.batch_encode_plus(self.val_examples,
                                                                   add_special_tokens=False,
                                                                   pad_to_max_length=False,
                                                                   return_token_type_ids=False,
                                                                   return_attention_mask=False)["input_ids"]
            if self.batch_strategy == "greedy":
                self._greedy_batch("val")
            elif self.batch_strategy == "2D_batching":
                self._2D_batch("val")

        elif type_ == "test":
            self.test_examples = self.tokenizer.batch_encode_plus(self.test_examples,
                                                                   add_special_tokens=False,
                                                                   pad_to_max_length=False,
                                                                   return_token_type_ids=False,
                                                                   return_attention_mask=False)["input_ids"]
            if self.batch_strategy == "greedy":
                self._greedy_batch("test")
            elif self.batch_strategy == "2D_batching":
                self._2D_batch("test")

    def _2D_batch(self, type_):
        examples_copy = None
        if type_ == "train":
            examples_copy = self.train_examples  # copy.deepcopy removed as it isn't needed.
            examples = []  # Q: does this change the self.train examples if examples is edited?
        elif type_ == "val":
            examples_copy = self.val_examples
            examples = []
        elif type_ == "test":
            examples_copy = self.test_examples
            examples = []
        else:
            raise Exception("Invalid type_ parameter!")

        end_token_id = tokenizer.encode(self.end_token,
                                        add_special_tokens=False,
                                        pad_to_max_length=False,
                                        return_token_type_ids=False,
                                        return_attention_mask=False)[0]
        pad_token_id = tokenizer.encode(self.pad_token,
                                        add_special_tokens=False,
                                        pad_to_max_length=False,
                                        return_token_type_ids=False,
                                        return_attention_mask=False)[0]
        N = self._getN(examples_copy)
        N = math.ceil(N % self.max_seq_len)
        print("N", N)

        for row in examples_copy:
            temp_list = []
            if len(row) <= self.max_seq_len:
                temp_list.append(row + [pad_token_id for _ in range(self.max_seq_len-len(row))])
                n = 1
                # pad the next N-1 segments.
                if N > 1:
                    while True:
                        temp_list.append([pad_token_id for _ in range(self.max_seq_len)])
                        n += 1
                        if n == N: break
            elif len(row) > self.max_seq_len:
                temp_list.append(row[:self.max_seq_len])
                temp_list = self._2D_recursion(temp_list, row[self.max_seq_len:], pad_token_id, N-1)

            examples.append(temp_list) # temp_list (max_seq_len, N) after appending.

        if type_ == "train":
            self.train_examples = examples
        elif type_ == "val":
            self.val_examples = examples
        elif type_ == "test":
            self.test_examples = examples

    def _2D_recursion(self, temp_list, row, pad_token_id, N):
        temp = temp_list
        if len(row) <= self.max_seq_len:
            temp.append(row + [pad_token_id for _ in range(self.max_seq_len-len(row))])
            n = 1
            if N > 1:
                while True:
                    temp.append([pad_token_id for _ in range(self.max_seq_len)])
                    n += 1
                    if n == N: break
        elif len(row) > self.max_seq_len:
            temp.append(row[:self.max_seq_len])
            temp = self._2D_recursion(temp, row[self.max_seq_len:], pad_token_id, N-1)
        return temp

    def _getN(self, examples):
        # examples size == (nrow|batch, seq_len)
        N = 0
        for row in examples:
            if len(row) > N:
                N = len(row)
        return N

    def _greedy_batch(self, type_):

        examples_copy = None
        if type_ == "train":
            examples_copy = self.train_examples # copy.deepcopy removed as it isn't needed.
            examples = [] # Q: does this change the self.train examples if examples is edited?
        elif type_ == "val":
            examples_copy = self.val_examples
            examples = []
        elif type_ == "test":
            examples_copy = self.test_examples
            examples = []
        else:
            raise Exception("Invalid type_ parameter!")

        counter = 0

        end_token_id = tokenizer.encode(self.end_token,
                         add_special_tokens=False,
                         pad_to_max_length=False,
                         return_token_type_ids=False,
                         return_attention_mask=False)[0]
        pad_token_id = tokenizer.encode(self.pad_token,
                         add_special_tokens=False,
                         pad_to_max_length=False,
                         return_token_type_ids=False,
                         return_attention_mask=False)[0]
        # note: padding to maximum sequence length is done below.
        while True:
            if examples_copy[counter][-1] == end_token_id and len(
                    examples_copy[counter]) <= self.max_seq_len:
                examples.append(examples_copy[counter]+[pad_token_id for _ in range(self.max_seq_len-len(examples_copy[counter]))])
            elif examples_copy[counter][-1] == end_token_id and len(
                    examples_copy[counter]) > self.max_seq_len:
                examples.append(examples_copy[counter][:self.max_seq_len])
                #examples, counter = self._greedy_recursion(examples, counter)-1 # minus one here as an additional increment is done each while loop iteration at the end.
                examples = self._greedy_recursion(examples, examples_copy[counter][self.max_seq_len:], pad_token_id)
                #examples.append(examples_copy[counter][self.max_seq_len:]+[pad_token_id for _ in range(self.max_seq_len*2-len(examples_copy[counter]))])
            elif len(examples_copy[counter]) > self.max_seq_len:
                print("Shouldn't reach here ")
                #print(examples_copy[counter])
                examples.append(examples_copy[counter][:self.max_seq_len])
                examples = self._greedy_recursion(examples, examples_copy[counter][self.max_seq_len:], pad_token_id)
                #examples.append(examples_copy[counter][self.max_seq_len:]+[pad_token_id for _ in range(self.max_seq_len*2-len(examples_copy[counter]))])
            elif len(examples_copy[counter]) <= self.max_seq_len:
                # In theory in wikitext-103 we will never reach here?
                print("In wikitext-103 this should never be reached?")
                if counter < len(examples_copy)-1:
                    # greedy as in it only combines the following into one. Greedy should work for wikitext-103
                    # becasue of the nature of the dataset.
                    if len(examples_copy[counter]) + len(examples_copy[counter+1]) <= self.max_seq_len:
                        com_ex = examples_copy[counter] + examples_copy[counter+1] + [pad_token_id for _ in range(self.max_seq_len-(len(examples_copy[counter])+
                                                                                                                                    len(examples_copy[counter+1])))]
                        # print(com_ex)
                        examples.append(com_ex)
                        # additional add to counter here as we skip the following element.
                        counter += 1
                else:
                    examples.append(examples_copy[counter]+[pad_token_id for _ in range(self.max_seq_len-len(examples_copy[counter]))])
            counter += 1
            # breaking condition.
            if counter >= len(examples_copy): break

        if type_ == "train":
            self.train_examples = examples
        elif type_ == "val":
            self.val_examples = examples
        elif type_ == "test":
            self.test_examples = examples

    # handles case where current sample length is greater than max_seq_len and handles recursively
    # for the current example only.
    def _greedy_recursion(self, examples, examples_copy, pad_token_id):
        # counter for examples_copy isn't needed any more here.
        if len(examples_copy) <= self.max_seq_len:
            examples.append(examples_copy+[pad_token_id for _ in range(self.max_seq_len-len(examples_copy))])
        else: # > max_seq_len
            examples = self._greedy_recursion(examples, examples_copy[self.max_seq_len:], pad_token_id)
        return examples

    def save_json(self, filepath, train_file_name="train.txt", val_file_name="val.txt", test_file_name="test.txt"):
        with open(filepath+train_file_name, "w") as f:
            json.dump(self.train_examples, f)
            print("save train")

        with open(filepath+val_file_name, "w") as f:
            json.dump(self.val_examples, f)
            print("save val")

        with open(filepath+test_file_name, "w") as f:
            json.dump(self.test_examples, f)
            print("save test")

    def _read_json(self, filepath, train_file_name="train.txt", val_file_name="val.txt", test_file_name="test.txt"):
        with open(filepath+train_file_name, "r") as f:
            self.train_examples = json.load(f)

        with open(filepath+val_file_name, "r") as f:
            self.val_examples = json.load(f)

        with open(filepath+test_file_name, "r") as f:
            self.test_examples = json.load(f)

    def _shuffle_fisher_yates(self, samples):
        # Fisher-Yates shuffle algorithm.
        # https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
        for i in range(len(samples)-1, 0, -1):
            j = random.randint(0, i) # 0 and i are inclusive.
            samples[i], samples[j] = samples[j], samples[i]
        return samples

    #TODO: add support for more efficient 2D batching with additional restrictions. (such as fixed batch size beforehand,
    #   and only being able to perform shuffle within batches, then across blocks of batches. Add support in get_batch_tfdec
    #   function.

    # tfdec stands for transformer decoder support.
    def get_batch_tfdec(self, type_, shuffle=True, return_tensors=False):
        # note batch strategy should be done beforehand
        # want to shuffle in a document dependant manner...
        samples = None
        if type_ == "train":
            samples = copy.deepcopy(self.train_examples)
        elif type_ == "val":
            samples = copy.deepcopy(self.val_examples)
        elif type_ == "test":
            samples = copy.deepcopy(self.test_examples)

        samples = self._shuffle_fisher_yates(samples)

        for i in range(0, math.ceil(len(samples)/self.batch_size)):
            if i == math.ceil(len(samples)/self.batch_size)-1:
                ret_ = samples[i*self.batch_size:]
            else:
                ret_ = samples[i*self.batch_size:(i+1)*self.batch_size]
            # catches case is last batch size is zero.
            if len(ret_) == 0:
                ret_ = None # make this iterable via yield.
            if ret_ is not None and return_tensors:
                ret_ = tf.convert_to_tensor(ret_)
            yield ret_

if __name__ == "__main__":

    max_seq_len = 512
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

    folderpath = "/large_data/wikitext-103"
    # un-comment below to process the data, then save.
    #load_tokenized_tokens = [False, '/large_data/wikitext-103/',
    #                         'train8k.txt', 'val8k.txt', 'test8k.txt']
    #wikitext_loader = Wikitext_103_Loader(4, "2D_batching", tokenizer,
    #             folderpath=folderpath, max_seq_len=max_seq_len, load_tokenized_tokens=load_tokenized_tokens)
    #wikitext_loader.save_json("/large_data/wikitext-103/", train_file_name="train_ids_2D.txt",
    #                          val_file_name="val_ids_2D.txt", test_file_name="test_ids_2D.txt")

    # below loads preprocessed data already
    load_tokenized_tokens = [True, '/large_data/wikitext-103/',
                             'train_ids_2D.txt', 'val_ids_2D.txt', 'test_ids_2D.txt']
    wikitext_loader = Wikitext_103_Loader(4, "2D_batching", tokenizer,
                                        folderpath=folderpath, max_seq_len=max_seq_len,
                                        load_tokenized_tokens=load_tokenized_tokens)

    counter = 0
    for batch in wikitext_loader.get_batch_tfdec(type_="train", shuffle=True, return_tensors=False):
        counter += 1
        if counter % 100 == 0:
            #print("batch: ", batch)
            #print("train:", tokenizer.batch_decode(batch[0]))
            print(len(batch))
    print("Train counter:", counter)

    counter = 0
    for batch in wikitext_loader.get_batch_tfdec(type_="val", shuffle=True, return_tensors=False):
        counter += 1
        if counter % 10 == 0:
            #print("val:", tokenizer.batch_decode(batch[0]))
            #print("batch: ", batch)
            print(len(batch))
    print("Val counter:", counter)

    counter = 0
    for batch in wikitext_loader.get_batch_tfdec(type_="test", shuffle=True, return_tensors=False):
        counter += 1
        if counter % 10 == 0:
            print("test:", tokenizer.batch_decode(batch[0]))
            #print("batch: ", batch)
            print(len(batch))

    print("Test counter:", counter)






