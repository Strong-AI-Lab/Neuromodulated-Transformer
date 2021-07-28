'''
File name: load_wikitext.py
Author: Kobe Knowles
Date created: 05/07/21
Data last modified: 17/07/21
Python Version: 3.6
Tensorflow version: 2
'''


import sys
sys.path.append("../..")

import tensorflow as tf
import numpy as np
import re
import json
import random
from text_processing.tokenizer import Tokenizer
from transformers import TransfoXLTokenizer

class load_wikitext103:
    '''
    Class: load_wikitext103V2
    Description: Class that pre-processes wikitext103.
    Input:
        filepath: (string) Filepath representing the string of one of the train, validation and test files.
        max_seq_len: (int) The maximum sequence length required for the pad_to_max_lenght parameter.
        tokenizer: Tokenizer from the transformers module.
        start_tok: (string) The start token to append at the start of each data set item.
        end_tok: (string) The end token to append at the end of each data set item, or for wikitext's case, to replace
            newline (\n) characters.
        pad_tok: (string) The token that is to be used to pad the input sequence to the maximum sequence length.
        pad_to_max_length: (bool) True is pad to the maximum length during dataset generation; False otherwise.
        strategy: (string) Type of strategy when processing the data.
            "default" is the default strategy that processes data based on article headings.
        heading_pattern: (string)
        load_data: (list)
    '''
    def __init__(self, filepath, max_seq_len=512, tokenizer=None, start_tok="<s>", end_tok="</s>", pad_tok="<pad>",
                 pad_to_max_length=True, strategy="default", load_data=[False, ""]):

        self.filepath = filepath
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.pad_to_max_length = pad_to_max_length

        self.start_tok = start_tok
        self.end_tok = end_tok
        self.pad_tok = pad_tok

        self.start_tok_id = None
        self.end_tok_id = None
        self.pad_tok_id = None

        if tokenizer is not None:
            self.start_tok_id = self.tokenizer.encode_single(self.start_tok)
            if len(self.start_tok_id) != 1 and (isinstance(self.start_tok_id, list)):
                raise Exception(f"The number of ids the start token is encoded into should be one, got {self.start_tok_id}!")
            else:
                self.start_tok_id = self.start_tok_id[0]
            print(f"The start token id is: {self.start_tok_id}")

            self.end_tok_id = self.tokenizer.encode_single(self.end_tok)
            if len(self.end_tok_id) != 1 and (isinstance(self.end_tok_id, list)):
                raise Exception(f"The number of ids the end token is encoded into should be one, got {len(self.end_tok_id)}!")
            else:
                self.end_tok_id = self.end_tok_id[0]
            print(f"The end token id is: {self.end_tok_id}")
            self.pad_tok_id = self.tokenizer.encode_single(self.pad_tok)
            if len(self.pad_tok_id) != 1 and (isinstance(self.pad_tok_id, list)):
                raise Exception(f"The number of ids the pad token is encoded into should be one, got {len(self.pad_tok_id)}!")
            else:
                self.pad_tok_id = self.pad_tok_id[0]
            print(f"The pad token id is: {self.pad_tok_id}")


        self.strategy = strategy
        assert self.strategy in ["default"], f"The strategy {self.strategy} is not supported!"

        self.main_heading_pattern = " = [^=]*[^=] = \n"
        # "[= ][^=]*[^=] [= ]" # for the main headings only.
        self.any_heading_pattern = ' [= ]+[^=]*[^=] [= ]+\n'

        self.data_dict = {} # each article heading will be put here with the associated input.

        if not load_data[0]:
            self._process_raw_file()
        else:
            self._read_json(load_data[1])

        #print(self.data_dict)

    def _process_raw_file(self):

        data = None
        with open(self.filepath, "r") as f:
            data = f.readlines()

        if self.strategy == "default":
            counter = 0
            main_heading = None
            document_content = ""
            for i in range(len(data)):
                if re.match(self.main_heading_pattern, data[i]):
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
                else:
                    document_content += re.sub("\n$", str(self.end_tok), data[i])
            self.data_dict[main_heading] = document_content
        else:
            raise Exception(f"Invalid strategy: {self.strategy}")

    def save_json(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.data_dict, f)
            print("Save successful at " + str(filepath))

    def _read_json(self, filepath):
        with open(filepath, "r") as f:
            self.data_dict = json.load(f)
            print("Load successful at " + str(filepath))


    def default_tok_strategy_iterator(self):
        # this is a naive way to go about it.
        # shufle, pad, nm_aux
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

                tar_inp = tf.cast(tf.convert_to_tensor(np.asarray(tar_inp)), dtype=tf.dtypes.int64)
                tar_real = tf.cast(tf.convert_to_tensor(np.asarray(tar_real)), dtype=tf.dtypes.int64)
                nm_inp = tf.cast(tf.convert_to_tensor(np.asarray(nm_inp)), dtype=tf.dtypes.int64)

                start_index = end_index
                end_index += self.max_seq_len

                yield tar_inp, tar_real, nm_inp

    def sliding_window_article_iterator(self):
        # this is a naive way to go about it.
        # shufle, pad, nm_aux
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

            article_ids = self.tokenizer.encode_single(self.data_dict[key])
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

                if self.pad: # TODO: interesting <pad> tokens to be added before hand because of sliding windows.
                    #tar_inp = tar_inp + [self.pad_tok_id for _ in range(self.max_seq_len-len(tar_inp))]
                    #tar_real = tar_real + [self.pad_tok_id for _ in range(self.max_seq_len-len(tar_real))]
                    tar_inp = [self.pad_tok_id for _ in range(self.max_seq_len-len(tar_inp))] + tar_inp
                    tar_real = [self.pad_tok_id for _ in range(self.max_seq_len-len(tar_real))] + tar_real

                nm_inp = nm_aux_tok + tar_inp

                tar_inp = tf.cast(tf.convert_to_tensor(np.asarray(tar_inp)), dtype=tf.dtypes.int64)
                tar_real = tf.cast(tf.convert_to_tensor(np.asarray(tar_real)), dtype=tf.dtypes.int64)
                nm_inp = tf.cast(tf.convert_to_tensor(np.asarray(nm_inp)), dtype=tf.dtypes.int64)

                start_index += self.sliding_window
                end_index += self.sliding_window

                if counter == 0:
                    counter += 1
                    yield tar_inp, tar_real, nm_inp, tf.ones((1), dtype=tf.dtypes.int64)
                    # 1 indicates that we are at the first sentence in the article. (we should calculate the loss for all tokens)
                else:
                    yield tar_inp, tar_real, nm_inp, tf.zeros((1), dtype=tf.dtypes.int64)
                    # 0 indicates that we are not at the first sentence in the article. (we should calculate the loss only for the last sliding_window tokens)

    def get_tf_dataset_generator(self, process_strategy, shuffle=False, pad=True, sliding_window=None, *nm_aux_tokens):

        self.shuffle = shuffle
        self.pad = pad

        if len(nm_aux_tokens) > 0:
            for word in nm_aux_tokens:
                assert isinstance(word, str), f"The word should be of string format. {word} is of format {type(word)}!"

        aux_tok = [word for word in nm_aux_tokens]
        self.aux_tok = aux_tok

        generator = None
        if process_strategy == "default_tokenize":
            generator = tf.data.Dataset.from_generator(self.default_tok_strategy_iterator,
                                                       output_types=(tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64))
        elif process_strategy == "sliding_window_article":
            assert isinstance(sliding_window, int), f"If sliding_window variable is must be an integer, got {type(sliding_window)}"
            self.sliding_window = sliding_window
            generator = tf.data.Dataset.from_generator(self.sliding_window_article_iterator,
                                                       output_types=(tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64,
                                                                     tf.dtypes.int64))
        elif process_strategy == "default_str_tok": # note: needed for my reading strategies later.
            pass
        else: raise Exception(f"Invalid processing strategy: {process_strategy}")

        return generator

if __name__ == "__main__":

    tok = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
    tokenizer = Tokenizer(tok)
    tokenizer.add_tokens_list(["<s>", "</s>", "<pad>", "<unk>", "<lm>", "<confidence>"])

    filepath = "/large_data/wikitext-103/wiki.valid.tokens"
    end_tok = "</s>"
    pad_tok = "<pad>"
    strategy = "default"
    #load_data = [False, ""]
    load_data = [True, "/large_data/wikitext-103/processed_data/train_heading_default_strategy.txt"]

    wiki_loader = load_wikitext103(filepath=filepath, tokenizer=tokenizer, end_tok=end_tok, pad_tok=pad_tok, strategy=strategy,
                                     load_data=load_data)
    #wiki_loader.save_json("/large_data/wikitext-103/processed_data/val_heading_default_strategy.txt")
    #load_data = [True, "/large_data/wikitext-103/val_test_v2.txt"]
    #wiki_loader = load_wikitext103V2(filepath=filepath, tokenizer=tokenizer, end_tok=end_tok, pad_tok=pad_tok, strategy=strategy,
    #                                 load_data=load_data)
    process_strategy = "sliding_window_article"
    shuffle = False
    pad = True
    generator = wiki_loader.get_tf_dataset_generator(process_strategy, shuffle, pad, 100, "<lm>", "<confidence>")
    counter = 0
    for batch, (inp, tar, nm, item) in enumerate(generator):
        print(f"batch: {batch}")
        print(f"inp.shape: {inp.shape} \t inp:{tokenizer.decode(inp)} \n"
              f"nm.shape: {nm.shape} \t nm: {tokenizer.decode(nm)} \n"
              f"tar.shape: {tar.shape} \t tar: {tokenizer.decode(tar)} \n"
              f"item: {item} \n")
        counter += 1
        if counter == 20: break
    '''
    process_strategy = "default_tokenize"
    shuffle = False
    pad = True
    generator = wiki_loader.get_tf_dataset_generator(process_strategy, shuffle, pad, "<lm>", "<confidence>")
    counter = 0
    for batch, (inp, tar, nm) in enumerate(generator):
        if batch < 12000: continue
        if batch >= 12003: break
        print(f"batch: {batch}")
        print(f"inp.shape: {inp.shape} \t inp: {inp} -- {tokenizer.decode(inp)} \n"
              f"nm.shape: {nm.shape} \t nm: {tokenizer.decode(nm)} \n"
              f"tar.shape: {tar.shape} \t tar: {tokenizer.decode(tar)} \n")

        if counter == 20: break
    '''
    '''
    filepath = "/large_data/wikitext-103/wiki.test.tokens"
    wiki_loader = load_wikitext103(filepath=filepath, tokenizer=tokenizer, end_tok=end_tok, pad_tok=pad_tok, strategy=strategy,
                                     load_data=load_data)
    wiki_loader.save_json("/large_data/wikitext-103/processed_data/test_heading_default_strategy.txt")

    filepath = "/large_data/wikitext-103/wiki.train.tokens"
    wiki_loader = load_wikitext103(filepath=filepath, tokenizer=tokenizer, end_tok=end_tok, pad_tok=pad_tok, strategy=strategy,
                                     load_data=load_data)
    wiki_loader.save_json("/large_data/wikitext-103/processed_data/train_heading_default_strategy.txt")
    '''

    '''
    process_strategy="default_tokenize"
    shuffle=True
    pad=True
    generator = wiki_loader.get_tf_dataset_generator(process_strategy, shuffle, pad, "<dec>", "<lm>", "<confidence>")
    generator = generator.batch(4)

    print("<dec> token id:",tokenizer.encode_single("<dec>"))
    print("<lm> token id:", tokenizer.encode_single("<lm>"))
    print("<confidence> token id:", tokenizer.encode_single("<confidence>"))

    counter = 0
    for batch, (inp, nm, tar) in enumerate(generator):
        print(f"batch: {batch}")
        print(f"inp.shape: {inp.shape} \t inp: {inp} \n"
              f"nm.shape: {nm.shape} \t nm: {nm} \n"
              f"tar.shape: {tar.shape} \t tar: {tar} \n")
        counter += 1
        if counter == 20: break
    '''












