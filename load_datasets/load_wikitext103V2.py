import sys
sys.path.append("..")

import tensorflow as tf
import numpy as np
import re
import json
import random
from text_processing.get_tokenizer import get_tfxl_tokenizer

class load_wikitext103V2(object):
    '''
    Class: load_wikitext103V2
    Description: Version 2 of class that processes wikitext103. Emphasis here is on simplicity!
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
            self.start_tok_id = self.tokenizer.encode(self.start_tok, add_special_tokens=True, pad_to_max_length=False,
                                        return_token_type_ids=False, return_attention_mask=False)
            if len(self.start_tok_id) != 1 and (isinstance(self.start_tok_id, list)):
                raise Exception(f"The number of ids the start token is encoded into should be one, got {len(self.start_tok_id)}!")
            else:
                self.start_tok_id = self.start_tok_id[0]
            print(f"The start token id is: {self.start_tok_id}")

            self.end_tok_id = self.tokenizer.encode(self.end_tok,
                                        add_special_tokens=True,
                                        pad_to_max_length=False,
                                        return_token_type_ids=False,
                                        return_attention_mask=False)
            if len(self.end_tok_id) != 1 and (isinstance(self.end_tok_id, list)):
                raise Exception(f"The number of ids the end token is encoded into should be one, got {len(self.end_tok_id)}!")
            else:
                self.end_tok_id = self.end_tok_id[0]
            print(f"The end token id is: {self.end_tok_id}")
            self.pad_tok_id = self.tokenizer.encode(self.pad_tok, add_special_tokens=True, pad_to_max_length=False,
                                                      return_token_type_ids=False, return_attention_mask=False)
            if len(self.pad_tok_id) != 1 and (isinstance(self.pad_tok_id, list)):
                raise Exception(
                    f"The number of ids the pad token is encoded into should be one, got {len(self.pad_tok_id)}!")
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

    def _process_raw_file(self):

        data = None
        with open(self.filepath, "r") as f:
            data = f.readlines()

        if self.strategy == "default":
            counter = 0
            main_heading = None
            document_content = ""
            for i in range(len(data)):
                if re.match(self.main, data[i]):
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



    def _default_tok_strategy_iterator(self, shuffle, pad, nm_aux_tokens):

        nm_aux_tok = []
        for word in nm_aux_tokens:
            w = tokenizer.encode(word,
                     add_special_tokens=False,
                     pad_to_max_length=False,
                     return_token_type_ids=False,
                     return_attention_mask=False)
            if len(w) > 1: raise Exception(f"{word} should only have on id associated with it in the tokenizer. Something went wrong!")
            else: w = w[0]
            nm_aux_tok.append(w)

        keys = self.data_dict.keys()
        if shuffle:
            random.shuffle(keys) # shuffles inplace.

        for i, key in enumerate(keys):

            article_ids = self.tokenizer.encode_plus(self.data_dict[key],
                                       add_special_tokens=False,
                                       pad_to_max_length=False,
                                       return_token_type_ids=False,
                                       return_attention_mask=False)
            start_index = 0
            end_index = self.max_seq_len
            tar_inp = None
            nm_tar_inp = None
            tar_real = None
            article_len = len(article_ids)
            while True:
                if start_index >= article_len: break # breaking condition for the article.

                if end_index < article_len-1:
                    tar_inp = article_ids[start_index:end_index]
                    tar_real = article_ids[start_index+1:end_index+1] # TODO: test and make sure this works.
                else:
                    tar_inp = article_ids[start_index:article_len-1]
                    tar_real = article_ids[start_index+1:article_len] # i.e. shifted to the right.

                tar_inp = tar_inp + [self.pad_tok_id for _ in range(self.max_seq_len-len(tar_inp))]
                nm_inp = nm_aux_tok + tar_inp
                tar_real = tar_real + [self.pad_tok_id for _ in range(self.max_seq_len-len(tar_real))]

                tar_inp = tf.cast(tf.convert_to_tensor(np.asarray(tar_inp)), dtype=tf.dtypes.int64)
                tar_real = tf.cast(tf.convert_to_tensor(np.asarray(tar_real)), dtype=tf.dtypes.int64)
                nm_inp = tf.cast(tf.convert_to_tensor(np.asarray(nm_inp)), dtype=tf.dtypes.int64)

                start_index = end_index
                end_index += self.max_seq_len

                yield tar_inp, nm_inp, tar_real

                # self.start_tok_id

    def get_tf_dataset_generator(self, process_strategy, shuffle=False, pad=True, *nm_aux_tokens):

        if len(nm_aux_tokens) > 0:
            for word in nm_aux_tokens:
                assert isinstance(word, str), f"The word should be of string format. {word} is of format {type(word)}!"

        aux_tok_id = [word for word in nm_aux_tokens]

        generator = None

        if process_strategy == "default_tokenize":
            generator = tf.data.Dataset.from_generator(self._default_strategy_iterator(shuffle, pad), output_types=(tf.dtypes.int64,
                                                                                                                    tf.dtypes.int64,
                                                                                                                    tf.dtypes.int64))
        elif process_strategy == "default_string":
            pass
        elif process_strategy == "default_str_tok": # note: needed for my reading strategies later.
            pass
        else: raise Exception(f"Invalid processing strategy: {process_strategy}")

        return generator











