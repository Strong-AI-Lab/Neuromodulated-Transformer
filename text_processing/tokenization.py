'''
This file will contain code implementation classes.
'''

import requests
import tensorflow as tf
import tensorflow_text as tf_text
import copy
import numpy as np

def download_WordPieceTokenizer8K(filename='vocab.txt'):
    # https://www.tensorflow.org/tutorials/tensorflow_text/tokenizers
    url = "https://github.com/tensorflow/text/blob/master/tensorflow_text/python/ops/test_data/test_wp_en_vocab.txt?raw=true"
    f = requests.get(url)
    open(filename, 'wb').write(f.content)

class BertTokenizer:
    '''
    Class: BertTokenizer
    Description: A simple implemtation of a tokenizer that builds upon the BertTokenizer.
    Input:
        filename: (string) Represents the path to the vocabulary file.
            note: Can be downloaded in the correct format with download_WordPieceTokenizer function.
    '''
    def __init__(self, filename, max_seq_len=None, lower_case=False):

        self.tokenizer = tf_text.BertTokenizer(filename, lower_case=lower_case)

        self.vocab_dict = dict()
        self.id_to_vocab = dict()
        self.create_vocab_dict(filename)
        self.max_seq_len = max_seq_len


    def create_vocab_dict(self, filename):
        '''
        Function: create_vocab_dict
        Description: Creation of the vocabulary dictionary which associates each token in the file to
        Input:
            filename: (string) Path to a binary file that holds the vocabulary tokens.
        Return:
            (boolean) True if the creation of such a dictionary is achieved; False otherwise.
        '''
        data = None
        with open(filename, "rb") as v:
            data = v.readlines()

        #print("\nLength of data is\n", len(data))
        for i in range(len(data)):
            self.vocab_dict[data[i].decode('utf-8').rstrip("\n")] = i
            self.id_to_vocab[i] = data[i].decode('utf-8').rstrip("\n")

    def tokenize(self, input_text, padding=True):
        '''
        Function: tokenize
        Description: Function that performs tokenization to the integer id version.
        Input:
            input_text: (list; [batch_size, seq_len (varies)]) Input text to be tokenized.
            padding: (boolean) True if padding is to be performed; False otherwise.
        Return:
            tokenized_tokens: (list; [batch_size, seq_len (varies)])
            reduce_subword_dim_tokens: (list; [batch_size, seq_len])

        '''
        tokenized_tokens = self.tokenizer.tokenize(input_text).to_list()
        #print("testsetset",tokenized_tokens)

        reduce_subword_dim_tokens = []
        # iterate through the batch_size (the number of sequences).
        for seq in range(len(tokenized_tokens)):
            seq_list = []
            # iterate through each item in sequence.
            for i in range(len(tokenized_tokens[seq])):
                # check if item is a list
                if isinstance(tokenized_tokens[seq][i], list):
                    # iterate through sub word elements. e.g. p ##o ##ker.
                    for j in range(len(tokenized_tokens[seq][i])):
                        seq_list.append(tokenized_tokens[seq][i][j])
                else:
                    seq_list.append(tokenized_tokens[seq][i])
            reduce_subword_dim_tokens.append(seq_list)

        if padding:
             reduce_subword_dim_tokens = self.padding(copy.deepcopy(reduce_subword_dim_tokens))
        # TODO: check [START] and [EOS] or [END]
        return reduce_subword_dim_tokens, tokenized_tokens

    def padding(self, tokens):
        '''
        Function: padding
        Description: Applies padding to the input tokens.
        Input:
            tokens: (list; [batch_size, seq_len (varies)])
        Return:
            pad_tokens: (list; [batch_size, seq_len])
        '''
        #print()
        #print("tokens!!", tokens, "\n")
        if self.max_seq_len is None:
            max_seq_len = self._getMaxSeqLen(tokens)
        else:
            max_seq_len = self.max_seq_len

        #print(max_seq_len)
        pad_tokens = tokens
        for i in range(len(pad_tokens)):
            if len(pad_tokens[i]) < max_seq_len:
                while True:
                    pad_tokens[i].append(0) # note: update this later to the padded token in a dictionary.
                    if len(pad_tokens[i]) == max_seq_len: break
        return pad_tokens

    def _getMaxSeqLen(self, tokens):
        '''
        Function: _getMaxSeqLen
        Description: Calculates the max sequence length and returns the value.
        tokens: (list; [batch_size, seq_len (varies)])
        Return:
            max_seq_len: (int)
        '''
        max_seq_len = 0
        for i in range(len(tokens)):
            max_seq_len = max(max_seq_len, len(tokens[i]))
        return max_seq_len

    def detokenize(self, tokens, pad_token = "[PAD]", sub_word_char = "##"):
        '''
        Function: detokenize
        Description: Detokenizes the token ids according to the vocabulary.
        Input:
            tokens: (tf.Tensor; [batch_size, seq_len]) Possibly padded sequence ids to detokenize.
        Return:
            (list; [batch_size, seq_len (varies)])
        '''
        assert len(sub_word_char) >= 2, "Length of sub_word_char string should be at least 2."

        token_list = tokens.numpy().tolist()
        # iterate through each sequence element.
        token_strings = []
        for seq in range(len(token_list)):
            # iterate through each item in list.
            str_ = ''
            prev_item = None
            for i in range(len(token_list[seq])):

                item = self.id_to_vocab[token_list[seq][i]]

                if len(item) >= 2:
                    if item[0:2] == sub_word_char:
                        item = item[2:]
                    else:
                        if i == 0 or token_list[seq][i] == 0:
                            pass
                        else:
                            str_ += " "
                else:
                    if i == 0 or token_list[seq][i] == 0:
                        pass
                    #TODO: # need test cases here for !.' etc... so know not to put a space before.
                    else:
                        str_ += " "
                #print("item", item)
                if token_list[seq][i] != 0: # otherwise it is a padded token.
                    str_ += item
                prev_item = item

            token_strings.append(str_)
        return token_strings

'''
    Below class is incomplete. It is a good baseline for an implementation of a tokenizer 
    myself if I ever get around to it.
    BertTokenizer does what I need initially.
'''
# note: below class should be able to chain multiple tokenizers together.
class BaseTokenizer:
    '''
    Class: BaseTokenizer
    Description: A simple implemtation of a tokenizer that splits via white spaces.
    Input:
        filename: (string) Represents the path to the vocabulary file.
            note: Can be downloaded in the correct format with download_WordPieceTokenizer function.
    '''
    def __init__(self, filename):

        self.ws_tokenizer = tf_text.WhitespaceTokenizer()
        self.sw_tokenizer = tf_text.UnicodeScriptTokenizer(filename)
        self.vocab_dict = dict()

    def tokenize_text(self, text):
        '''
        Function: tokenize_text
        Description: Tokenizes the input text, returning
        Input:
            text: (list; [batch_size, seq_len (note: this can vary, padding is done later]) Input text to tokenize.
        Return:
            subtokens: (list; [batch_size, seq_len (varies)])
        '''
        tokens = self.ws_tokenizer.tokenize(text)
        #subtokens = self.sw_tokenizer.tokenize(tokens)[0]
        return tokens

    def get_token_ids(self, tokens, put_start_token, put_end_token):
        '''
        Function: get_token_ids
        Description: Convert tokens to their integer ids in the vocabulary dictionary.
        Input:
            tokens: (list; [batch_size, seq_len (varies), 1 (b'')]) Tokens to convert to integers.
            put_start_token: (list; batch_size) Each list item is either True if append start token at the
                beginning of sequence; False otherwise.
            put_end_token: (list; batch_size) Each list item is either True if append end token at the
                end of sequence; False otherwise.
        Return:
            token_id_list: (list; [batch_size, seq_len (varies)]) List of integers representing the tokens.
        '''
        assert "[UNK]" in self.vocab_dict.keys(), "[UNK] token is missing from the vocabulary!"
        assert "[START]" in self.vocab_dict.keys(), "[START] token is missing from the vocabulary!"
        assert "[END]" in self.vocab_dict.keys(), "[END] token is missing from the vocabulary!"

        token_id_list = []

        # iterate through each item in batch (i.e. each sequence).
        for seq in range(len(tokens)):
            token_id_seq = []

            if put_start_token[seq]:
                token_id_seq.append(self.vocab_dict["[START]"])

            # iterate through each item in sequence
            for tok in range(len(tokens[b])):
                encoded_token = tokens[seq][tok][0].encode('utf-8')
                # get ID
                id_ = None
                if encoded_token in self.vocab_dict.keys():
                    id_ = self.vocab_dict[encoded_token]
                else:
                    id_ = self.vocab_dict["[UNK]"]
                token_id_seq.append(id_)

            if put_end_token[seq]:
                token_id_seq.append(self.vocab_dict["[END]"])

            token_id_list.append(token_id_seq)

        return token_id_list

    def pad_tokens(self, tokens):
        '''
        Function: pad_tokens
        Description: Pad the tokens passed in as input and return.
        Input:
            tokens: (list; [batch_size, seq_len (varies)])
        Return:
            pad_tokens: (list; [batch_size, max_seq_len])
        '''
        assert "[PAD]" in self.vocab_dict.keys(), "[PAD] token is missing from the vocabulary!"
        # get max seq_len.
        # then iterate through all sequences and append pad token

        return None

    def untokenize(self, tokens):
        pass

    def create_vocab_dict(self, filename, reserve_tokens = [4, 100]):
        '''
        Function: create_vocab_dict
        Description: Creation of the vocabulary dictionary which associates each token in the file to
        Input:
            filename: (string) Path to a binary file that holds the vocabulary tokens.
            reserve_token: (list; [start (inclusive), end (exclusive)]) Reserve token id number to leave
                black in vocabulary dictionary.
        Return:
            (boolean) True if the creation of such a dictionary is achieved; False otherwise.
        '''
        data = None
        with open(filename, "rb") as v:
            data = v.readlines()

        # counter for token id.
        counter = 0
        # counter for iteration through data list.
        c1 = 0
        for i in range(reserve_tokens[0]):
            self.vocab_dict[data[counter].rstrip("\n")] = counter
            coutner += 1
            c1 += 1
        for i in range(reserve_tokens[0], reserve_tokens[1]):
            self.vocab_dict['token'+str(counter)] = counter
            counter += 1
        for i in range(c1, len(data)):
            self.vocab_dict[data[c1].rstrip("\n")] = counter
            counter += 1

    def check_vocab_len(self):
        '''
        Function: check_vocab_len
        Description: Cheks the number of words/characters in the vocabulary.
        Input:
            None
        Return:
            (int) Representing the length|number of elements in the vocabulary.
        '''
        return len(self.vocab_dict.keys())

if __name__ == "__main__":
    '''
    tokenizer = tf_text.WhitespaceTokenizer()
    tokens = tokenizer.tokenize(["What you know you can't explain, but you feel it.", "batch element two.", "batch element three halo cs30."])
    print(tokens.to_list())
    url = "https://github.com/tensorflow/text/blob/master/tensorflow_text/python/ops/test_data/test_wp_en_vocab.txt?raw=true"
    f = requests.get(url)
    filepath = "vocab.txt"
    open(filepath, 'wb').write(f.content)
    subtokenizer = tf_text.UnicodeScriptTokenizer(filepath)
    subtokens = tokenizer.tokenize(tokens)[0] # gets first batch element. Number 0.
    subtokens = tokenizer.tokenize(tokens)
    print(subtokens.to_list())
    '''
    filepath = "vocab.txt"
    #tokenizer = tf_text.BertTokenizer(filepath, token_out_type=tf.string, lower_case=True)
    #tokenizer = tf_text.BertTokenizer(filepath, token_out_type=tf.string, lower_case=True)
    #tokens = tokenizer.tokenize(["What you know you can't explain, but you feel it halo cs30.", "The lakers are the best team in the NBA!"])
    #print(tokens)
    tokenizer = BertTokenizer(filepath)
    tokenizer.create_vocab_dict(filepath)
    reduce_tokens, tokens = tokenizer.tokenize(["What you know you can't explain, but you feel it halo cs30.", "The lakers are the best team in the NBA!"])
    print("Reduced dimension tokens:\n", reduce_tokens, "\n")
    print("Tokens:\n", tokens, "\n")
    print("Reduced dimension detokenize\n", tokenizer.detokenize(tf.convert_to_tensor(reduce_tokens)), "\n")
    #print("Token detokenized:\n", tokenizer.detokenize(tf.convert_to_tensor(tokens)))
    #print(tokenizer.tokenizer._wordpiece_tokenizer.)
    reduce_tokens, tokens = tokenizer.tokenize(
        ["What you know you can't explain, but you feel it halo cs30.", "The are the best team in the NBA!"])




