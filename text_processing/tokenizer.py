'''
File name: tokenizer.py
Author: Kobe Knowles
Date created: 05/07/21
Data last modified: 17/07/21
Python Version: 3.6
Tensorflow version: 2
'''

import transformers
from transformers import TransfoXLTokenizer

class Tokenizer(object):
    '''
    Class: Tokenizer \n
    Description: Parent class that takes various forms of tokenizers and puts them in a
        general class with support for the typical functions. Support is only given in the parent class
        to transformers library's tokenizers. \n
    Attributes:
        tokenizer: A tokenizer from the huggingface transformers library. \n
        vocab_size_dec: (int) The vocabulary size of the decoder in a transformer, it is only updated when instructed to do
            so when a token is added it doesn't change the vocabulary size, allowing tokens to be added even after the
            vocabulary size in the decoder has been initilaized and set as a constant value.
            (added tokens are appended at an id greater than the decoder's vocabulary size) \n
        vocab_size: (int) The vocabulary size of the tokenizer.
    '''

    def __init__(self, tokenizer):
        '''
        Function: __init__ \n
        Description: Initialize the tokenizer class with the passed parameters. \n
        Input:
            tokenizer: A tokenizer from the huggingface transformers library.
        '''

        self.tokenizer = tokenizer
        self.vocab_size_dec = len(self.tokenizer.get_vocab().keys()) # this will be used in the transformer decoder layer. It separates `auxiliary' tokens that will never be part of the output.
        self.vocab_size = len(self.tokenizer.get_vocab().keys())
        self.special_tokens = []

    def get_vocab_size_dec(self):
        return self.vocab_size_dec

    def get_vocab_size(self):
        return self.vocab_size

    def add_tokens_list(self, tokens, update_vocab_size_dec=True):
        assert isinstance(tokens, list), f"The 'tokens' variable should be a list (of strings), got {type(tokens)}!"
        for tok in tokens:
            assert isinstance(tok, str), f"The contents of the 'tokens' list should all be strings, got {type(tok)}!"
            self.tokenizer.add_tokens(tok)
            self.vocab_size += 1
            if update_vocab_size_dec: self.vocab_size_dec += 1

    def add_token_string(self, token, update_vocab_size_dec=True):
        assert isinstance(token, str), f"The 'token' variable should be a string, got {type(token)}!"
        self.tokenizer.add_tokens(token)
        self.vocab_size += 1
        if update_vocab_size_dec: self.vocab_size_dec += 1

    def get_vocab_dict(self):
        return self.tokenizer.get_vocab()

    def encode_single(self, input):
        assert isinstance(input, str), f"The input ({input}) is not of type string, got {type(input)}!"
        return self.tokenizer.encode(input, add_special_tokens=False, pad_to_max_length=False) # this will be a list.

    def encode_single_plus(self, input):
        assert isinstance(input, str), f"The input ({input}) is not of type string, got {type(input)}!"
        return self.tokenizer.encode_plus(input,
                                       add_special_tokens=False,
                                       pad_to_max_length=False,
                                       return_token_type_ids=True,
                                       return_attention_mask=True) # this will be a dictionary.

    def batch_encode_plus(self, input):
        assert isinstance(input, list), f"The input ({input}) is not of type list, got {type(input)}!"
        return self.tokenizer.batch_encode_plus(input,
                                       add_special_tokens=False,
                                       pad_to_max_length=False,
                                       return_token_type_ids=True,
                                       return_attention_mask=True) # this will be a dictionary.

    def batch_encode(self, input):
        assert isinstance(input, list), f"The input ({input}) is not of type list, got {type(input)}!"
        return self.tokenizer.batch_encode_plus(input,
                                       add_special_tokens=False,
                                       pad_to_max_length=False) # this will be a dictionary.

    def encode_single_id_string(self, input):
        assert isinstance(input, str), f"The input ({input}) is not of type string, got {type(input)}!"
        out1 = self.tokenizer.encode_plus(input,
                                       add_special_tokens=False,
                                       pad_to_max_length=False,
                                       return_token_type_ids=False,
                                       return_attention_mask=False)
        out2 = self.tokenizer.tokenize(input) # this is for the reading stratetgies later, need the original word when it is unknown.
        return (out1, out2)

    def encode_single_string_only(self, input):
        assert isinstance(input, str), f"The input ({input}) is not of type string, got {type(input)}!"
        return self.tokenizer.tokenize(input) # this is for the reading stratetgies later, need the original word when it is unknown.

    #TODO add in decoding methods.
    def batch_decode(self, input):
        # list of list of integers.
        return self.tokenizer.batch_decode(input)

    def decode(self, input):
        # list of integers
        return self.tokenizer.decode(input)

if __name__ == "__main__":

    tok = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
    tokenizer = Tokenizer(tok)
    print(tokenizer.encode_single_id_string("The lakers have 17 NBA championships!!!! <dec>"))

    print(tokenizer.get_vocab_size_dec())
    print(tokenizer.get_vocab_size())
    tokenizer.add_tokens_list(["<dec>", "<enc>"], update_vocab_size_dec=False)
    print(tokenizer.get_vocab_size_dec())
    print(tokenizer.get_vocab_size())
    print(tokenizer.encode_single_id_string("The lakers have 17 NBA championships!!!! <dec>"))
    tokenizer.add_token_string("<confidence>", update_vocab_size_dec=True)
    print(tokenizer.get_vocab_size_dec())
    print(tokenizer.get_vocab_size())
    #print(tokenizer.batch_encode_plus("The lakers have 17 NBA championships!!!!"))


