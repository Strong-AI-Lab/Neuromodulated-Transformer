'''
Filename: model_hyper_parameters.py
Description: Use/modify this file to create model hyperparameters wrapped up in a class.
'''

import tensorflow as tf
import numpy as np
from transformers import TransfoXLTokenizer
from text_processing.tokenizer import Tokenizer

class TestModel:
    '''
    class: TestModel \n
    Desctiption: The hyperparameters of a model put into a class for ease of access.
    '''
    def __init__(self, strategy=None, batch_size=4, rate=0.1):
        '''
        Function: __init__ \n
        Description: Initialize the hyperparameters of the model. \n
        Input:
            strategy: (str) The Strategy to distribute the data across multiple GPUs. If no strategy it is set to None. \n
            batch_size: (int) The number of examples per batch. \n
            rate: (float) The dropout percentage of the dropout layers.
        '''

        self.strategy = None
        if strategy is not None:
            if strategy == "MirroredStrategy":
                self.strategy = tf.distribute.MirroredStrategy()
            #elif... add support for more here.
            else:
                assert Exception(f"Invalid strategy")

        self.batch_size = batch_size
        self.rate = rate

        tok = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
        self.tokenizer = Tokenizer(tok)
        self.tokenizer.add_tokens_list(["<s>", "</s>", "<pad>", "<dec>", "<lm>", "<confidence>"], update_vocab_size_dec=True)

        self.num_layers_dec = 12
        self.num_layers_nm = 6
        self.d_model = 512
        self.num_heads = 8
        self.dff = self.d_model*2

        self.num_aux_tokens = 0
        self.max_seq_len_dec = 512
        self.max_seq_len_nm = self.max_seq_len_dec + self.num_aux_tokens

        self.parallel_layers = dict()
        self.parallel_layers["nm_attn_gate"] = ["GateLayerAttn", 3, True]  # True inside the parenthesis mean aux loss + additional layers are to be created for it.
        self.parallel_layers["nm_eol_gate"] = ["EncoderLayer", 3, True]
        self.num_aux_losses = 2 # the number of True values in the lists above.

        self.nm_attn = True
        self.nm_eol = True

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,
                                                                         reduction='none')
        self.learning_rate = tf.keras.optimizers.schedules.CosineDecay(0.00025, decay_steps=100000)

        self.rel_pos_emb = True
        self.max_position_encoding_dec = 2000
        self.max_position_encoding_nm = 2000

        def __str__(self):
            return '''Initialization parameters of a model with a dimension size of 512 and a
            maximum sequence length of 512 tokens for the decoder and 522 for the nm encoder... this is a sample description.
            '''

        def __repr__(self):
            return '''Initialization parameters of a model with a dimension size of 512 and a
            maximum sequence length of 512 tokens for the decoder and 522 for the nm encoder... this is a sample description.
            '''