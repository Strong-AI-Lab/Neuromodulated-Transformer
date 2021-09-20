'''
Filename: model_hyper_parameters.py
Description: Use/modify this file to create model hyperparameters wrapped up in a class.
'''

import tensorflow as tf
import numpy as np
from transformers import TransfoXLTokenizer
from transformers import BertTokenizer
from text_processing.tokenizer import Tokenizer
import json

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

        self.num_layers_dec = 8
        self.num_layers_nm = 8
        self.num_layers_gating = 2
        self.d_model = 512
        self.num_heads = 8
        self.dff = self.d_model*2

        self.num_aux_tokens = 0
        self.max_seq_len_dec = 512
        self.max_seq_len_nm = self.max_seq_len_dec + self.num_aux_tokens

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

class WikitextBigAbsPosEmb:
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
        self.num_layers_nm = 8
        self.num_layers_gating = 3
        self.d_model = 768
        self.num_heads = 12
        self.dff = self.d_model*2

        self.num_aux_tokens = 0
        self.max_seq_len_dec = 1024 # note: this acts as training input size as input to the generator as well.
        self.max_seq_len_nm = self.max_seq_len_dec + self.num_aux_tokens

        self.nm_attn = True
        self.nm_eol = True

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,
                                                                         reduction='none')
        self.learning_rate = tf.keras.optimizers.schedules.CosineDecay(0.00025, decay_steps=100000)

        self.rel_pos_emb = False
        self.max_position_encoding_dec = 5000
        self.max_position_encoding_nm = 5000

        def __str__(self):
            return '''Initialization parameters of a model with a dimension size of 512 and a
            maximum sequence length of 512 tokens for the decoder and 522 for the nm encoder... this is a sample description.
            '''

        def __repr__(self):
            return '''Initialization parameters of a model with a dimension size of 512 and a
            maximum sequence length of 512 tokens for the decoder and 522 for the nm encoder... this is a sample description.
            '''

class FinalModelSmall:
    '''
    class: FinalModelSmall \n
    Desctiption: The hyperparameters/parameters for the final small model.
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

        tok = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = Tokenizer(tok)
        vocab_to_add = None
        with open("../../vocabulary/vocab1.txt", "r") as f:
            vocab_to_add = json.load(f)
        #print(f"\n\n vocab to add: {vocab_to_add} \n\n")
        self.tokenizer.add_tokens_list(vocab_to_add, update_vocab_size_dec=True)

        self.num_layers_dec = 10
        self.num_layers_nm = 6
        self.num_layers_gating = 2
        self.d_model = 768
        self.num_heads = 8
        self.dff = self.d_model*4

        self.num_aux_tokens = 8 # mode, type of input/question, reading strategies aux tokens...
        self.num_reading_strategies = 6
        self.max_seq_len_dec = 768
        self.max_seq_len_nm = self.max_seq_len_dec + self.num_aux_tokens

        self.nm_attn = True
        self.nm_eol = True

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,
                                                                         reduction='none')
        self.learning_rate = tf.keras.optimizers.schedules.CosineDecay(0.001, decay_steps=5000000)

        self.rel_pos_emb = True
        self.max_position_encoding_dec = self.max_seq_len_dec
        self.max_position_encoding_nm = self.max_seq_len_nm

        self.parallel_layers = dict()
        self.parallel_layers["unknown_rs"] = ["MetacognitionSequenceLayer", 2, False]
        self.parallel_layers["aoint_rs"] = ["MetacognitionSingleLayer", 2, False]
        self.parallel_layers["highlighting_rs"] = ["MetacognitionSingleLayer", 2, False]
        self.parallel_layers["rereading_rs"] = ["MetacognitionSingleLayer", 2, False] # keep this at single. Decoder will sort out re-reading
        self.parallel_layers["summarization_rs"] = ["MetacognitionSingleLayer", 2, False]
        self.parallel_layers["paraphrasing_rs"] = ["MetacognitionSingleLayer", 2, False]

        def __str__(self):
            return '''Variables used for initialization of the final small neuromodulated transformer model.
            This is the base model includeing variables for metacognition components. \nA bert based uncased tokenizer
            from the huggingface transformers library. 
            '''

        def __repr__(self):
            return '''Variables used for initialization of the final small neuromodulated transformer model.
            This is the base model includeing variables for metacognition components. \nA bert based uncased tokenizer
            from the huggingface transformers library. 
            '''