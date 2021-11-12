import tensorflow as tf
import numpy as np

import sys
sys.path.append("..")

from transformers import TransfoXLTokenizer
from transformers import BertTokenizer
from text_processing.tokenizer import Tokenizer
import json

class V4ConfigMediumSize(object):

    def __init__(self, strategy, batch_size, loss_object, learning_rate,
                 lm_tok="<lm>", mqa_tok="<mqa>", gqa_tok="<gqa>",
                 highlighting_rs="<highlighting_rs>",
                 summarize_rs="<summarize_rs>",
                 paraphrase_rs="<paraphrase_rs>",
                 vocab_filepath="../vocabulary/vocab1.txt"):

        self.lm_tok = lm_tok
        self.mqa_tok = mqa_tok
        self.gqa_tok = gqa_tok
        self.highlighting_rs = highlighting_rs
        self.summarize_rs = summarize_rs
        self.paraphrase_rs = paraphrase_rs

        self.strategy = None
        if strategy is not None:
            if strategy == "MirroredStrategy":
                self.strategy = tf.distribute.MirroredStrategy()
            else:
                assert Exception(f"Invalid strategy")
        self.batch_size = batch_size

        self.loss_object = None
        if loss_object is None:
            # default value for the loss_object.
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
        else:
            self.loss_object = loss_object

        self.learning_rate = None
        if learning_rate is None:
            self.learning_rate = 0.0001
        else:
            self.learning_rate = learning_rate

        self.num_layers_vanilla = 12
        self.num_layers_nm = 12
        self.num_layers_mc = 3
        self.num_layers_output = 3

        self.d_model = 768
        self.num_heads = 12
        self.dff = self.d_model*4

        tok = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = Tokenizer(tok)
        vocab_to_add = None
        with open(vocab_filepath, "r") as f:
            vocab_to_add = json.load(f)
        # print(f"\n\n vocab to add: {vocab_to_add} \n\n")
        self.tokenizer.add_tokens_list(vocab_to_add, update_vocab_size_dec=True)

        self.input_vocab_size = self.tokenizer.get_vocab_size()
        self.output_vocab_size = self.tokenizer.get_vocab_size()

        self.num_aux_toks = 3 # <cls> <dec|enc|mc> <lm|mqa|gqa|highlighting_rs|summarize_rs|paraphrase_rs>
        self.max_seq_len_dec = 768
        self.max_seq_len_nm = self.max_seq_len_dec + self.num_aux_toks
        self.max_position_encoding = self.max_seq_len_nm

        self.mask_strategy = "default"
        self.rate=0.1

        self.parallel_layers = ["aoint_rs", "highlighting_rs", "unknown_rs", "summarization_rs", "paraphrasing_rs"]
        self.output_layers = ["lm", "mqa", "gqa", "highlighting_rs", "summarize_rs", "paraphrase_rs"]

        self.aux_tok_output_layer_map = {} # key:item | id(int):output_layer(str)

        self.lm_tok_id = self.tokenizer.encode_single(self.lm_tok)
        if len(self.lm_tok_id) != 1 and (isinstance(self.lm_tok_id, list)):
            raise Exception(
                f"The number of ids the start token is encoded into should be one, got {self.lm_tok_id}!")
        else:
            self.lm_tok_id = self.lm_tok_id[0]
        self.aux_tok_output_layer_map[self.lm_tok_id] = "lm"

        self.mqa_tok_id = self.tokenizer.encode_single(self.mqa_tok)
        if len(self.mqa_tok_id) != 1 and (isinstance(self.mqa_tok_id, list)):
            raise Exception(
                f"The number of ids the start token is encoded into should be one, got {self.mqa_tok_id}!")
        else:
            self.mqa_tok_id = self.mqa_tok_id[0]
        self.aux_tok_output_layer_map[self.mqa_tok_id] = "mqa"

        self.gqa_tok_id = self.tokenizer.encode_single(self.gqa_tok)
        if len(self.gqa_tok_id) != 1 and (isinstance(self.gqa_tok_id, list)):
            raise Exception(
                f"The number of ids the start token is encoded into should be one, got {self.gqa_tok_id}!")
        else:
            self.gqa_tok_id = self.gqa_tok_id[0]
        self.aux_tok_output_layer_map[self.gqa_tok_id] = "gqa"

        self.highlighting_rs_id = self.tokenizer.encode_single(self.highlighting_rs)
        if len(self.highlighting_rs_id) != 1 and (isinstance(self.highlighting_rs_id, list)):
            raise Exception(
                f"The number of ids the start token is encoded into should be one, got {self.highlighting_rs_id}!")
        else:
            self.highlighting_rs_id = self.highlighting_rs_id[0]
        self.aux_tok_output_layer_map[self.highlighting_rs_id] = "highlighting_rs"

        self.summarize_rs_id = self.tokenizer.encode_single(self.summarize_rs)
        if len(self.summarize_rs_id) != 1 and (isinstance(self.summarize_rs_id, list)):
            raise Exception(
                f"The number of ids the start token is encoded into should be one, got {self.summarize_rs_id}!")
        else:
            self.summarize_rs_id = self.summarize_rs_id[0]
        self.aux_tok_output_layer_map[self.summarize_rs_id] = "summarize_rs"

        self.paraphrase_rs_id = self.tokenizer.encode_single(self.paraphrase_rs)
        if len(self.paraphrase_rs_id) != 1 and (isinstance(self.paraphrase_rs_id, list)):
            raise Exception(
                f"The number of ids the start token is encoded into should be one, got {self.paraphrase_rs_id}!")
        else:
            self.paraphrase_rs_id = self.paraphrase_rs_id[0]
        self.aux_tok_output_layer_map[self.paraphrase_rs_id] = "paraphrase_rs"

        self.mode_ids = {}
        self.mode_ids[self.lm_tok] = self.lm_tok_id
        self.mode_ids[self.mqa_tok] = self.mqa_tok_id
        self.mode_ids[self.gqa_tok] = self.gqa_tok_id
        self.mode_ids[self.highlighting_rs] = self.highlighting_rs_id
        self.mode_ids[self.summarize_rs] = self.summarize_rs_id
        self.mode_ids[self.paraphrase_rs] = self.paraphrase_rs_id

class V4Wikitext103Medium(object):

    def __init__(self, strategy, batch_size, loss_object, learning_rate,
                 lm_tok="<lm>", mqa_tok="<mqa>", gqa_tok="<gqa>",
                 highlighting_rs="<highlighting_rs>",
                 summarize_rs="<summarize_rs>",
                 paraphrase_rs="<paraphrase_rs>",
                 vocab_filepath="../vocabulary/vocab1.txt"):

        self.lm_tok = lm_tok
        self.mqa_tok = mqa_tok
        self.gqa_tok = gqa_tok
        self.highlighting_rs = highlighting_rs
        self.summarize_rs = summarize_rs
        self.paraphrase_rs = paraphrase_rs

        self.strategy = None
        if strategy is not None:
            if strategy == "MirroredStrategy":
                self.strategy = tf.distribute.MirroredStrategy()
            else:
                assert Exception(f"Invalid strategy")
        self.batch_size = batch_size

        self.loss_object = None
        if loss_object is None:
            # default value for the loss_object.
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
        else:
            self.loss_object = loss_object

        self.learning_rate = None
        if learning_rate is None:
            self.learning_rate = 0.0001
        else:
            self.learning_rate = learning_rate

        self.num_layers_vanilla = 12
        self.num_layers_nm = 12
        self.num_layers_mc = 0
        self.num_layers_output = 3

        self.d_model = 768
        self.num_heads = 12
        self.dff = self.d_model*4

        tok = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
        self.tokenizer = Tokenizer(tok)
        vocab_to_add = None
        with open(vocab_filepath, "r") as f:
            vocab_to_add = json.load(f)
        # print(f"\n\n vocab to add: {vocab_to_add} \n\n")
        self.tokenizer.add_tokens_list(vocab_to_add, update_vocab_size_dec=True)

        self.input_vocab_size = self.tokenizer.get_vocab_size()
        self.output_vocab_size = self.tokenizer.get_vocab_size()

        self.num_aux_toks = 3 # <cls> <dec|enc|mc> <lm|mqa|gqa|highlighting_rs|summarize_rs|paraphrase_rs>
        self.max_seq_len_dec = 768
        self.max_seq_len_nm = self.max_seq_len_dec + self.num_aux_toks
        self.max_position_encoding = self.max_seq_len_nm

        self.mask_strategy = "default"
        self.rate=0.1

        self.parallel_layers = []
        self.output_layers = ["lm"]

        # below can stay the same, it doesn't make a difference if the above two lists
        # (parallel_layers and output_layers) is empty/near empty.
        self.aux_tok_output_layer_map = {} # key:item | id(int):output_layer(str)

        self.lm_tok_id = self.tokenizer.encode_single(self.lm_tok)
        if len(self.lm_tok_id) != 1 and (isinstance(self.lm_tok_id, list)):
            raise Exception(
                f"The number of ids the start token is encoded into should be one, got {self.lm_tok_id}!")
        else:
            self.lm_tok_id = self.lm_tok_id[0]
        self.aux_tok_output_layer_map[self.lm_tok_id] = "lm"

        self.mqa_tok_id = self.tokenizer.encode_single(self.mqa_tok)
        if len(self.mqa_tok_id) != 1 and (isinstance(self.mqa_tok_id, list)):
            raise Exception(
                f"The number of ids the start token is encoded into should be one, got {self.mqa_tok_id}!")
        else:
            self.mqa_tok_id = self.mqa_tok_id[0]
        #self.aux_tok_output_layer_map[self.mqa_tok_id] = "mqa"

        self.gqa_tok_id = self.tokenizer.encode_single(self.gqa_tok)
        if len(self.gqa_tok_id) != 1 and (isinstance(self.gqa_tok_id, list)):
            raise Exception(
                f"The number of ids the start token is encoded into should be one, got {self.gqa_tok_id}!")
        else:
            self.gqa_tok_id = self.gqa_tok_id[0]
        #self.aux_tok_output_layer_map[self.gqa_tok_id] = "gqa"

        self.highlighting_rs_id = self.tokenizer.encode_single(self.highlighting_rs)
        if len(self.highlighting_rs_id) != 1 and (isinstance(self.highlighting_rs_id, list)):
            raise Exception(
                f"The number of ids the start token is encoded into should be one, got {self.highlighting_rs_id}!")
        else:
            self.highlighting_rs_id = self.highlighting_rs_id[0]
        #self.aux_tok_output_layer_map[self.highlighting_rs_id] = "highlighting_rs"

        self.summarize_rs_id = self.tokenizer.encode_single(self.summarize_rs)
        if len(self.summarize_rs_id) != 1 and (isinstance(self.summarize_rs_id, list)):
            raise Exception(
                f"The number of ids the start token is encoded into should be one, got {self.summarize_rs_id}!")
        else:
            self.summarize_rs_id = self.summarize_rs_id[0]
        #self.aux_tok_output_layer_map[self.summarize_rs_id] = "summarize_rs"

        self.paraphrase_rs_id = self.tokenizer.encode_single(self.paraphrase_rs)
        if len(self.paraphrase_rs_id) != 1 and (isinstance(self.paraphrase_rs_id, list)):
            raise Exception(
                f"The number of ids the start token is encoded into should be one, got {self.paraphrase_rs_id}!")
        else:
            self.paraphrase_rs_id = self.paraphrase_rs_id[0]
        #self.aux_tok_output_layer_map[self.paraphrase_rs_id] = "paraphrase_rs"

        self.mode_ids = {}
        self.mode_ids[self.lm_tok] = self.lm_tok_id
        self.mode_ids[self.mqa_tok] = self.mqa_tok_id
        self.mode_ids[self.gqa_tok] = self.gqa_tok_id
        self.mode_ids[self.highlighting_rs] = self.highlighting_rs_id
        self.mode_ids[self.summarize_rs] = self.summarize_rs_id
        self.mode_ids[self.paraphrase_rs] = self.paraphrase_rs_id

