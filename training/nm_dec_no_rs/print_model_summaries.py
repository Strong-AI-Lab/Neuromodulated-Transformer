import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices' # TODO test that this works.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" #"0,3,4,5,6,7"

import sys
sys.path.append("../..")

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np

from training.parent_train import *
from models.AttentionMasks import *
from models.NMMultiHeadAttention import NMMultiHeadAttention
from models.FeedForwardNetwork import FeedForwardNetwork
from models.PositionEncoding import get_angles, positional_encoding
from models.NMTransformerDec import NMTransformerDec
from text_processing.tokenizer import Tokenizer
from transformers import TransfoXLTokenizer
from load_datasets.language_modelling.load_wikitext import *

if __name__ == "__main__":

    tok = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
    tokenizer = Tokenizer(tok)
    tokenizer.add_tokens_list(["<s>", "</s>", "<pad>"], update_vocab_size_dec=True)
    tokenizer.add_tokens_list(["<dec>", "<lm>", "<confidence>"], update_vocab_size_dec=True)

    num_layers_dec, num_layers_nm = 12, 12
    d_model, num_heads, dff = 512, 8, 1024
    max_seq_len_dec, max_seq_len_nm = 512, 514
    target_vocab_size, nm_vocab_size = tokenizer.get_vocab_size_dec(), tokenizer.get_vocab_size_dec()
    batch_size = 2  # last number is the number of gpus, first is the batch size per gpu.
    parallel_layers = {}
    parallel_layers["nm_attn_gate"] = "GateLayerAttn"
    parallel_layers["nm_eol_gate"] = "NMEncoderLayerNoRC"

    transformer = NMTransformerDec(num_layers_dec, num_layers_nm, d_model, num_heads, dff, max_seq_len_dec,
                                   max_seq_len_nm, target_vocab_size, nm_vocab_size,
                                   max_position_encoding_dec=2000,
                                   max_position_encoding_nm=2000, rate=0.1, nm_attn=True, nm_eol=True,
                                   parallel_layers=parallel_layers)

    dec_inp = tf.random.uniform((batch_size, max_seq_len_dec), minval=0, maxval=24)
    nm_inp = tf.random.uniform((batch_size, max_seq_len_nm), minval=0, maxval=24)
    output = transformer(dec_inp, nm_inp, True, 0, 0)

    print(transformer.summary())