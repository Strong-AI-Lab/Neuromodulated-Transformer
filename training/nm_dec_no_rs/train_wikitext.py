import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"

import sys

sys.path.append("../..")

import tensorflow as tf

print(f"\n\nTensorflow version: {tf.__version__}\n\n")

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np

from training.parent_train import *
from training.nm_dec_no_rs.window_train import *
from models.AttentionMasks import *
from models.NMMultiHeadAttention import NMMultiHeadAttention
from models.FeedForwardNetwork import FeedForwardNetwork
from models.PositionEncoding import get_angles, positional_encoding
from models.NMTransformer import NMTransformer
from text_processing.tokenizer import Tokenizer
from transformers import TransfoXLTokenizer
from load_datasets.language_modelling.load_wikitext import *
from models.Model_Hyperparameters.model_hyperparameters import TestModel


def get_generator(tokenizer, filepath="/large_data/wikitext-103/wiki.valid.tokens", load_strategy="default", load_data=[False, ""],
                  process_strategy="default_tokenize", start_tok="<s>", end_tok="</s>", pad_tok="<pad>",
                  pad=True, shuffle=True, max_seq_len=512, batch_size=4, window_size=None, nm_aux_token=[]):

    wiki_loader = load_wikitext103(filepath=filepath, tokenizer=tokenizer, start_tok=start_tok, end_tok=end_tok,
                                   pad_tok=pad_tok, strategy=load_strategy, load_data=load_data,
                                   max_seq_len=max_seq_len)

    return wiki_loader.get_tf_dataset_generator(process_strategy, shuffle, pad, window_size, nm_aux_token).batch(batch_size)

if __name__ == "__main__":

    config = TestModel(strategy="MirroredStrategy", batch_size=8*4, rate=0.1) #TODO add pad token... and others

    strategy = config.strategy
    # strategy = None
    process_strategy = "sliding_window_article"

    target_vocab_size, nm_vocab_size = config.tokenizer.get_vocab_size_dec(), config.tokenizer.get_vocab_size_dec()

    nm_attn = config.nm_attn
    nm_eol = config.nm_eol

    max_seq_len_dec = config.max_seq_len_dec
    max_seq_len_nm = config.max_seq_len_nm

    rel_pos_emb = config.rel_pos_emb
    max_position_encoding_dec = config.max_position_encoding_dec
    max_position_encoding_nm = config.max_position_encoding_nm

    loss_object = config.loss_object
    learning_rate = config.learning_rate

    # learning_rate = 0.0001
    transformer = None
    optimizer = None

    '''
    num_layers_dec, num_layers_nm, num_layers_gating, d_model, num_heads, dff, max_seq_len_dec, max_seq_len_nm,
                 target_vocab_size, nm_vocab_size, max_position_encoding_dec=10000, max_position_encoding_nm=10000,
                 rate=0.1, nm_attn=False, nm_eol=False, rel_pos_emb=True
    '''

    if strategy is not None:
        with strategy.scope():
            transformer = NMTransformer(config.num_layers_dec, config.num_layers_nm, config.num_layers_gating,
                                        config.d_model, config.num_heads, config.dff, max_seq_len_dec,
                                        max_seq_len_nm, target_vocab_size, nm_vocab_size,
                                        max_position_encoding_dec=max_position_encoding_dec,
                                        max_position_encoding_nm=max_position_encoding_nm,
                                        rate=config.rate, nm_attn=nm_attn, nm_eol=nm_eol,
                                        rel_pos_emb=rel_pos_emb)
            optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999)
    else:
        transformer = NMTransformer(config.num_layers_dec, config.num_layers_nm, config.num_layers_gating,
                                    config.d_model, config.num_heads, config.dff, max_seq_len_dec,
                                    max_seq_len_nm, target_vocab_size, nm_vocab_size,
                                    max_position_encoding_dec=max_position_encoding_dec,
                                    max_position_encoding_nm=max_position_encoding_nm,
                                    rate=config.rate, nm_attn=nm_attn, nm_eol=nm_eol,
                                    rel_pos_emb=rel_pos_emb)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999)

    data_dict = {}
    data_dict["train"] = get_generator(filepath="/large_data/wikitext-103/wiki.train.tokens",
                                       load_data=[True, "/large_data/wikitext-103/processed_data/train_heading_default_strategy.txt"],
                                       batch_size=config.batch_size,
                                       process_strategy=process_strategy,
                                       window_size=max_seq_len_dec,
                                       max_seq_len=max_seq_len_dec,
                                       tokenizer=config.tokenizer)
    if strategy is not None:
        data_dict["train"] = strategy.experimental_distribute_dataset(data_dict["train"])

    data_dict["val"] = get_generator(filepath="/large_data/wikitext-103/wiki.valid.tokens",
                                     load_data=[True, "/large_data/wikitext-103/processed_data/val_heading_default_strategy.txt"],
                                     batch_size=config.batch_size,
                                     process_strategy=process_strategy,
                                     window_size=max_seq_len_dec, shuffle=False,
                                     max_seq_len=max_seq_len_dec,
                                     tokenizer = config.tokenizer)
    if strategy is not None:
        data_dict["val"] = strategy.experimental_distribute_dataset(data_dict["val"])

    train_class = SlidingWindowTrain(transformer, optimizer, config.loss_object, loss_function_window_size, config.tokenizer,
                                     checkpoint_path_recent="../../checkpoints/v3_test/",
                                     checkpoint_path_best="", strategy=strategy, pad_token="<pad>",
                                     recent_to_keep=50, load_recent=False, best_to_keep=5, load_best=False,
                                     window_size_train=max_seq_len_dec, window_size_val=max_seq_len_dec)

    train_class.train_iteration(epoch_start=0, epoch_end=2,
                                save_filepath_train="../../results/v3_test/",
                                save_filepath_val="../../results/v3_test/",
                                data_dict=data_dict, num_aux_tokens=0)


