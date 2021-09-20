import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
GPUS_AVAILABLE = 3

import sys

sys.path.append("../..")

import tensorflow as tf

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

print(f"\n\nTensorflow version: {tf.__version__}\n\n")

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np

from training.parent_train import *
from training.pre_training.pretrain_train import *
from models.AttentionMasks import *
from models.NMMultiHeadAttention import NMMultiHeadAttention
from models.FeedForwardNetwork import FeedForwardNetwork
from models.PositionEncoding import get_angles, positional_encoding
from models.NMTransformer import NMTransformer
from text_processing.tokenizer import Tokenizer
from transformers import TransfoXLTokenizer
from load_datasets.MasterDataLoaderTF import * # MasterDataLoaderTF
from models.Model_Hyperparameters.model_hyperparameters import *

if __name__ == "__main__":

    config = FinalModelSmall(strategy="MirroredStrategy", batch_size=4*GPUS_AVAILABLE, rate=0.1)
    #config = FinalModelSmall(strategy=None, batch_size=4 * GPUS_AVAILABLE, rate=0.1)

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
                                        rel_pos_emb=rel_pos_emb, parallel_layers=config.parallel_layers)
            optimizer = tf.keras.optimizers.Adam(learning_rate)
    else:
        transformer = NMTransformer(config.num_layers_dec, config.num_layers_nm, config.num_layers_gating,
                                    config.d_model, config.num_heads, config.dff, max_seq_len_dec,
                                    max_seq_len_nm, target_vocab_size, nm_vocab_size,
                                    max_position_encoding_dec=max_position_encoding_dec,
                                    max_position_encoding_nm=max_position_encoding_nm,
                                    rate=config.rate, nm_attn=nm_attn, nm_eol=nm_eol,
                                    rel_pos_emb=rel_pos_emb, parallel_layers=config.parallel_layers)
        optimizer = tf.keras.optimizers.Adam(learning_rate)

    filepaths = {"C4_nm_pre_train": "/large_data/C4/en/"}
    dloader = MasterDataLoaderTF(filepaths=filepaths, seq_len=config.max_seq_len_dec,
                                 batch_size=config.batch_size, tokenizer=config.tokenizer,
                                 enc_tok="<enc>", dec_tok="<dec>", mlm_tok="<mlm>", lm_tok="<lm>",
                                 cls_tok="<cls>", sep_tok="<sep>", mask_tok="<mask>", pad_tok="<pad>",
                                 start_tok="<s>", end_tok="</s>", null_tok="<null>",
                                 num_reading_strategies=config.num_reading_strategies,
                                 pad_to_max_length=True, strategy="random",
                                 C4_processed_filepath="/large_data/C4/en/../processed.txt")
    generator = dloader.get_generator("C4_pretrain_enc_dec", False).batch(config.batch_size)

    data_dict = {}
    data_dict["train"] = generator
    if strategy is not None:
        data_dict["train"] = strategy.experimental_distribute_dataset(data_dict["train"])

    print(f'encoder {config.tokenizer.encode_single("<enc>")[0]}')

    train_class = NMTransformerEncDecTrain(transformer, optimizer, config.loss_object, loss_function_encoder, loss_function_decoder, config.tokenizer,
                                     checkpoint_path_recent="/home/kkno604/Documents/Final_model_pretraining/Checkpoints/pretraining_c4/",
                                     checkpoint_path_best="", strategy=strategy, pad_token="<pad>",
                                     recent_to_keep=5, load_recent=True, best_to_keep=5, load_best=False,
                                     load_specific_path="", enc_tok_id=config.tokenizer.encode_single("<enc>")[0],
                                     dec_tok_id=config.tokenizer.encode_single("<dec>")[0])
    train_class.train_iteration(epoch_start=0, epoch_end=1, iteration_counter=50000,
                                save_filepath_train="/home/kkno604/Documents/Final_model_pretraining/Results/pretraining_c4/",
                                data_dict=data_dict, num_aux_tokens=config.num_aux_tokens, save_end_epoch=True,
                                print_every_iterations=100, save_every_iterations=5000)


