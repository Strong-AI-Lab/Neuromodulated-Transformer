import os

import tensorflow.python.framework.ops

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
GPUS_AVAILABLE = 4

import sys
sys.path.append("../..")

import tensorflow as tf

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

print(f"\nTensorflow version: {tf.__version__}\n")

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np

tf.config.run_functions_eagerly(False)

from training.parent_train import *
from training.pre_training.pre_train_class import *
from models.NMTransformer import *
from models.config_model import *
from load_datasets.MasterDataLoader import *

if __name__ == "__main__":

    config = V4ConfigMediumSize(strategy="MirroredStrategy", batch_size=8*GPUS_AVAILABLE,
                                loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none'),
                                #learning_rate=tf.keras.optimizers.schedules.CosineDecay(0.0001, decay_steps=1000000),
                                learning_rate=0.00005,
                                vocab_filepath="/data/kkno604/Neuromodulated-Transformer/vocabulary/vocab1.txt")
    strategy = config.strategy

    transformer, optimizer = None, None
    if strategy is not None:
        with strategy.scope():
            transformer = NMTransformer(num_layers_vanilla=config.num_layers_vanilla, num_layers_nm=config.num_layers_nm,
                                        num_layers_mc=config.num_layers_mc, num_layers_output=config.num_layers_output,
                                        d_model=config.d_model, num_heads=config.num_heads, dff=config.dff,
                                        input_vocab_size=config.input_vocab_size, output_vocab_size=config.output_vocab_size,
                                        max_position_encoding=config.max_position_encoding,
                                        max_seq_len_dec=config.max_seq_len_dec, num_aux_toks=config.num_aux_toks,
                                        mask_strategy=config.mask_strategy, rate=config.rate,
                                        parallel_layers=config.parallel_layers, output_layers=config.output_layers,
                                        aux_tok_output_layer_map=config.aux_tok_output_layer_map, mode_ids=config.mode_ids)
            optimizer = tf.keras.optimizers.Adam(config.learning_rate)
    else:
        transformer = NMTransformer(num_layers_vanilla=config.num_layers_vanilla, num_layers_nm=config.num_layers_nm,
                                    num_layers_mc=config.num_layers_mc, num_layers_output=config.num_layers_output,
                                    d_model=config.d_model, num_heads=config.num_heads, dff=config.dff,
                                    input_vocab_size=config.input_vocab_size,
                                    output_vocab_size=config.output_vocab_size,
                                    max_position_encoding=config.max_position_encoding,
                                    max_seq_len_dec=config.max_seq_len_dec, num_aux_toks=config.num_aux_toks,
                                    mask_strategy=config.mask_strategy, rate=config.rate,
                                    parallel_layers=config.parallel_layers, output_layers=config.output_layers,
                                    aux_tok_output_layer_map=config.aux_tok_output_layer_map, mode_ids=config.mode_ids)
        optimizer = tf.keras.optimizers.Adam(config.learning_rate)

    filepaths = {"C4_nm_pre_train": "/large_data/C4/en/"}
    dloader = MasterDataLoaderTF(filepaths=filepaths, seq_len=config.max_seq_len_dec, batch_size=config.batch_size, tokenizer=config.tokenizer)
    generator = dloader.get_generator(type="C4_pretrain_dec", shuffle=False).batch(config.batch_size)

    data_dict = {}
    data_dict["train"] = generator
    if strategy is not None:
        data_dict["train"] = strategy.experimental_distribute_dataset(data_dict["train"])

    train_class = NMTransformerPreTrainDec(transformer, optimizer, config.loss_object, loss_function, config.tokenizer,
                                           checkpoint_path_recent="/data/kkno604/NMTransformer_pretraining/Checkpoints/pretrain_C4/",
                                           checkpoint_path_best="", strategy=strategy, pad_token="<pad>",
                                           recent_to_keep=5, load_recent=True, best_to_keep=5, load_best=False,
                                           load_specific_path="",
                                           enc_tok_id=config.tokenizer.encode_single("<enc>")[0],
                                           dec_tok_id=config.tokenizer.encode_single("<dec>")[0])
    train_class.train_iteration(epoch_start=0, epoch_end=1, iteration_counter=110000, # 913565
                                save_filepath_train="/data/kkno604/NMTransformer_pretraining/Results/pretrain_C4/gelu",
                                data_dict=data_dict, num_aux_tokens=config.num_aux_toks, save_end_epoch=True,
                                print_every_iterations=100, save_every_iterations=5000) 
