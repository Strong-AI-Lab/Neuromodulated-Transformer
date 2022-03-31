import os

import tensorflow.python.framework.ops

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "4,6"
GPUS_AVAILABLE = 2

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

from training.fine_tuning.fine_tuning_class import *
from models.NMTransformer import *
from models.config_model import *
from models.custom_lr_schedules import CosineDecayLW
from load_datasets.MasterDataLoader import *
from load_datasets.question_answering.loadRACE import RACEDataLoader

#tf.keras.backend.clear_session()

if __name__ == "__main__":

    #config = V4ConfigMediumSize(strategy="MirroredStrategy", batch_size=8*GPUS_AVAILABLE,
    #                            loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none'),
    #                            learning_rate=0.00001,
    #                            vocab_filepath="/data/kkno604/Neuromodulated-Transformer/vocabulary/vocab1.txt")

    config = V4ConfigMediumSize(strategy="MirroredStrategy", batch_size=8*GPUS_AVAILABLE,
                                loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none'),
                                #learning_rate=tf.keras.optimizers.schedules.CosineDecay(0.0001, decay_steps=1000000),
                                learning_rate=0.00001,
                                #learning_rate=CosineDecayLW(start_lr=0.0001, lower_bound_lr=0.00001, upper_bound_lr=0.005,
                                #                            warmup_steps=200, decay_steps=607*20),
                                vocab_filepath="/data/kkno604/Neuromodulated-Transformer/vocabulary/vocab1.txt",
                                gpt2_117=True,
                                tokenizer="gpt2")
    strategy = config.strategy 
    #strategy = None

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
                                        aux_tok_output_layer_map=config.aux_tok_output_layer_map, mode_ids=config.mode_ids,
                                        gpt2_117=config.gpt2_117)
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
                                    aux_tok_output_layer_map=config.aux_tok_output_layer_map, mode_ids=config.mode_ids,
                                    gpt2_117=config.gpt2_117)
        optimizer = tf.keras.optimizers.Adam(config.learning_rate)
    #print("\n\n\nbefore training:", optimizer.iterations.numpy(), "\n\n\n")
    filepaths = {"MCTest_train": "/large_data/MCTest/",
                 "MCTest_val": "/large_data/MCTest/"}
    dloader_train = MasterDataLoaderTF(filepaths=filepaths, seq_len=config.max_seq_len_dec, batch_size=config.batch_size, tokenizer=config.tokenizer)
    #generator = dloader_train.get_generator(type="C4_pretrain_dec", shuffle=False).batch(config.batch_size)
    #generator_train = dloader_train.get_generator("RACE_combined_train", False).batch(config.batch_size)
    generator_train = dloader_train.get_generator("MCTest_train_label", True, override_lm=True).batch(config.batch_size)

    data_dict = {}
    data_dict["train"] = generator_train
    if strategy is not None:
        data_dict["train"] = strategy.experimental_distribute_dataset(data_dict["train"])

    dloader_val = MasterDataLoaderTF(filepaths=filepaths, seq_len=config.max_seq_len_dec, batch_size=config.batch_size,
                                 tokenizer=config.tokenizer)

    #"CQA_train_label" or type == "CQA_val_label"

    #generator_val = dloader_val.get_generator("RACE_combined_val", False).batch(config.batch_size)
    generator_val = dloader_val.get_generator("MCTest_val_label", False, override_lm=True).batch(config.batch_size)

    data_dict["val"] = generator_val
    if strategy is not None:
        data_dict["val"] = strategy.experimental_distribute_dataset(data_dict["val"])

    train_class = FineTuningClass(transformer, optimizer, config.loss_object, loss_function, config.tokenizer,
                                  checkpoint_path_recent="/home/kkno604/Documents/V4 results/Specific-fine-tuning/MCTest/Checkpoints/",
                                  strategy=strategy, pad_token="<pad>", end_tok = "</s>",
                                  recent_to_keep=20, load_recent=False,
                                  #load_specific_path="/data/kkno604/NMTransformer_pretraining/Checkpoints/pretrain-C4-v4-gpt2/ckpt-48",
                                  load_specific_path="/data/kkno604/NMTransformer_pretraining/Checkpoints/gpt-2-saved-checkpoints/ckpt-200",
                                  #load_specific_path="",
                                  enc_tok="<enc>", dec_tok="<dec>",
                                  output_layer_name="lm", fixed_output=True, stop_gradient=False,
                                  reading_strat_mc_bool=False, lambda_vanilla_set=0.5, lambda_lm=0.2,
                                  vanilla_set_aux_loss_bool=True,
                                  lm_aux_loss_global=True, train_cutoff=0,
                                  train_vanilla_set_only_on_task=False)

    train_class.train_batch_MQA(epoch_start=0, epoch_end=20,
                                save_filepath_train="/home/kkno604/Documents/V4 results/Specific-fine-tuning/MCTest/Results/",
                                save_filepath_val="/home/kkno604/Documents/V4 results/Specific-fine-tuning/MCTest/Results/",
                                data_dict=data_dict, num_aux_tokens=config.num_aux_toks, save_end_epoch=True,
                                print_every_iterations=100, reset_global_step=True, reset_value=0)
