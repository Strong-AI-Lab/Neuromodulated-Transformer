import os

import tensorflow.python.framework.ops

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
GPUS_AVAILABLE = 1

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

from training.fine_tuning.parent_fine_tune_train import *
#from training.pre_training.pre_train_class import *
from models.NMTransformer import *
from models.config_model import *
from models.custom_lr_schedules import CosineDecayLW
from load_datasets.MasterDataLoader import *
from load_datasets.question_answering.loadRACE import RACEDataLoader

if __name__ == "__main__":

    #config = V4ConfigMediumSize(strategy="MirroredStrategy", batch_size=8*GPUS_AVAILABLE,
    #                            loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none'),
    #                            learning_rate=0.00001,
    #                            vocab_filepath="/data/kkno604/Neuromodulated-Transformer/vocabulary/vocab1.txt")

    config = V4ConfigMediumSize(strategy="MirroredStrategy", batch_size=16*GPUS_AVAILABLE,
                                loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none'),
                                #learning_rate=tf.keras.optimizers.schedules.CosineDecay(0.0001, decay_steps=1000000),
                                learning_rate=0.00001,
                                #learning_rate=CosineDecayLW(start_lr=0.0001, lower_bound_lr=0.000001, upper_bound_lr=0.001,
                                #                            warmup_steps=2000, decay_steps=3000*20),
                                vocab_filepath="/data/kkno604/Neuromodulated-Transformer/vocabulary/vocab1.txt",
                                gpt2_117=True,
                                tokenizer="gpt2")
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

    filepaths = {"RACE_high_test": "/large_data/RACE/test/high/",
                 "RACE_middle_test": "/large_data/RACE/test/middle/"}
    dloader_test = MasterDataLoaderTF(filepaths=filepaths, seq_len=config.max_seq_len_dec, batch_size=config.batch_size, tokenizer=config.tokenizer)
    #generator = dloader_train.get_generator(type="C4_pretrain_dec", shuffle=False).batch(config.batch_size)
    #generator_test = dloader_test.get_generator("RACE_middle_test", False).batch(config.batch_size)
    generator_test = dloader_test.get_generator("RACE_high_test", False).batch(config.batch_size)

    data_dict = {}
    data_dict["test"] = generator_test
    if strategy is not None:
        data_dict["test"] = strategy.experimental_distribute_dataset(data_dict["test"])

    #train_class = ParentFineTuningNL(transformer, optimizer, config.loss_object, loss_function, config.tokenizer,
    #                                       checkpoint_path_recent="/data/kkno604/NMTransformer_fine_tuning/RACE/Checkpoints2/",
    #                                       strategy=strategy, pad_token="<pad>", recent_to_keep=5, load_recent=False,
    #                                       load_specific_path="/data/kkno604/NMTransformer_fine_tuning/RACE/Checkpoints/no_freeze_label_only/ckpt-204",
    #                                       enc_tok_id=config.tokenizer.encode_single("<enc>")[0],
    #                                       dec_tok_id=config.tokenizer.encode_single("<dec>")[0],
    #                                       output_layer_name="lm", fine_tuning=False)
    #TODO remember "lm" and "mqa" to switch between the two.

    train_class = ParentFineTuningNL(transformer, optimizer, config.loss_object, loss_function, config.tokenizer,
                                     checkpoint_path_recent="/data/kkno604/Specific-fine-tuning/RACE/Checkpoints/no_freeze_gpt2_update_all_pretrain/",
                                     strategy=strategy, pad_token="<pad>", recent_to_keep=10, load_recent=False,
                                     # load_specific_path="/data/kkno604/NMTransformer_pretraining/Checkpoints/pretrain-C4-v4-gpt2/ckpt-48",
                                     #load_specific_path="/data/kkno604/NMTransformer_pretraining/Checkpoints/gpt-2-saved-checkpoints/ckpt-200",
                                     load_specific_path="/data/kkno604/Specific-fine-tuning/RACE/Checkpoints/Final_V4_model_gpt2_nofreeze/ckpt-220",
                                     #load_specific_path="",
                                     enc_tok_id=config.tokenizer.encode_single("<enc>")[0],
                                     dec_tok_id=config.tokenizer.encode_single("<dec>")[0],
                                     output_layer_name="lm", fine_tuning=False,
                                     fixed_output=True, stop_gradient=False,
                                     reading_strat_mc_bool=False, lambda_vanilla_set=0.5, lambda_lm=0.1,
                                     vanilla_set_aux_loss_bool=False,
                                     lm_aux_loss_global=False, train_cutoff=3)

    train_class.generate_answer_test(e=0, save_filepath="/data/kkno604/Specific-fine-tuning/RACE/Results/Final_V4_model_gpt2_nofreeze/",
                                     data=data_dict["test"], num_aux_tokens=config.num_aux_toks,
                                     max_generate_len=1, attn_strat="full_attn", filename_prefix="test_high_final_epoch_20",
                                     test_step_type="label")
