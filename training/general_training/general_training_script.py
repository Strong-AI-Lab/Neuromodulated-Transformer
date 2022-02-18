import os

import tensorflow.python.framework.ops

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

from training.fine_tuning.fine_tuning_class import *
#from training.pre_training.pre_train_class import *
from models.NMTransformer import *
from models.config_model import *
from models.custom_lr_schedules import CosineDecayLW
from load_datasets.MasterDataLoader import *

#tf.keras.backend.clear_session()

if __name__ == "__main__":

    config = V4ConfigMediumSize(strategy="MirroredStrategy", batch_size=2*GPUS_AVAILABLE,
                                loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none'),
                                #learning_rate=tf.keras.optimizers.schedules.CosineDecay(0.0001, decay_steps=1000000),
                                #learning_rate=0.00001,
                                learning_rate=CosineDecayLW(start_lr=0.00005, lower_bound_lr=0.00001,
                                                            upper_bound_lr=0.0001, warmup_steps=2000, decay_steps=400000),
                                vocab_filepath="/data/kkno604/Neuromodulated-Transformer/vocabulary/vocab1.txt",
                                gpt2_117=True,
                                tokenizer="gpt2")
    strategy = config.strategy
    #strategy = None

    cls_tok_id = config.tokenizer.encode_single("<cls>")[0]
    dec_tok_id = config.tokenizer.encode_single("<dec>")[0]

    "lm",
    "mqa",
    "generate_answer",
    "highlighting",
    "summarization",
    "paraphrasing"

    lm_tok_id = config.tokenizer.encode_single("<lm>")[0]
    mqa_tok_id = config.tokenizer.encode_single("<mqa>")[0]
    dec_tok_id = config.tokenizer.encode_single("<gqa>")[0]
    highlighting_rs_tok_id = config.tokenizer.encode_single("<highlighting_rs>")[0]
    summarize_rs_tok_id = config.tokenizer.encode_single("<summarize_rs>")[0]
    paraphrase_rs_tok_id = config.tokenizer.encode_single("<paraphrase_rs>")[0]


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
            #transformer.set_output_layers_equal_to_lm("lm")
            transformer.build_custom(cls_tok_id, dec_tok_id, [lm_tok_id, mqa_tok_id, dec_tok_id, highlighting_rs_tok_id,
                                                       summarize_rs_tok_id, paraphrase_rs_tok_id])
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

    filepaths = {"NarrativeQA_train": "/large_data/NarrativeQA/narrativeqa-master/",
                 "SQuAD_train_default": "/large_data/SQuAD 2.0/train-v2.0.json",
                 "RACE_high_train": "/large_data/RACE/train/high/",
                 "RACE_middle_train": "/large_data/RACE/train/middle/",
                 "BoolQ_train": "/large_data/BoolQ/train.jsonl",
                 "OBQA_train": "/large_data/OpenBookQA-V1-Sep2018/Data/Main/train.jsonl",
                 "ARC_train_easy": "/large_data/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Train.jsonl",
                 "ARC_train_challenge": "/large_data/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Train.jsonl",
                 "MCTest_train": "/large_data/MCTest/"}

    dloader_train = MasterDataLoaderTF(filepaths=filepaths, seq_len=config.max_seq_len_dec,
                                       batch_size=config.batch_size, tokenizer=config.tokenizer)
    # generator_train = dloader_train.get_generator(type="NarrativeQA_train", shuffle=True, override_lm=True).batch(config.batch_size)
    generator_train = dloader_train.general_generator_text_to_text_format(seed_datasets=["RACE", "OBQA", "ARC", "MCTest",
                                                                                   "SQuADv2", "NarrativeQA", "BoolQ"],
                                                                    min_train_size=1400, batch_size=config.batch_size,
                                                                    shuffle=True).batch(config.batch_size)

    data_dict = {}
    data_dict["train"] = generator_train
    if strategy is not None:
        data_dict["train"] = strategy.experimental_distribute_dataset(data_dict["train"])

    train_class = FineTuningClass(transformer, optimizer, config.loss_object, loss_function, config.tokenizer,
                                  checkpoint_path_recent="/home/kkno604/Documents/V4 results/General-fine-tuning/Default/Checkpoints/",
                                  strategy=strategy, pad_token="<pad>", end_tok="</s>",
                                  recent_to_keep=20, load_recent=False,
                                  # load_specific_path="/data/kkno604/NMTransformer_pretraining/Checkpoints/pretrain-C4-v4-gpt2/ckpt-48",
                                  load_specific_path="/data/kkno604/NMTransformer_pretraining/Checkpoints/gpt-2-saved-checkpoints/ckpt-200",
                                  #load_specific_path="",
                                  enc_tok="<enc>", dec_tok="<dec>",
                                  output_layer_name=None, fixed_output=False, stop_gradient=False,
                                  reading_strat_mc_bool=False, lambda_vanilla_set=0.5, lambda_lm=0.2,
                                  vanilla_set_aux_loss_bool=True, train_vanilla_set_only_on_task=False,
                                  lm_aux_loss_global=True, train_cutoff=0)


    train_class.train_iteration_GENERAL(epoch_start=0, epoch_end=1,
                                        save_filepath_train="/home/kkno604/Documents/V4 results/General-fine-tuning/Default/Results/",
                                        data_dict=data_dict,
                                        num_aux_tokens=config.num_aux_toks, print_every_iterations=100,
                                        reset_global_step=True, reset_value=0,
                                        iteration=0, save_every_iteration=5000)

