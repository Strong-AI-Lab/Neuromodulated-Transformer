import os

import tensorflow.python.framework.ops

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
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
#from training.pre_training.pre_train_class import *
from models.NMTransformer import *
from models.GPT_baseline_model import *
from models.config_model import *
from models.custom_lr_schedules import CosineDecayLW
from load_datasets.MasterDataLoader import *
from load_datasets.question_answering.NarrativeQA import NarrativeQADataLoader

#tf.keras.backend.clear_session()

if __name__ == "__main__":

    config = V4ConfigMediumSize(strategy="MirroredStrategy", batch_size=1*GPUS_AVAILABLE,
                                loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none'),
                                #learning_rate=tf.keras.optimizers.schedules.CosineDecay(0.0001, decay_steps=1000000),
                                #learning_rate=0.00001,
                                learning_rate=CosineDecayLW(start_lr=0.00005, lower_bound_lr=0.00001,
                                                            upper_bound_lr=0.0001, warmup_steps=2000, decay_steps=(4000*4)*20),
                                vocab_filepath="/data/kkno604/Neuromodulated-Transformer/vocabulary/vocab1.txt",
                                gpt2_117=True,
                                tokenizer="gpt2")
    strategy = config.strategy
    #strategy = None

    ### Override default sequence length 1300
    config.max_seq_len_dec = 1300
    config.max_seq_len_nm = config.max_seq_len_dec + config.num_aux_toks
    config.max_position_encoding = config.max_seq_len_nm

    transformer, optimizer = None, None
    if strategy is not None:
        with strategy.scope():
            transformer = GPT2Class(d_model=1024, input_vocab_size=config.input_vocab_size,
                                    output_vocab_size=config.output_vocab_size, max_seq_len_dec=config.max_seq_len_dec,
                                    num_aux_toks=3, gpt_pretrained_model="gpt2-medium")
            optimizer = tf.keras.optimizers.Adam(config.learning_rate)
    else:
        transformer = GPT2Class(d_model=1024, input_vocab_size=config.input_vocab_size,
                                output_vocab_size=config.output_vocab_size, max_seq_len_dec=config.max_seq_len_dec,
                                num_aux_toks=3, gpt_pretrained_model="gpt2-medium")
        optimizer = tf.keras.optimizers.Adam(config.learning_rate)

    filepaths = {"NarrativeQA_train": "/large_data/NarrativeQA/narrativeqa-master/",
                 "NarrativeQA_val": "/large_data/NarrativeQA/narrativeqa-master/",
                 "NarrativeQA_test": "/large_data/NarrativeQA/narrativeqa-master/"}

    # override_lm doesn't matter here as the tokens aren't used.
    dloader_train = MasterDataLoaderTF(filepaths=filepaths, seq_len=config.max_seq_len_dec,
                                       batch_size=config.batch_size, tokenizer=config.tokenizer)
    generator_train = dloader_train.get_generator(type="NarrativeQA_train", shuffle=True, override_lm=True).batch(config.batch_size)

    data_dict = {}
    data_dict["train"] = generator_train
    if strategy is not None:
        data_dict["train"] = strategy.experimental_distribute_dataset(data_dict["train"])

    dloader_val = MasterDataLoaderTF(filepaths=filepaths, seq_len=config.max_seq_len_dec,
                                       batch_size=config.batch_size, tokenizer=config.tokenizer)
    generator_val = dloader_val.get_generator(type="NarrativeQA_val", shuffle=False, override_lm=True).batch(config.batch_size)

    data_dict["val"] = generator_val
    if strategy is not None:
        data_dict["val"] = strategy.experimental_distribute_dataset(data_dict["val"])

    train_class = FineTuningClass(transformer, optimizer, config.loss_object, loss_function, config.tokenizer,
                                  checkpoint_path_recent="/data/kkno604/Specific-fine-tuning-baseline/NarrativeQA/Checkpoints/",
                                  strategy=strategy, pad_token="<pad>", end_tok="</s>",
                                  recent_to_keep=20, load_recent=False,
                                  load_specific_path="",
                                  enc_tok="<enc>", dec_tok="<dec>",
                                  output_layer_name=None, fixed_output=False, stop_gradient=False,
                                  reading_strat_mc_bool=False, lambda_vanilla_set=0.5, lambda_lm=0.2,
                                  vanilla_set_aux_loss_bool=False,
                                  lm_aux_loss_global=True, train_cutoff=0,
                                  train_vanilla_set_only_on_task=False, gpt_baseline=True)

    train_class.train_batch_GQA(epoch_start=0, epoch_end=20,
                                save_filepath_train="/data/kkno604/Specific-fine-tuning-baseline/NarrativeQA/Results/",
                                save_filepath_val="/data/kkno604/Specific-fine-tuning-baseline/NarrativeQA/Results/",
                                data_dict=data_dict, num_aux_tokens=config.num_aux_toks,
                                save_end_epoch=False, print_every_iterations=100, reset_global_step=False,
                                reset_value=0)