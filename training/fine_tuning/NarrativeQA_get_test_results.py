# Get val and test results (f1 and em scores) for SQuAD.

import os

import tensorflow.python.framework.ops

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
GPUS_AVAILABLE = 1

import sys
sys.path.append("../..")
import copy

import tensorflow as tf

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

print(f"\nTensorflow version: {tf.__version__}\n")

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np

tf.config.run_functions_eagerly(False)

from training.fine_tuning.fine_tuning_class import * #FineTuningClass
#from training.pre_training.pre_train_class import *
from models.NMTransformer import *
from models.config_model import *
from models.custom_lr_schedules import CosineDecayLW
from load_datasets.MasterDataLoader import *
from load_datasets.question_answering.loadRACE import RACEDataLoader

#tf.keras.backend.clear_session()

if __name__ == "__main__":

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
    #strategy = None

    ### Override default sequence length 1300
    #config.max_seq_len_dec = 1300
    #config.max_seq_len_nm = config.max_seq_len_dec + config.num_aux_toks
    #config.max_position_encoding = config.max_seq_len_nm

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

    filepaths = {"NarrativeQA_test": "/large_data/NarrativeQA/narrativeqa-master/"}
    GETVALRESULTS = False

    data_dict = {}

    dloader_test = MasterDataLoaderTF(filepaths=filepaths, seq_len=config.max_seq_len_dec,
                                       batch_size=config.batch_size, tokenizer=config.tokenizer)
    # get results for the test set...
    if GETVALRESULTS:
        dloader_test.dataLoaders["NarrativeQA_test"].test_data = copy.deepcopy(dloader_test.dataLoaders["NarrativeQA_test"].val_data)
        assert dloader_test.dataLoaders["NarrativeQA_test"].test_data == dloader_test.dataLoaders["NarrativeQA_test"].val_data
    generator_test = dloader_test.get_generator("NarrativeQA_test", False, override_lm=True).batch(config.batch_size)

    data_dict["test"] = generator_test
    if strategy is not None:
        data_dict["test"] = strategy.experimental_distribute_dataset(data_dict["test"])

    for i in [1,2]:

        train_class = FineTuningClass(transformer, optimizer, config.loss_object, loss_function, config.tokenizer,
                                      checkpoint_path_recent="/home/kkno604/Documents/V4 results/Specific-fine-tuning/NarrativeQA/Checkpoints/",
                                      strategy=strategy, pad_token="<pad>", end_tok="</s>",
                                      recent_to_keep=20, load_recent=False,
                                      load_specific_path="/home/kkno604/Documents/V4 results/Specific-fine-tuning/NarrativeQA/Checkpoints/ckpt-"+str(200+i),
                                      enc_tok="<enc>", dec_tok="<dec>",
                                      output_layer_name="lm", fixed_output=True, stop_gradient=False,
                                      reading_strat_mc_bool=False, lambda_vanilla_set=0.5, lambda_lm=0.2,
                                      vanilla_set_aux_loss_bool=False,
                                      lm_aux_loss_global=False, train_cutoff=0)

        train_class.get_test_results(e=0, save_filepath="/home/kkno604/Documents/V4 results/Specific-fine-tuning/NarrativeQA/Results/test/",
                                     data=data_dict["test"], num_aux_tokens=config.num_aux_toks, max_generate_len=100,
                                     filename_prefix="test-epoch-"+str(i), metrics=["rouge-l-score"], mode="GQA", multiple_answers=True)
