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
#tf.config.run_functions_eagerly(True)
import numpy as np

from training.parent_train import *
from training.QA_single_training.train_race_class import *
from models.AttentionMasks import *
from models.NMMultiHeadAttention import NMMultiHeadAttention
from models.FeedForwardNetwork import FeedForwardNetwork
from models.PositionEncoding import get_angles, positional_encoding
from models.NMTransformer import NMTransformer
from text_processing.tokenizer import Tokenizer
from transformers import TransfoXLTokenizer
from load_datasets.MasterDataLoaderTF import * # MasterDataLoaderTF
from models.Model_Hyperparameters.model_hyperparameters import *


def stop_eol_gating(transformer):
    #do stuff here.
    transformer.nm_eol = False
    decoder = transformer.decoder
    for dec_layer in decoder.decoder_layers:
        dec_layer.nm_eol = False
    return transformer

if __name__ == "__main__":

    config = FinalModelSmall(strategy="MirroredStrategy", batch_size=4*GPUS_AVAILABLE, rate=0.1)
    #config = FinalModelSmall(strategy=None, batch_size=4 * GPUS_AVAILABLE, rate=0.1)

    strategy = config.strategy
    # strategy = None

    target_vocab_size, nm_vocab_size = config.tokenizer.get_vocab_size_dec(), config.tokenizer.get_vocab_size_dec()

    #nm_attn = config.nm_attn
    nm_attn = True
    #nm_eol = config.nm_eol #TODO: note the change here.
    nm_eol = False

    max_seq_len_dec = config.max_seq_len_dec
    max_seq_len_nm = config.max_seq_len_nm

    rel_pos_emb = config.rel_pos_emb
    max_position_encoding_dec = config.max_position_encoding_dec
    max_position_encoding_nm = config.max_position_encoding_nm

    loss_object = config.loss_object
    #learning_rate = config.learning_rate
    #learning_rate = 0.00001 # from 0.0001
    #learning_rate = 0.0001
    learning_rate = 0.0001

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
                                        rel_pos_emb=rel_pos_emb, parallel_layers=config.parallel_layers,
                                        stop_grad_strat=1)
            # 0 is stop all but the last layer, 1 is no stopping the gradient at all, 2 is stopping all gradient.
            #optimizer = tf.keras.optimizers.Adam(learning_rate)
            optimizer = tf.keras.optimizers.SGD(learning_rate)
    else:
        transformer = NMTransformer(config.num_layers_dec, config.num_layers_nm, config.num_layers_gating,
                                    config.d_model, config.num_heads, config.dff, max_seq_len_dec,
                                    max_seq_len_nm, target_vocab_size, nm_vocab_size,
                                    max_position_encoding_dec=max_position_encoding_dec,
                                    max_position_encoding_nm=max_position_encoding_nm,
                                    rate=config.rate, nm_attn=nm_attn, nm_eol=nm_eol,
                                    rel_pos_emb=rel_pos_emb, parallel_layers=config.parallel_layers,
                                    stop_grad_strat=1)
        optimizer = tf.keras.optimizers.Adam(learning_rate)


    filepaths = {"RACE_high_train_label": "/large_data/RACE/train/high/",
                 "RACE_high_val_label": "/large_data/RACE/dev/high/",
                 "RACE_middle_train_label": "/large_data/RACE/train/middle/",
                 "RACE_middle_val_label": "/large_data/RACE/dev/middle/"}
    #dloader = MasterDataLoaderTF(filepaths=filepaths, seq_len=config.max_seq_len_dec,
    #                             batch_size=config.batch_size, tokenizer=config.tokenizer,
    #                             enc_tok="<enc>", dec_tok="<dec>", mlm_tok="<mlm>", lm_tok="<lm>",
    #                             cls_tok="<cls>", sep_tok="<sep>", mask_tok="<mask>", pad_tok="<pad>",
    #                             start_tok="<s>", end_tok="</s>", null_tok="<null>",
    #                             num_reading_strategies=config.num_reading_strategies,
    #                             pad_to_max_length=True, strategy="random",
    #                             C4_processed_filepath="/large_data/C4/en/../processed.txt")
    dloader_train = MasterDataLoaderTF(filepaths=filepaths, seq_len=config.max_seq_len_dec, batch_size=config.batch_size, tokenizer=config.tokenizer) # all of the other parameters are equal to the default value, so not included here.
    generator_train = dloader_train.get_generator("RACE_combined_train_label", False).batch(config.batch_size)
    #generator_train = dloader_train.get_generator("RACE_middle_val", False).batch(config.batch_size)


    data_dict = {}
    data_dict["train"] = generator_train
    if strategy is not None:
        data_dict["train"] = strategy.experimental_distribute_dataset(data_dict["train"])

    dloader_val = MasterDataLoaderTF(filepaths=filepaths, seq_len=config.max_seq_len_dec, batch_size=config.batch_size, tokenizer=config.tokenizer)  # all of the other parameters are equal to the default value, so not included here.
    generator_val = dloader_val.get_generator("RACE_combined_val_label", False).batch(config.batch_size)

    data_dict["val"] = generator_val
    if strategy is not None:
        data_dict["val"] = strategy.experimental_distribute_dataset(data_dict["val"])

    #batch_ = 1
    #for (inp_str, inp_id, tar_id, nm_inp_id) in data_dict["train"]:
        #print(f"batch: {batch_}")
        #print(f"inp_str: {inp_str.shape} \t inp_str: {inp_str} \n"
        #    f"inp_id.shape: {inp_id.shape} \t inp_id: {inp_id} \n"
        #      f"tar_id.shape: {tar_id.shape} \t tar_id: {tar_id} \n"
        #      f"nm_inp_id.shape: {nm_inp_id.shape} \t nm_inp_id: {nm_inp_id} \n")
        #if batch_ == 1: break
    #    batch_ += 1
    #print(f"batch_ counter: {batch_}")

    print(f'encoder {config.tokenizer.encode_single("<enc>")[0]}')

    #loss_object = config.loss_object
    #loss_object = tf.keras.losses.MeanSquaredError(reduction="none")
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction="none")
    #loss_object = tf.keras.losses.MeanAbsoluteError(reduction="none")

    train_class = NMTransformerEncDecTrain(transformer, optimizer, loss_object, multi_choice_qa_loss_function, config.tokenizer,
                                     checkpoint_path_recent="/home/kkno604/Documents/Final_model_pretraining/modern_checkpoints/BERT_mode/",
                                     checkpoint_path_best="", strategy=strategy, pad_token="<pad>",
                                     recent_to_keep=10, load_recent=False, best_to_keep=5, load_best=False,
                                     #load_specific_path="/home/kkno604/Documents/Final_model_pretraining/Checkpoints/pretraining_c4_notable/ckpt-440",
                                     load_specific_path="",
                                     enc_tok_id=config.tokenizer.encode_single("<enc>")[0],
                                     dec_tok_id=config.tokenizer.encode_single("<dec>")[0],
                                     end_tok_id=config.tokenizer.encode_single("</s>")[0])

    ## build the model with inputs so eol can be diabled.
    #for (inp_str, inp_id, label, nm_inp_id) in data_dict["train"]:
    #    strategy.run(train_class.model, args=(inp_id, nm_inp_id, False,))
    #    break

    #train_class.model = stop_eol_gating(train_class.model)

    train_class.train_iteration(epoch_start=0, epoch_end=3, iteration_counter=0,
                                save_filepath_train="/home/kkno604/Documents/Final_model_pretraining/modern_results/BERT_mode/",
                                save_filepath_val="/home/kkno604/Documents/Final_model_pretraining/modern_results/BERT_mode/",
                                data_dict=data_dict, num_aux_tokens=config.num_aux_tokens, save_end_epoch=True,
                                print_every_iterations=25, save_every_iterations=1000000, mask_strategy="none",
                                mode="multi_choice") # 1 million b/c it will only save at the end of each epoch...

