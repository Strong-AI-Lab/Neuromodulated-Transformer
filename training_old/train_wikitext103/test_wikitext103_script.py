import os

import tensorflow.python.framework.ops

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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

from training.parent_train import *
from training.train_wikitext103.train_wikitext103_class import *
from models.NMTransformer import *
from models.config_model import *
from load_datasets.language_modelling.load_wikitext import *

if __name__ == "__main__":

    config = V4ConfigMediumSize(strategy="MirroredStrategy", batch_size=8*GPUS_AVAILABLE,
                                loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none'),
                                learning_rate=tf.keras.optimizers.schedules.CosineDecay(0.001, decay_steps=400000),
                                #learning_rate=0.00005,
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

    dataloader1 = load_wikitext103(filepath="/large_data/wikitext-103/wiki.test.tokens", max_seq_len=config.max_seq_len_dec,
                                  tokenizer=config.tokenizer, start_tok="<s>", end_tok="</s>", pad_tok="<pad>",
                                  pad_to_max_length=True, strategy="gpt2-remove", load_data=[False, ""])
    # strategy="default"

    process_strategy = "sliding_window_article"

    data_dict = {}

    #window_size = config.max_seq_len_dec
    window_size = 32

    # for both the process strategy is sliding_window_article as if you set window size to max_seq_size then it is the same as defaut...
    data_dict["test"] = dataloader1.get_tf_dataset_generator(process_strategy="sliding_window_article", shuffle=True, pad=True,
                                                             sliding_window=window_size,
                                                             nm_aux_tokens=["<dec>", "<lm>",
                                                                          "<null>"]).batch(config.batch_size)

    #for (tar_inp, tar_real, isStart) in data_dict["test"]:
    #    print(tar_inp.shape)

    if strategy is not None:
        data_dict["test"] = strategy.experimental_distribute_dataset(data_dict["test"])



    test_class = SlidingWindowTrain(transformer, optimizer, config.loss_object, loss_function_window_size, config.tokenizer,
                                     checkpoint_path_recent="/data/kkno604/NMTransformer_wikitext103/Checkpoints/",
                                     checkpoint_path_best="", strategy=strategy, pad_token="<pad>",
                                     recent_to_keep=10, load_recent=False, best_to_keep=5, load_best=False,
                                     window_size_train=config.max_seq_len_dec, window_size_val=window_size,
                                     load_specific_path="/data/kkno604/NMTransformer_pretraining/Checkpoints/gpt-2-saved-checkpoints/ckpt-200")
                                     #load_specific_path="/data/kkno604/NMTransformer_pretraining/Checkpoints/pretrain-C4-v4-gpt2/ckpt-111")

    res_dict = test_class.run_no_train(data_dict["test"], config.num_aux_toks)
    print("\n",res_dict,"\n")

    #print(f"\n\n\n{test_class.model.summary()}\n\n\n")

    #print(f"\n\n\n{test_class.model.nm_set.summary()}\n\n\n")

    #print(f"\n\n\n{test_class.model.output_set.summary()}\n\n\n")