import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices' # TODO test that this works.
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7" #"0,3,4,5,6,7"

import sys
sys.path.append("../..")

import tensorflow as tf
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
from models.NMTransformerDec import NMTransformerDec
from text_processing.tokenizer import Tokenizer
from transformers import TransfoXLTokenizer
from load_datasets.language_modelling.load_wikitext import *

def get_generator(filepath="/large_data/wikitext-103/wiki.valid.tokens", load_strategy="default", load_data=[False, ""],
                  process_strategy="default_tokenize", start_tok="<s>", end_tok="</s>", pad_tok="<pad>",
                  pad=True, shuffle=True, max_seq_len=512, batch_size=4, window_size=None):

    wiki_loader = load_wikitext103(filepath=filepath, tokenizer=tokenizer, start_tok=start_tok, end_tok=end_tok,
                                   pad_tok=pad_tok, strategy=load_strategy, load_data=load_data, max_seq_len=max_seq_len)

    return wiki_loader.get_tf_dataset_generator(process_strategy, shuffle, pad, window_size).batch(batch_size)


if __name__ == "__main__":

    strategy = tf.distribute.MirroredStrategy()
    #strategy = None

    tok = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
    tokenizer = Tokenizer(tok)
    tokenizer.add_tokens_list(["<s>", "</s>", "<pad>"], update_vocab_size_dec=True)
    tokenizer.add_tokens_list(["<dec>", "<lm>", "<confidence>"], update_vocab_size_dec=True)

    num_layers_dec, num_layers_nm = 8, 8
    d_model, num_heads, dff = 768,12, 768*2
    max_seq_len_dec, max_seq_len_nm = 512, 512
    target_vocab_size, nm_vocab_size = tokenizer.get_vocab_size_dec(), tokenizer.get_vocab_size_dec()
    batch_size = 6*4 # last number is the number of gpus, first is the batch size per gpu.
    parallel_layers = {}
    parallel_layers["nm_attn_gate"] = "GateLayerAttn"
    parallel_layers["nm_eol_gate"] = "NMEncoderLayerNoRC"

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, # no softmax is applied to the output before hand.
                                                                reduction='none')

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.00001, decay_steps=1000,
                                                                   decay_rate=0.95, staircase=False, name=None)
    #learning_rate = 0.0001
    transformer = None
    optimizer = None

    if strategy is not None:
        with strategy.scope():
            transformer = NMTransformerDec(num_layers_dec, num_layers_nm, d_model, num_heads, dff, max_seq_len_dec,
                                           max_seq_len_nm, target_vocab_size, nm_vocab_size, max_position_encoding_dec=2000,
                                           max_position_encoding_nm=2000, rate=0.1, nm_attn=True, nm_eol=True,
                                           parallel_layers=parallel_layers)
            optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999)
            #optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.1)
    else:
        transformer = NMTransformerDec(num_layers_dec, num_layers_nm, d_model, num_heads, dff, max_seq_len_dec,
                                       max_seq_len_nm, target_vocab_size, nm_vocab_size,
                                       max_position_encoding_dec=2000,
                                       max_position_encoding_nm=2000, rate=0.1, nm_attn=True, nm_eol=True,
                                       parallel_layers=parallel_layers)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999)

    data_dict = {}
    data_dict["train"] = get_generator(filepath="/large_data/wikitext-103/wiki.train.tokens",
                                       load_data=[True, "/large_data/wikitext-103/processed_data/train_heading_default_strategy.txt"],
                                       batch_size=batch_size,
                                       process_strategy="sliding_window_article",
                                       window_size=512)
    if strategy is not None:
        data_dict["train"] = strategy.experimental_distribute_dataset(data_dict["train"])

    data_dict["val"] = get_generator(filepath="/large_data/wikitext-103/wiki.valid.tokens",
                                     load_data=[True, "/large_data/wikitext-103/processed_data/val_heading_default_strategy.txt"],
                                     batch_size=batch_size,
                                     process_strategy="sliding_window_article",
                                     window_size=32, shuffle=False)
    if strategy is not None:
        data_dict["val"] = strategy.experimental_distribute_dataset(data_dict["val"])

    train_class = SlidingWindowTrain(transformer, optimizer, loss_object, loss_function_window_size, tokenizer,
                 checkpoint_path_recent="../../checkpoints", checkpoint_path_best="", strategy=strategy, pad_token="<pad>",
                 recent_to_keep=50, load_recent=True, best_to_keep=5, load_best=False, window_size_train=32, window_size_val=32)
    train_class.train_iteration(epoch_start=0, epoch_end=1, save_filepath_train="../../results/",
                       save_filepath_val="../../results/", data_dict=data_dict, num_aux_tokens=0)


'''
Documenting results here.

1: layernormalization at the beginning. 0.1 dropout worked but reached peak at 3.8 (loss) and started overfitting. (and essentially a sliding window size of 512 - not good for results.)
loss went down to ~2.2 in the training set. - not only done for 1 epoch.
2: above but with dropout = 0.3 - see if val loss can keep up. definitely worse, loss shot back up to 11...
The first experiment except a different learning rate schedule (hopefully stop/slow down overfitting). 3.24 loss (~25 perplexity) 3.1396 Val Perplexity 23.0936

sliding window if size 32 in val (not train) with same parameters as 2. Diverged...
Lowered decay_rate (.96->.95) and rate_steps (1500->1000). --- still diverged- was slightly better. possibly a better learning rate schedule or cap gradients. 

changed to 768 hidden dimension and 12 heads. good initial results. got down to 16.... perplexity, then went up again. (1 epoch)
Run again on 2nd epoch but with adjusted learning rate (i.e. very small). 0.00001 learning rate now with exponential decay. 
'''

