#import os
# NOTE: below should always be forst.
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"#"1,3,6,7"

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import sys
sys.path.append("..")

#from training.wikitext_103_nm_tfdec_train import *

from models.NMTransformer import * # includes all other relevant imports in this file.
from models.attention_masks import *
from text_processing.create_tfxl_vocab import * #get_tfxl_tokenizer, get_xlnet_tokenizer
from load_datasets.load_wikitext_103 import *
from models.parent_train import *

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

BUFFERSIZE = 4000

#if len(gpus) >= 1:
#    tf.config.experimental.set_virtual_device_configuration(gpus[1,3,6,7], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1000*47)])

def process_data(data, batch_size):
    return (
        data
        .cache()
        .shuffle(BUFFERSIZE)
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

'''
# TODO test below instead and see if this is better.
gpus = tf.config.list_logical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[:4])
        tf.config.experimental.set_memory_growth(gpus[:4], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e) 
'''

def nmdec_params(target_vocab_size, nm_vocab_size, num_aux_tokens):
    nm_tf_dec_dict = dict()
    nm_tf_dec_dict["num_layers"] = 12
    nm_tf_dec_dict["d_model"] = 512
    nm_tf_dec_dict["num_heads"] = 8
    nm_tf_dec_dict["dff"] = 2048

    names = ["default"]
    ffn1 = [2, [nm_tf_dec_dict["dff"], 'relu', False],
            [nm_tf_dec_dict["d_model"], 'none', False]]
    nm_tf_dec_dict["ffn_dict_dec"] = create_ffn_dict(names, ffn1)

    nm_tf_dec_dict["max_seq_len_dec"] = 511 # note: because in wikitext code we set it to 512, but removed the last eos token.
    nm_tf_dec_dict["target_vocab_size"] = target_vocab_size
    nm_tf_dec_dict["pe_target"] = 1000
    nm_tf_dec_dict["rate_dec"] = 0.1
    nm_tf_dec_dict["nm_mha_dec"] = True
    nm_tf_dec_dict["enc_out"] = False
    nm_tf_dec_dict["neuromodulation"] = True
    nm_tf_dec_dict["nm_net_vocab_size"] = nm_vocab_size
    nm_tf_dec_dict["pe_nm_net"] = 1000
    nm_tf_dec_dict["rate_nm_enc"] = 0.1
    nm_tf_dec_dict["nm_mha_net"] = False
    nm_tf_dec_dict["max_seq_len_nm"] = nm_tf_dec_dict["max_seq_len_dec"] + num_aux_tokens

    names = ["default", "attention_nm", "start_layer_nm"]
    ffn1 = [2, [nm_tf_dec_dict["dff"], 'relu', False],
            [nm_tf_dec_dict["d_model"], 'none', False]] # (batch, max_seq_len_nm, d_model)
    ffn2 = [2, [nm_tf_dec_dict["dff"], 'relu', False],
            [nm_tf_dec_dict["max_seq_len_nm"], 'none', False]] # (batch, max_seq_len_nm, max_seq_len_nm) # in decoder index excluding the auxiliary tokens for both nm_seq_len dimensions.
    ffn3 = [2, [nm_tf_dec_dict["dff"], 'relu', False],
            [nm_tf_dec_dict["d_model"], 'none', False]] # (batch, max_seq_len_nm, d_model) # in decoder index index excluding the auxiliary tokens in the single max_seq_len_nm dimension.
    nm_tf_dec_dict["ffn_dict_nm"] = create_ffn_dict(names, ffn1, ffn2, ffn3)

    return nm_tf_dec_dict

if __name__ == "__main__":

    print(f"The current tensorflow version is: {tf.__version__}")

    #strategy = tf.distribute.MirroredStrategy()
    strategy = None
    batch_size = 8

    # initialize tokenizer
    tokenizer = get_tfxl_tokenizer()
    target_vocab_size = len(tokenizer.get_vocab().keys())
    nm_vocab_size = target_vocab_size # they share the same vocab size

    dec_ = tokenizer.encode("<dec>",
                            add_special_tokens=True,
                            pad_to_max_length=False,
                            return_token_type_ids=False,
                            return_attention_mask=False)[0]
    #print(f"dec_: {dec_}")
    lm_ = tokenizer.encode("<lm>",
                            add_special_tokens=True,
                            pad_to_max_length=False,
                            return_token_type_ids=False,
                            return_attention_mask=False)[0]
    #print(f"lm_ {lm_}")

    # initialize model hyperparameters.
    nm_tf_dec_dict = nmdec_params(target_vocab_size, nm_vocab_size, num_aux_tokens=2)
    _dict = nm_tf_dec_dict

    # none keeps the loss in an array/tensor
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction='none')
    #learning_rate = CustomTransformerSchedule(nm_tf_dec_dict["d_model"], warmup_steps=10000)
    transformer = None
    optimizer = None
    if strategy is not None:
        with strategy.scope():
            transformer = NMTransformerDec(_dict["num_layers"], _dict["d_model"], _dict["num_heads"], _dict["dff"],
                                                _dict["ffn_dict_dec"], _dict["max_seq_len_dec"], _dict["target_vocab_size"],
                                                _dict["pe_target"], rate_dec=_dict["rate_dec"],
                                                nm_mha_dec=_dict["nm_mha_dec"],
                                                enc_out=_dict["enc_out"], neuromodulation=_dict["neuromodulation"],
                                                nm_net_vocab_size=_dict["nm_net_vocab_size"], pe_nm_net=_dict["pe_nm_net"],
                                                rate_nm_enc=_dict["rate_nm_enc"], nm_mha_net=_dict["nm_mha_net"],
                                                ffn_dict_nm=_dict["ffn_dict_nm"], max_seq_len_nm=_dict["max_seq_len_nm"])
            optimizer = tf.keras.optimizers.Adam(0.0005, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    else:
        #with tf.device('/gpu:0'):
        transformer = NMTransformerDec(_dict["num_layers"], _dict["d_model"], _dict["num_heads"], _dict["dff"],
                                       _dict["ffn_dict_dec"], _dict["max_seq_len_dec"], _dict["target_vocab_size"],
                                       _dict["pe_target"], rate_dec=_dict["rate_dec"],
                                       nm_mha_dec=_dict["nm_mha_dec"],
                                       enc_out=_dict["enc_out"], neuromodulation=_dict["neuromodulation"],
                                       nm_net_vocab_size=_dict["nm_net_vocab_size"], pe_nm_net=_dict["pe_nm_net"],
                                       rate_nm_enc=_dict["rate_nm_enc"], nm_mha_net=_dict["nm_mha_net"],
                                       ffn_dict_nm=_dict["ffn_dict_nm"], max_seq_len_nm=_dict["max_seq_len_nm"])
        optimizer = tf.keras.optimizers.Adam(0.0005, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


    checkpoint_path = "/data/kkno604/Neuromodulated-Transformer-with-External-Memory/checkpoints/test_nmdec"

    data_dict = {}

    # get train data loader
    filepath = "/large_data/wikitext-103/wiki.test.tokens"
    max_seq_len, pad_to_max_length = nm_tf_dec_dict["max_seq_len_dec"], True

    load_data_test = [True, "/large_data/wikitext-103/greedy_tfxltok_512seqlen_test.txt"]
    load_data_train = [True, "/large_data/wikitext-103/greedy_tfxltok_512seqlen_train.txt"]
    load_data_val = [True, "/large_data/wikitext-103/greedy_tfxltok_512seqlen_val.txt"]

    tfwiki_val = load_Wikitext_tf(filepath, max_seq_len, tokenizer, pad_to_max_length, load_data=load_data_val)
    tar_inp, tar_real, nm_inp = tfwiki_val.get_data_nm(dec_, lm_)
    data_val = tf.data.Dataset.from_tensor_slices((tar_inp, tar_real, nm_inp))#.batch(batch_size)
    data_val = process_data(data_val, batch_size)
    if strategy is not None:
        data_val = strategy.experimental_distribute_dataset(data_val)
    print(f"VAlidation data test: \n {data_val}")
    #data_dict["val"] = data_val
    data_dict["train"] = data_val

    #tfwiki_train = load_Wikitext_tf(filepath, max_seq_len, tokenizer, pad_to_max_length, load_data=load_data_train)
    #tar_inp, tar_real, nm_inp = tfwiki_train.get_data_nm(dec_, lm_)
    #data_train = tf.data.Dataset.from_tensor_slices((tar_inp, tar_real, nm_inp)).batch(batch_size)
    #if strategy is not None:
    #    data_train = strategy.experimental_distribute_dataset(data_train)
    #data_dict["train"] = data_train

    #tfwiki_test = load_Wikitext_tf(filepath, max_seq_len, tokenizer, pad_to_max_length, load_data=load_data_test)
    #tar_inp, tar_real, nm_inp = tfwiki_test.get_data_nm(dec_, lm_)
    #data_test = tf.data.Dataset.from_tensor_slices((tar_inp, tar_real, nm_inp)).batch(batch_size)
    #if strategy is not None:
    #   data_test = strategy.experimental_distribute_dataset(data_test)
    #data_dict["test"] = data_test


    # loss_function=loss_function passes a reference to the function, loss_function.
    #with tf.device("/gpu:0"):
    transformer_train_class = ParentTrainNL(model=transformer, optimizer=optimizer, loss_object=loss_object,
                                   loss_function=loss_function_sequence_split, tokenizer=tokenizer, checkpoint_path=checkpoint_path,
                                   strategy=strategy, pad_token="<pad>")

    #print("\n REACH \n")

    #transformer_train_class.train_
    transformer_train_class.train_(epoch_start=0, epoch_end=2, save_filepath_train= "../results/test_nmdec/",
                         save_filepath_val="../results/test_nmdec/", data_dict=data_dict)

