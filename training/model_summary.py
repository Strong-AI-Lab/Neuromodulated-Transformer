import tensorflow as tf

import sys
sys.path.append("..")

#from training.wikitext_103_nm_tfdec_train import *

from models.NMTransformer import * # includes all other relevant imports in this file.
from models.attention_masks import *
from text_processing.create_tfxl_vocab import * #get_tfxl_tokenizer, get_xlnet_tokenizer
from load_datasets.load_wikitext_103 import *
from models.parent_train import *

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"#"1,3,6,7"

gpus = tf.config.experimental.list_physical_devices('GPU')
counter = 0
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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

    batch_size = 8
    tokenizer = get_tfxl_tokenizer()
    target_vocab_size = len(tokenizer.get_vocab().keys())
    nm_vocab_size = target_vocab_size # they share the same vocab size

    dec_ = tokenizer.encode("<dec>",
                            add_special_tokens=False,
                            pad_to_max_length=False,
                            return_token_type_ids=False,
                            return_attention_mask=False)[0]
    lm_ = tokenizer.encode("<lm>",
                            add_special_tokens=False,
                            pad_to_max_length=False,
                            return_token_type_ids=False,
                            return_attention_mask=False)[0]

    # initialize model hyperparameters.
    nm_tf_dec_dict = nmdec_params(target_vocab_size, nm_vocab_size, num_aux_tokens=2)
    _dict = nm_tf_dec_dict

    # none keeps the loss in an array/tensor
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction='none')
    #learning_rate = CustomTransformerSchedule(nm_tf_dec_dict["d_model"], warmup_steps=10000)

    transformer = NMTransformerDec(_dict["num_layers"], _dict["d_model"], _dict["num_heads"], _dict["dff"],
                                        _dict["ffn_dict_dec"], _dict["max_seq_len_dec"], _dict["target_vocab_size"],
                                        _dict["pe_target"], rate_dec=_dict["rate_dec"],
                                        nm_mha_dec=_dict["nm_mha_dec"],
                                        enc_out=_dict["enc_out"], neuromodulation=_dict["neuromodulation"],
                                        nm_net_vocab_size=_dict["nm_net_vocab_size"], pe_nm_net=_dict["pe_nm_net"],
                                        rate_nm_enc=_dict["rate_nm_enc"], nm_mha_net=_dict["nm_mha_net"],
                                        ffn_dict_nm=_dict["ffn_dict_nm"], max_seq_len_nm=_dict["max_seq_len_nm"])

    # let tensorflow build the method.
    tar = tf.random.uniform((batch_size, _dict["max_seq_len_dec"]), minval=0, maxval=400, dtype=tf.dtypes.int64)
    nm_inp_dec = tf.random.uniform((batch_size, _dict["max_seq_len_nm"]), minval=0, maxval=400, dtype=tf.dtypes.int64)
    training = True
    look_ahead_mask = create_look_ahead_mask(tar.shape[1])
    dec_padding_mask = create_padding_mask(tar, padding_id=5)
    nm_dec_comb_mask = create_combined_mask(nm_inp_dec,
                                            padding_id=5)  # this includes look_ahead and padding_token masking.

    _, _ = transformer(tar, nm_inp_dec, training, look_ahead_mask, dec_padding_mask,
                                       nm_dec_comb_mask, external_memory=False)


    print(transformer.summary())
    print(transformer.nm_encoder.summary())

'''
WARNING:tensorflow:Gradients do not exist for variables ['dense_220/kernel:0', 'dense_220/bias:0', 'dense_221/kernel:0', 
'dense_221/bias:0', 'dense_230/kernel:0', 'dense_230/bias:0', 'dense_231/kernel:0', 'dense_231/bias:0', 'dense_240/kernel:0', 
'dense_240/bias:0', 'dense_241/kernel:0', 'dense_241/bias:0', 'dense_250/kernel:0', 'dense_250/bias:0', 'dense_251/kernel:0', 
'dense_251/bias:0', 'dense_260/kernel:0', 'dense_260/bias:0', 'dense_261/kernel:0', 'dense_261/bias:0', 'dense_270/kernel:0', 
'dense_270/bias:0', 'dense_271/kernel:0', 'dense_271/bias:0', 'dense_280/kernel:0', 'dense_280/bias:0', 'dense_281/kernel:0', 
'dense_281/bias:0', 'dense_290/kernel:0', 'dense_290/bias:0', 'dense_291/kernel:0', 'dense_291/bias:0', 'dense_300/kernel:0', 
'dense_300/bias:0', 'dense_301/kernel:0', 'dense_301/bias:0', 'dense_310/kernel:0', 'dense_310/bias:0', 'dense_311/kernel:0', '
dense_311/bias:0', 'dense_320/kernel:0', 'dense_320/bias:0', 'dense_321/kernel:0', 'dense_321/bias:0', 'dense_326/kernel:0', 
'dense_326/bias:0', 'dense_327/kernel:0', 'dense_327/bias:0', 'dense_330/kernel:0', 'dense_330/bias:0', 'dense_331/kernel:0', 
'dense_331/bias:0', 'nm_transformer_dec/nm_encoder/nm_encoder_layer_11/layer_normalization_59/gamma:0', 
'nm_transformer_dec/nm_encoder/nm_encoder_layer_11/layer_normalization_59/beta:0'] when minimizing the loss.
'''
