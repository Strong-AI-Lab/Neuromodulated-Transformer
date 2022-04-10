import os

import tensorflow.python.framework.ops

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
from load_datasets.language_modelling.WikiText import *

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
    filepath = "/large_data/wikitext-103/wiki.test.tokens"
    #filepath = "/large_data/wikitext-2/wiki.test.tokens"
    sliding_window = config.max_seq_len_dec
    #sliding_window = 32

    config = V4ConfigMediumSize(strategy=None, batch_size=2, loss_object=None, learning_rate=None, gpt2_117=True,
                                tokenizer="gpt2",
                                vocab_filepath="/data/kkno604/Neuromodulated-Transformer/vocabulary/vocab1.txt")

    enc_tok = "<enc>"
    dec_tok = "<dec>"
    mlm_tok = "<mlm>"
    lm_tok = "<lm>"
    cls_tok = "<cls>"
    sep_tok = "<sep>"
    mask_tok = "<mask>"
    pad_tok = "<pad>"
    start_tok = "<s>"
    end_tok = "</s>"
    null_tok = "<null>"
    mqa = "<mqa>"
    pmqa = "<pmqa>"
    bmqa = "<bmqa>"
    peentailment = "<peentailment>"
    pbentailment = "<pbentailment>"
    pcoreference = "<pcoreference>"
    bcoreference = "<bcoreference>"
    psentiment = "<psentiment>"
    pgqa = "<pgqa>"
    psqa = "<psqa>"
    gqa = "<gqa>"
    pbqa = "<pbqa>"
    placeholder = "<placeholder>"
    translation = "<translation>"
    a1 = "(1)"
    a2 = "(2)"
    a3 = "(3)"
    a4 = "(4)"
    a5 = "(5)"
    a6 = "(6)"
    a7 = "(7)"
    a8 = "(8)"
    a9 = "(9)"
    passage = "<passage>"
    p1 = "<p1>"
    p2 = "<p2>"
    p3 = "<p3>"
    p4 = "<p4>"
    p5 = "<p5>"
    p6 = "<p6>"
    p7 = "<p7>"
    p8 = "<p8>"
    p9 = "<p9>"
    hypothesis = "<h>"
    question = "<q>"
    metacognition = "<mc>"
    unk_rs = "<unk_rs>"
    aoint_rs = "<aoint_rs>"
    highlighting_rs = "<highlighting_rs>"
    reread_rs = "<reread_rs>"
    summarize_rs = "<summarize_rs>"
    paraphrase_rs = "<paraphrase_rs>"
    num_reading_strategies = 6
    pad_to_max_length = True
    #strategy = "random"
    C4_processed_filepath = ""
    num_aux_toks = 3

    aux_toks = ["<cls>", "<dec>", "<lm>"]

    dloader = WikiTextDataLoader(strategy="gpt2-remove",
                                 filepath=filepath,
                                 enc_tok=enc_tok, dec_tok=dec_tok,
                                 mlm_tok=mlm_tok, lm_tok=lm_tok,
                                 start_tok=start_tok, end_tok=end_tok,
                                 cls_tok=cls_tok,
                                 sep_tok=sep_tok, mask_tok=mask_tok,
                                 pad_tok=pad_tok, seq_len=768,
                                 pad=True,
                                 a1=a1, a2=a2, a3=a3, a4=a4,
                                 a5=a5, a6=a6, a7=a7, a8=a8,
                                 a9=a9,
                                 passage=passage, p1=p1, p2=p2,
                                 p3=p3, p4=p4, p5=p5, p6=p6,
                                 p7=p7, p8=p8, p9=p9, mqa=mqa,
                                 pmqa=pmqa, bmqa=bmqa,
                                 peentailment=peentailment,
                                 pbentailment=pbentailment,
                                 pcoreference=pcoreference,
                                 bcoreference=bcoreference,
                                 psentiment=psentiment,
                                 pgqa=pgqa, psqa=psqa, gqa=gqa,
                                 pbqa=pbqa,
                                 placeholder=placeholder,
                                 translation=translation,
                                 hypothesis=hypothesis, question=question,
                                 metacognition=metacognition,
                                 unk_rs=unk_rs,
                                 aoint_rs=aoint_rs,
                                 highlighting_rs=highlighting_rs,
                                 reread_rs=reread_rs,
                                 summarize_rs=summarize_rs,
                                 paraphrase_rs=paraphrase_rs,
                                 # tokenizer=None)
                                 tokenizer=config.tokenizer,
                                 sliding_window=sliding_window,
                                 aux_toks=aux_toks,
                                 shuffle=False)

    generator_test = tf.data.Dataset.from_generator(dloader.call__,
                                               output_types=(tf.dtypes.int64,
                                                             tf.dtypes.int64,
                                                             tf.dtypes.int64)).batch(config.batch_size)

    data_dict = {}
    data_dict["test"] = generator_test
    if strategy is not None:
        data_dict["test"] = strategy.experimental_distribute_dataset(data_dict["test"])

    train_class = FineTuningClass(transformer, optimizer, config.loss_object, loss_function_window_size, config.tokenizer,
                                  checkpoint_path_recent="/home/kkno604/Documents/V4 results/Specific-fine-tuning/CommonsenseQA/Checkpoints/",
                                  strategy=strategy, pad_token="<pad>", end_tok = "</s>",
                                  recent_to_keep=20, load_recent=False,
                                  #load_specific_path="/data/kkno604/NMTransformer_pretraining/Checkpoints/pretrain-C4-v4-gpt2/ckpt-48",
                                  load_specific_path="/data/kkno604/NMTransformer_pretraining/Checkpoints/gpt-2-saved-checkpoints/ckpt-111", # 200, 183, 173, 155, 130, 111
                                  #load_specific_path="",
                                  enc_tok="<enc>", dec_tok="<dec>",
                                  output_layer_name="lm", fixed_output=True, stop_gradient=False,
                                  reading_strat_mc_bool=False, lambda_vanilla_set=0.5, lambda_lm=0.2,
                                  vanilla_set_aux_loss_bool=False,
                                  lm_aux_loss_global=False, train_cutoff=0,
                                  train_vanilla_set_only_on_task=False, window_size=sliding_window)

    train_class.get_test_results(e=0,
                                 save_filepath="/data/kkno604/zero-shot-lm/WikiText-103/",
                                 data=data_dict["test"], num_aux_tokens=config.num_aux_toks,
                                 max_generate_len=1, filename_prefix="Wikitext-103-test-ckpt-111-768sw", metrics=["accuracy"],
                                 mode="language_modelling", multiple_answers=False)
