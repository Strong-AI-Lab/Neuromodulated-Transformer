import os

import tensorflow.python.framework.ops

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
GPUS_AVAILABLE = 1

import sys
sys.path.append("..")

import tensorflow as tf

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

print(f"\nTensorflow version: {tf.__version__}\n")

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np

tf.config.run_functions_eagerly(False)

#from training.parent_train import *
#from training.train_wikitext103.train_wikitext103_class import *
from models.NMTransformer import *
from models.config_model import *
#from load_datasets.language_modelling.load_wikitext import *

if __name__ == "__main__":

    config = V4ConfigMediumSize(strategy="MirroredStrategy", batch_size=32*GPUS_AVAILABLE,
                                loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none'),
                                learning_rate=tf.keras.optimizers.schedules.CosineDecay(0.0001, decay_steps=1000000),
                                #learning_rate=0.00005,
                                vocab_filepath="/data/kkno604/Neuromodulated-Transformer/vocabulary/vocab1.txt",
                                gpt2_117=True,
                                tokenizer="gpt2")

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

    ckpt = tf.train.Checkpoint(model=transformer)
    ckpt.restore("/data/kkno604/NMTransformer_pretraining/Checkpoints/gpt-2-saved-checkpoints/ckpt-200")

    #string_input = "<dec> <lm> <null> Kobe Bryant is the greatest laker of all time"
    #string_input = "<dec> <lm> <null> Al Bunshaft is the Senior Vice President of Dassault Systèmes’ Americas Corporation where he spearheads key strategic initiatives and corporate leadership programs, including the company’s expansion into the U.S. government federal sector. He is also responsible for the company’s relationships with key stakeholders such as academic institutions, financial and industry analysts, public officials, as well as foreign diplomats.\nAs Managing Director of Dassault Systèmes Americas from 2010 until 2013, he helped build the foundation for future company growth by leading the operations and communications of an organization of more than 3,000 employees. Bunshaft led the selection, design, construction and opening of the company’s North American headquarters, an award-winning campus recognized for its sustainable innovation practices and located in the heart of Boston’s technology belt in Waltham, Mass. The campus was awarded Best New Workplace in New England in 2012, and has been awarded one of only a handful of LEED Platinum designations in the Boston area.\nA common thread in Mr. Bunshaft’s career has been his expertise in visualization and computer graphics. In the thirty years since doing his post-graduate work at the National Science Foundation Center for Interactive Computer Graphics he has led efforts to introduce new technologies into far ranging industries. For example, he assisted major manufacturing companies’ in their transition from physical to digital design practices and played a key role in the development of the first digitally-designed automobile. With vehicles now being mobile computing platforms, he is focused on how this intelligence will be integrated into our urban infrastructure of the future. A subtle balance of technology, policy, and infrastructure evolution will be needed maximize society’s benefits.\nBunshaft is Dassault Systèmes’ leading voice in science, technology, engineering and mathematics (STEM) initiatives and was named one of the top 100 CEO leaders in STEM in the U.S. He represents the company as a member of the STEM subcommittee of the Clinton Global Initiative – America, and is a board member of the Massachusetts High Technology Council, where he co-chairs the Education and Talent Development Committee. Bunshaft also serves as an advisory board member at the Department of Information and Computer Science of The State University of New York at Albany. He was recently named to the Olin College of Engineering President’s Council. He regularly writes and speaks about STEM topics.\nHe received his Bachelor of Science in Computer Science and Mathematics from University at Albany,"
    string_input = "<dec> <lm> <null> I enjoy walking with my cute dog"
    #string_input = "<dec> <lm> <null> I love my dog Marley, she is so"
    #string_input = "<dec> <lm> <null> Kobe Bryant"
    pad_to_length = config.max_seq_len_nm

    padding_id = config.tokenizer.encode_single("<pad>")
    if len(padding_id) == 1:
        padding_id = padding_id[0]
    else:
        raise Exception("The padding token should only have one id! (it hasn't been added to the vocabulary)")

    num_aux_tokens = config.num_aux_toks
    num_to_generate = 100
    k = 4
    tokenizer=config.tokenizer

    enc_string, original, generated_output = transformer.generate_natural_language_top_k(string_input, tokenizer, pad_to_length,
                                                                                         padding_id, num_aux_tokens,
                                                                                         num_to_generate, k)

    print(f"enc_string: {enc_string} \n"
          f"original: {original} \n"
          f"generated_output: {generated_output}")
