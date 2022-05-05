# The Neuromodulated Transformer (NeMoT)

This repository contains code for an extension to the Transformer via the integration of neuromodulation. The goal is to explore neuromodulation and its potential impacts on generalisation in QA. The architecture is coded in TensorFlow and can be found in the models folder.

## Datasets

Code to load all datasets is found in the load_datasets folder. We utilise the Colossol Cleaned Crawled Corpus (C4) for pre-training our model; LAMBADA, WikiText, and PTB to test our pre-trained model's language modelling capabilities; performance in QA is measured on ARC, BoolQ, CommonsenseQA, DROP, MCTest, NarrativeQA, OBQA, PIQA, Quoref, ROPES, SIQA, WG, RACE, and SQuADv2.

## Training

All training files can be found in the training folder. 

