'''
File name: tokens_to_add.py
Author: Kobe Knowles
Date created: 03/09/21
Data last modified: 03/09/21
Python Version: 3.8
'''

import json

def json_save_vocab(vocab, filename="vocab1.txt"):
    assert isinstance(vocab, list), f"The vocabulary must be of type list, got {type(vocab)}!"
    with open(filename, "w") as f:
        json.dump(vocab, f)

if __name__ == "__main__":
    vocab_to_add = ["<passage>", "<p1>", "<p2>", "<p3>", "<p4>", "<p5>", "<p6>", "<p7>", "<p8>", "<p9>",
                    "<h>", "<q>", "(1)", "(2)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)", "(8)", "(9)",
                    "<enc>", "<dec>", "<mc>", "<placeholder>",
                    "<lm>", "<mlm>", "<mqa>", "<pmqa>", "<bmqa>", "<pbmqa>", "<peentailment>", "<pbentailment>",
                    "<pcoreference>", "<bcoreference>", "<pbcoreference>", "<psentiment>",
                    "<pgqa>", "<psqa>", "<gqa>", "<pbqa>", "<translation>",
                    "<s>", "</s>", "<cls>", "<sep>", "<mask>", "<pad>", "<null>",
                    "<unk_rs>", "<aoint_rs>", "<highlighting_rs>", "<reread_rs>", "<summarize_rs>", "<paraphrase_rs>"]
    json_save_vocab(vocab_to_add, filename="vocab1.txt")
    with open("vocab1.txt", "r") as f:
        print(json.load(f))