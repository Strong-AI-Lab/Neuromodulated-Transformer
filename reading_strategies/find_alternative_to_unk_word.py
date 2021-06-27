from nltk.corpus import lin_thesaurus as thes

import sys
sys.path.append("..")

from text_processing.create_tfxl_vocab import get_tfxl_tokenizer

class unk_RS:

    def __init__(self, tokenizer_vocab_dict):

        self.tokenizer_vocab_dict = tokenizer_vocab_dict

    def get_synomyms(self, word, top):

        assert isinstance(word, str), f"The word should be of string type (word is of type :{type(word)})"






if __name__ == "__main__":

    word1 = "business"
    word2 = "basketball"

    tokenizer = get_tfxl_tokenizer()
    vocab = tokenizer.get_vocab()

    thesaurus = unk_RS(vocab)

    print(f"Getting synomyms for {word1}")
    #print(f"list of synomyms for the current ngram: \n{thes.scored_synonyms(word1)}")
    print(f"list of scored synomyms for the current ngram: \n{thes.scored_synonyms(word1)[0][1]}")