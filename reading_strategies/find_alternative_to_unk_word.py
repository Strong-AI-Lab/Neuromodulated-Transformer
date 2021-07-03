from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import lin_thesaurus as thes

from spellchecker import SpellChecker # otherwise use fuzzy matching to see
from textblob import Word

import sys
sys.path.append("..")

from text_processing.create_tfxl_vocab import get_tfxl_tokenizer

class replace_unk_RS:
    '''
    Class: replace_unk_RS
    Description: Implementation of a reading strategy that `guesses the meaning of unknown words'. Four strategies
        are applied here to attempt to achieve this (in the following order).
        1. Lemmatization,
        2. Thesaurus lookup.
        3. Stemming,
        4. Check for spelling errors, fix the errors and check if in tokenizers vocab. (repeat steps 1-3 if not).
    Input:
        tokenizer: (transformer class tokenizer)
        spellchecker: (string) String representing which spell checker to use.
        spell_checker_num_candidates: (int) Represents the maximum number of candidates to consider.
    '''
    def __init__(self, tokenizer, spellchecker):

        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.get_vocab()

        assert spellchecker in ["textblob", "spellchecker", "blob_then_spellchecker"], f"Invalid spell checker parameter: {spellchecker} \n" \
                                                             f"It should be in one of \"textblob\", \"spellchecker\" or \"blob_then_spellchecker\"."
        self.spellingchecker = spellchecker

        #self.input = input # example ["shaq", "and", "kobe", "are", "the", "greatest", "one", "two", "punch"]
        #self.input_unk_indices = input_unk_indices # [8,24,33]

        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()



    def __call__(self, input, input_unk_indices):
        '''
        Function: __call__
        Description: When the function is called find a new word that is in the tokenizer's vocabulary.
        Input:
            input: (list; (string)) list of words
            input_unk_index: (list; (int)) list of integers representing indices of an unknown token.
        Return:
            todo
        '''
        for index in input_unk_indices:
            corr_word = self.get_replacement_word(input[index])
            if corr_word[0]:
                input[index] = corr_word[1]
                continue
            corrected_word = self.spelling_check_greedy(input[index])  # TODO: then iterate though lem_, synomyums and stem_ again.
            if isinstance(corrected_word, str):
                input[index] = corrected_word
            # otherwise the input at the chosen index is still <unk>


    def get_replacement_word(self, word):
        '''
        Function: get_replacement_word
        Description: find a replacement for the
        Input:
            word
        Return:
        '''
        lem_ = self.lemmatization(word)
        if lem_[0]: return [True, lem_[1]]
        syn = self.get_synomyms(word)  # return a synonym, that is in the vocab dictionary greedily. # TODO
        if syn[0]: return [True, syn[1]]
        stem_ = self.stemming(self.input[index])
        if stem_[0]: return [True, stem_[1]]
        return [False, '']

    def lemmatization(self, word):
        assert isinstance(word, str), f"word should be of type str, it currently is of type {type(word)}!"
        lem_word = self.lemmatizer.lemmatize(word)
        return [True, lem_word] if lem_word in self.vocab else [False, word]

    def stemming(self, word):
        assert isinstance(word, str), f"word should be of type str, it currently is of type {type(word)}!"
        stem_word = self.stemmer.stem(word)
        return [True, stem_word] if lem_word in self.vocab else [False, word]

    def _get_blob_spellcheck(self, word, cutoff):
        spell_checker = Word(word)
        candidate_words = spell_checker.spellcheck()  # sample output: [('Basket', 0.42105263157894735), ('Casket', 0.2631578947368421),
        # ('Bassett', 0.21052631578947367), ('Caskets', 0.05263157894736842), ('Baskets', 0.05263157894736842)]
        for tup in candidate_words:
            if tup[0] in self.vocab and tup[1] > cutoff:
                return tup[0]
            elif tup[0].lower() in self.vocab and tup[1] > cutoff:  # TODO: consider this, is it worth it factoring in the large vocabulary size.
                return tup[0].lower()
        return False

    def _get_pyspellcheck(self, word):
        spell_checker = SpellChecker()
        candidate_words = list(spell_checker.candidates(word))
        for cword in candidate_words: # cword is a string.
            if cword in self.vocab:
                return cword
            elif cword.lower() in self.vocab:  # TODO: consider if this should be here (consider efficiency).
                return cword.lower()
        return False

    def spelling_check_greedy(self, word, cutoff=0.5):
        '''
        Function: spelling_check_greedy
        Description: Checks for a spelling error, if one is found greedily then return it if it is in the tokenizer's vocabulary.
          (i.e. return the first spelling correction).
        Input:
            word: (string) Word to get a synonym for.
            cutoff: (float) Represents the minumum score given by the thesaurus with an word for it to be considered.
        Return:
            (list; (string) where each string represents a candidate spelling error correction)
        '''

        if self.spellingchecker == "textblob":
            word_ = self._get_blob_spellcheck(word, cutoff)
            if isinstance(word_, str): return word_ # if not then it will be False
        elif self.spellingchecker == "spellchecker":
            word_ = self._get_pyspellcheck(word)
            if isinstance(word_, str): return word_  # if not then it will be False
        elif self.spellingchecker == "blob_then_spellchecker":
            word_ = self._get_blob_spellcheck(word, cutoff)
            if isinstance(word_, str): return word_  # if not then it will be False
            word__ = self._get_pyspellcheck(word)
            if isinstance(word_, str): return word__  # if not then it will be False
        else:
            raise Exception(f"Invalid spellingchecker value: {self.spellingchecker}")
        return False

    def get_synomyms(self, word, cutoff=0.1):
        '''
        Function: get_synonyms
        Description: Function that returns the first synonym that is in the tokenizer's vocab, and its score is above the cutoff.
            (note: need to stress that this shouldn't be done during the training of pre-trained model as the
            synonym may be innacurate)
        Input:
            word: (string) Word to get a synonym for.
            cutoff: (float) Represents the minumum score given by the thesaurus with an word for it to be considered.
        Return:
            (string) representing the replacement synonym if found; None otherwise.
        '''
        # TODO: Returns the first synonym with the highest score for simplicity.
        assert isinstance(word, str), f"The word should be of string type (word is of type :{type(word)})"

        syn_word = None
        score = 0
        for i in range(len(thes.scored_synonyms(word1))):
            # i = 0 simA: adjectives?
            # i = 1 simN: nouns
            # i = 2 simV: verbs?
            # print(f"list of scored synomyms for the current ngram: \n{list(thes.scored_synonyms(word1)[i][0])}")
            for tup in list(thes.scored_synonyms(word1)[i][1]): # tup shape is (str, int)
                if tup[1] > score and tup[0] in self.vocab: # TODO put lower case here? probably note and remove others as well. reminder.
                    syn_word = tup[0]
                    score = tup[1]
        if syn_word is None: return [False, '']
        else: return [True, syn_word]


if __name__ == "__main__":

    word1 = "hairy"
    word2 = "basketball"

    tokenizer = get_tfxl_tokenizer()
    vocab = tokenizer.get_vocab()

    print(f"Getting synomyms for {word1}")
    #print(f"list of synomyms for the current ngram: \n{thes.scored_synonyms(word1)}")
    for i in range(len(thes.scored_synonyms(word1))):
        #print(f"list of scored synomyms for the current ngram: \n{list(thes.scored_synonyms(word1)[i][0])}")
        print(f"list of scored synomyms for the current ngram: \n{list(thes.scored_synonyms(word1)[i][0])}\t{list(thes.scored_synonyms(word1)[i][1])}")
    #print(f"list of scored synomyms for the current ngram: \n{list(thes.scored_synonyms(word1)[1][0])}\t{list(thes.scored_synonyms(word1)[1][1])}")
