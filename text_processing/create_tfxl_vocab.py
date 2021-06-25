from transformers import TransfoXLTokenizer, XLNetTokenizer, RobertaTokenizer

# note: the order matters in list_of_tokens_to_add. Need to keep a record of this, possibly in another file.
def get_tfxl_tokenizer(list_of_tokens_to_add=["<s>", "</s>", "<nline>", "<enc>", "<dec>", "<lm>",
                                              "<pad>", "<qa>", "<ir>", "<cls>", "<confidence>"]):

    tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")

    for string in list_of_tokens_to_add:
        tokenizer.add_tokens(string)
    print(f"Tokenizer size is {len(tokenizer.get_vocab().keys())}!\n")

    return tokenizer

def get_xlnet_tokenizer(list_of_tokens_to_add=["<nline>", "<enc>", "<dec>", "<lm>", "<qa>",
                                               "<ir>", "<cls>", "<confidence>"]):

    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased") # xlnet-large-cased

    for string in list_of_tokens_to_add:
        tokenizer.add_tokens(string)
    print(f"Tokenizer size is {len(tokenizer.get_vocab().keys())}!\n")

    return tokenizer

def get_roberta_tokenizer(pre_train_tok="roberta-base",
                          list_of_tokens_to_add=["<s>", "</s>", "<nline>", "<enc>", "<dec>", "<lm>",
                                                "<pad>", "<qa>", "<ir>", "<cls>", "<confidence>"]):
    # roberta-large is the name of the other pretrained tokenizer.
    tokenizer = RobertaTokenizer.from_pretrained(pre_train_tok)

    for string in list_of_tokens_to_add:
        tokenizer.add_tokens(string)
    print(f"Tokenizer size is {len(tokenizer.get_vocab().keys())}!\n") # 50273 for base tokenizer (minus 11)
                                                                       # No difference between large and base versions.

    return tokenizer


if __name__ == "__main__":

    tokenizer = get_roberta_tokenizer(pre_train_tok='roberta-large')