from transformers import TransfoXLTokenizer, XLNetTokenizer, RobertaTokenizer
#from collections import OrderedDict

def get_tfxl_tokenizer(pad_tok="<pad>", start_tok="<s>", end_tok="</s>", cls_tok="<cls>", sep_tok="<sep>",
                       nline_tok="<nline>", confidence_tok="<confidence>",
                       list_of_aux_tokens=["<enc>", "<dec>", "<lm>", "<qa>", "<ir>"]):

    tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")

    if pad_tok is not None:
        tokenizer.add_tokens(pad_tok)
    if start_tok is not None:
        tokenizer.add_tokens(start_tok)
    if end_tok is not None:
        tokenizer.add_tokens(end_tok)
    if cls_tok is not None:
        tokenizer.add_tokens(cls_tok)
    if sep_tok is not None:
        tokenizer.add_tokens(sep_tok)
    if nline_tok is not None:
        tokenizer.add_tokens(nline_tok)
    if confidence_tok is not None:
        tokenizer.add_tokens(confidence_tok)
    for i, token in enumerate(list_of_aux_tokens):
        tokenizer.add_tokens(token)

    return tokenizer

if __name__ == "__main__":
    tokenizer = get_tfxl_tokenizer()
    pad_tok_id = tokenizer.encode("<pad>", add_special_tokens=True, pad_to_max_length=False,
                                              return_token_type_ids=False, return_attention_mask=False)
    print(f"The pad token id is {pad_tok_id}.")
    print(tokenizer.batch_encode_plus(["the hawks are overrated","shaq is a beast and a monster!"], max_length=20,
                                add_special_tokens=True, pad_to_max_length=True,
                                return_token_type_ids=False, return_attention_mask=False,
                                      return_overflowing_tokens=True))