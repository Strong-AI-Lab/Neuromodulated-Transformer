from transformers import TransfoXLTokenizer, XLNetTokenizer, RobertaTokenizer
from collections import OrderedDict

def get_tfxl_tokenizer(pad_tok="<pad>", start_tok="<s>", end_tok="<\s>", cls_tok="<cls>", sep_tok="<sep>",
                       nline_tok="<nline>", confidence_tok="<confidence>",
                       list_of_aux_tokens=["<enc>", "<dec>", "<lm>", "<qa>", "<ir>"]):

    tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")

    special_tokens = OrderedDict()
    non_special_tokens = OrderedDict()

    counter = 0
    counter_spec = 0
    if pad_tok is not None:
        special_tokens["pad_token"] = pad_tok
        counter_spec += 1
    if start_tok is not None:
        non_special_tokens["start_tok"] = start_tok
        counter += 1
    if end_tok is not None:
        non_special_tokens["end_tok"] = end_tok
        counter += 1
    if cls_tok is not None:
        special_tokens["cls_token"] = cls_tok
        counter_spec += 1
    if sep_tok is not None:
        special_tokens["sep_token"] = sep_tok
        counter_spec += 1
    if nline_tok is not None:
        non_special_tokens["nline_tok"] = nline_tok
        counter += 1
    if confidence_tok is not None:
        non_special_tokens["confidence_tok"] = confidence_tok
        counter += 1
    for i, token in enumerate(list_of_aux_tokens):
        non_special_tokens["aux_tok_"+str(i+1)+"_"+str(token)] = token
        counter += 1

    if counter != 0:
        tokenizer.add_tokens(non_special_tokens)

    if counter_spec != 0:
        tokenizer.add_special_tokens(special_tokens)

    return tokenizer

if __name__ == "__main__":
    tokenizer = get_tfxl_tokenizer()
    pad_tok_id = tokenizer.encode("<pad>", add_special_tokens=True, pad_to_max_length=False,
                                              return_token_type_ids=False, return_attention_mask=False)
    print(f"The pad token id is {pad_tok_id}.")
    print(tokenizer.batch_encode_plus(["the hawks are overrated","shaq is a beast and is a monster! marley hello hello hey hi hi a a a a a a a a a a a a a a "], max_length=20,
                                add_special_tokens=True, pad_to_max_length=True,
                                return_token_type_ids=False, return_attention_mask=False,
                                      return_overflowing_tokens=True))