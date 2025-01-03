def get_mamba_model():
    """ Load Mamba embeddings model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # load model from
    path = "/home/leon/tesis/mamba-ir/results_300M_diverse_shuffle_75train/mamba-130m-spanish-legal-300M-tokens-diverse"
    model = AutoModelForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


def embed_mamba(model, tokenizer, docs, queries, doc_ids, query_ids):
    inputs_docs = tokenizer(
        docs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    )
    input_ids_docs = inputs_docs["input_ids"]
    attention_mask_docs = inputs_docs["attention_mask"]

    inputs_queries = tokenizer(
        queries,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )

    out = model.generate(
        input_ids,
        attention_mask=attention_mask,
        repetition_penalty=1.2,
        temperature=0.7,
        do_sample=True
    )
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    print(response)


def main():
    model, tokenizer = get_mamba_model()
    embed_mamba(model, tokenizer, None, None, None, None)


if __name__ == "__main__":
    main()