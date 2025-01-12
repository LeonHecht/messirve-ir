from transformers import AutoModel

def get_mamba_model():
    """ Load Mamba embeddings model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # load model from
    path = "/home/leon/tesis/mamba-ir/results_300M_diverse_shuffle_75train/mamba-130m-spanish-legal-300M-tokens-diverse"
    model = AutoModelForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


def get_xlm_roberta_model():
    # Load model directly
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    model = AutoModelForMaskedLM.from_pretrained("FacebookAI/xlm-roberta-base")
    return model, tokenizer