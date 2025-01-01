from transformers import AutoModel

def load_model(checkpoint="bert-base-uncased"):
    """
    Loads a Hugging Face transformer model.

    Args:
        model_name (str): Name of the pretrained model.

    Returns:
        model: Hugging Face model instance.
    """
    model = AutoModel.from_pretrained(checkpoint)
    return model
