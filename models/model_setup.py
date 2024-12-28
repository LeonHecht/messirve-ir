from transformers import AutoModel

def load_model(model_name="bert-base-uncased"):
    """
    Loads a Hugging Face transformer model.

    Args:
        model_name (str): Name of the pretrained model.

    Returns:
        model: Hugging Face model instance.
    """
    model = AutoModel.from_pretrained(model_name)
    return model

model = load_model()
