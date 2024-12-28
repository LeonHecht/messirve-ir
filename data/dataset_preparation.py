from transformers import AutoTokenizer
from datasets import Dataset

def prepare_dataset(data, tokenizer):
    """
    Prepares and tokenizes the dataset for InfoNCE training.

    Args:
        data (dict): A dictionary with keys `query`, `positive_doc`, and optionally `negative_docs`.
        tokenizer: Hugging Face tokenizer instance.

    Returns:
        Dataset: Tokenized dataset.
    """
    def tokenize(batch):
        """
        Tokenizes the batch
        """
        # Tokenize each field and convert BatchEncoding to a dict of lists
        tokenized_query = tokenizer(batch["query"], truncation=True, padding=True, return_tensors=None)
        tokenized_positive = tokenizer(batch["positive_doc"], truncation=True, padding=True, return_tensors=None)
        
        result = {
            "query_input_ids": tokenized_query["input_ids"],
            "query_attention_mask": tokenized_query["attention_mask"],
            "positive_input_ids": tokenized_positive["input_ids"],
            "positive_attention_mask": tokenized_positive["attention_mask"],
        }
        
        # Handle negative documents, if available
        if "negative_docs" in batch:
            tokenized_negatives = tokenizer(batch["negative_docs"], truncation=True, padding=True, return_tensors=None)
            result["negative_input_ids"] = tokenized_negatives["input_ids"]
            result["negative_attention_mask"] = tokenized_negatives["attention_mask"]

        return result


    dataset = Dataset.from_dict(data)
    return dataset.map(tokenize, batched=True)