from config import MAX_QUERY_LEN, MAX_DOC_LEN
from tqdm import tqdm


def tokenize_with_manual_eos(tokenizer, text_list, max_length):
    """
    Tokenize each item in text_list to max_length - 1, 
    then manually append EOS, then (optional) pad up to max_length.
    """
    # 1) Tokenize with max_length - 1, no padding.
    #    We rely on a custom approach to handle final EOS and padding.
    partial_encoded = tokenizer(
        text_list,
        truncation=True,
        max_length=max_length - 1,
        padding=False,
    )

    # 2) Manually append EOS for each sequence
    eos_id = tokenizer.eos_token_id
    new_input_ids = []
    new_attention_masks = []

    for inp_ids, att_mask in zip(partial_encoded["input_ids"], partial_encoded["attention_mask"]):
        # Append EOS token
        inp_ids.append(eos_id)
        att_mask.append(1)

        # 3) (Optional) Now pad if we want a fixed size == max_length
        #    If you truly want each sequence to be exactly max_length:
        pad_len = max_length - len(inp_ids)
        if pad_len > 0:
            inp_ids.extend([tokenizer.pad_token_id] * pad_len)
            att_mask.extend([0] * pad_len)

        new_input_ids.append(inp_ids)
        new_attention_masks.append(att_mask)

    return {
        "input_ids": new_input_ids,
        "attention_mask": new_attention_masks,
    }


def tokenize_with_hard_negatives_messirve(tokenizer, examples, append_eos=False):
    # Flatten the list of hard negatives for tokenization
    flattened_negatives = []
    for hard_neg_docs in examples["hard_negatives"]:
        flattened_negatives.extend(hard_neg_docs)
    
    # Tokenize queries
    tokenized_queries = tokenize_with_manual_eos(tokenizer, examples["query"], max_length=MAX_QUERY_LEN)
    
    # Tokenize positive documents
    tokenized_docs = tokenize_with_manual_eos(tokenizer, examples["docid_text"], max_length=MAX_DOC_LEN)
    
    # Tokenize hard negatives (flattened)
    tokenized_all_negatives = tokenize_with_manual_eos(tokenizer, flattened_negatives, max_length=MAX_DOC_LEN)
    
    # Rolling index to rebuild the structure
    rolling_index = 0
    neg_input_ids = []
    neg_attention_masks = []
    for neg_list in examples["hard_negatives"]:
        length = len(neg_list)
        neg_input_ids.append(
            tokenized_all_negatives["input_ids"][rolling_index : rolling_index + length]
        )
        neg_attention_masks.append(
            tokenized_all_negatives["attention_mask"][rolling_index : rolling_index + length]
        )
        rolling_index += length
    
    return {
        "query_input_ids": tokenized_queries["input_ids"],
        "query_attention_mask": tokenized_queries["attention_mask"],
        "doc_input_ids": tokenized_docs["input_ids"],
        "doc_attention_mask": tokenized_docs["attention_mask"],
        "neg_input_ids": neg_input_ids,
        "neg_attention_mask": neg_attention_masks,
    }


def tokenize_with_hard_negatives_msmarco(tokenizer, examples: dict, qid_to_query, pid_to_passage, num_negs, max_query_len, max_doc_len):
    """
    Due to dataset.map(batched=True) parameter, examples is a dictionary where
    each key corresponds to a column in your dataset and the value is a list
    of items for that column.
    Example:
    {
        "query": ["how to make pizza", "how to make pasta"],
        "positive": ["To make pizza, you need...", "To make pasta, you need..."],
        "negative_1": ["To make a cake, you need...", "To make a salad, you need..."],
        "negative_2": ["To make a sandwich, you need...", "To make a burger, you need..."],
        ...
    }
    """
    queries = [qid_to_query[qid] for qid in examples["query"]]
    positives = [pid_to_passage[pid] for pid in examples["positive"]]

    # Flatten the list of hard negatives for tokenization
    flattened_negatives = []
    for i in tqdm(range(num_negs)):
        neg_pids = examples[f"negative_{i+1}"]
        flattened_negatives.extend([pid_to_passage[neg_pid] for neg_pid in neg_pids])
    
    print("Starting tokenization of queries...", end="")
    # Tokenize queries
    tokenized_queries = tokenize_with_manual_eos(tokenizer, queries, max_length=max_query_len)
    print("Done")

    print("Starting tokenization of positive documents...", end="")
    # Tokenize positive documents
    tokenized_docs = tokenize_with_manual_eos(tokenizer, positives, max_length=max_doc_len)
    print("Done")

    print("Starting tokenization of hard negatives...", end="")
    # Tokenize hard negatives (flattened)
    tokenized_all_negatives = tokenize_with_manual_eos(tokenizer, flattened_negatives, max_length=max_doc_len)
    print("Done")

    # Rolling index to rebuild the structure
    rolling_index = 0
    neg_input_ids = []
    neg_attention_masks = []
    for i in range(len(examples["negative_1"])):
        neg_input_ids.append(
            tokenized_all_negatives["input_ids"][rolling_index : rolling_index + num_negs]
        )
        neg_attention_masks.append(
            tokenized_all_negatives["attention_mask"][rolling_index : rolling_index + num_negs]
        )
        rolling_index += num_negs
    
    return {
        "query_input_ids": tokenized_queries["input_ids"],
        "query_attention_mask": tokenized_queries["attention_mask"],
        "doc_input_ids": tokenized_docs["input_ids"],
        "doc_attention_mask": tokenized_docs["attention_mask"],
        "neg_input_ids": neg_input_ids,     # list of lists
        "neg_attention_mask": neg_attention_masks,      # list of lists
    }


def tokenize_function(tokenizer, examples, append_eos=False):
    # Append the EOS token to queries and documents
    if append_eos:
        examples["query"] = [q + tokenizer.eos_token for q in examples["query"]]
        examples["docid_text"] = [d + tokenizer.eos_token for d in examples["docid_text"]]

    tokenized_queries = tokenizer(examples["query"], truncation=True, padding=True, max_length=MAX_QUERY_LEN)
    tokenized_docs = tokenizer(examples["docid_text"], truncation=True, padding=True, max_length=MAX_DOC_LEN)

    # Return tokenized queries and documents
    return {
        "query_input_ids": tokenized_queries["input_ids"],
        "query_attention_mask": tokenized_queries["attention_mask"],
        "doc_input_ids": tokenized_docs["input_ids"],
        "doc_attention_mask": tokenized_docs["attention_mask"],
    }


def custom_data_collator(batch):
    """
    Since tokenize_with_manual_eos() already pads each sequence to a uniform max_length,
    here we just stack them into tensors. We assume:
      - Each example in 'batch' has identical shapes for query, doc, etc.
      - The number of negatives (e.g. n_neg=5) is the same for all examples.

      QUESTION: Are we assuming that the query length is the same as the doc length?
    """
    import torch

    # -- Queries --
    query_input_ids = torch.stack(
        [torch.tensor(example["query_input_ids"], dtype=torch.long) for example in batch],
        dim=0
    )  # shape: (batch_size, query_seq_len)
    query_attention_mask = torch.stack(
        [torch.tensor(example["query_attention_mask"], dtype=torch.long) for example in batch],
        dim=0
    )  # shape: (batch_size, query_seq_len)

    # -- Positive Docs --
    doc_input_ids = torch.stack(
        [torch.tensor(example["doc_input_ids"], dtype=torch.long) for example in batch],
        dim=0
    )  # shape: (batch_size, doc_seq_len)
    doc_attention_mask = torch.stack(
        [torch.tensor(example["doc_attention_mask"], dtype=torch.long) for example in batch],
        dim=0
    )  # shape: (batch_size, doc_seq_len)

    # -- Hard Negatives --
    # Each example["neg_input_ids"] is a list of length n_neg,
    # where each item is a list of length doc_seq_len (already padded).
    # So example["neg_input_ids"] => shape (n_neg, doc_seq_len)
    neg_input_ids = torch.stack(
        [torch.tensor(example["neg_input_ids"], dtype=torch.long) for example in batch],
        dim=0
    )  # shape: (batch_size, n_neg, doc_seq_len)

    neg_attention_mask = torch.stack(
        [torch.tensor(example["neg_attention_mask"], dtype=torch.long) for example in batch],
        dim=0
    )  # shape: (batch_size, n_neg, doc_seq_len)

    return {
        "query_input_ids": query_input_ids,
        "query_attention_mask": query_attention_mask,
        "doc_input_ids": doc_input_ids,
        "doc_attention_mask": doc_attention_mask,
        "neg_input_ids": neg_input_ids,
        "neg_attention_mask": neg_attention_mask,
    }