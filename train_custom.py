from models.model_setup import get_mamba_model, get_auto_model
from datasets import load_from_disk
from trainers.info_nce_trainer import InfoNCERetrievalTrainerHNLLM
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import torch
from torch.nn.utils.rnn import pad_sequence
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
print("Executable", sys.executable)

import os
# make only GPU0 visible
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

STORAGE_DIR = os.getenv("STORAGE_DIR", "/media/discoexterno/leon")

MAX_QUERY_LEN = 128
MAX_DOC_LEN = 256


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


def tokenize_with_hard_negatives(tokenizer, examples, append_eos=False):
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


def train():
    country = "ar"
    print(f"Training on {country} dataset...")
    ds = load_from_disk(f"messirve_{country}")

    # load train_df from disk
    train_ds = load_from_disk(f"messirve_train_{country}_hard_negatives")

    # train_ds = train_ds.select(range(5000))
    test_ds = ds["test"]

    checkpoint = STORAGE_DIR + "/qwen-2-vec/run_89622_texts_1_epoch/output-model/checkpoint-2500"

    # model, tokenizer = get_mamba_model()
    # model, tokenizer = get_auto_model(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    # Apply tokenization to the dataset
    train_ds = train_ds.map(lambda x: tokenize_with_hard_negatives(tokenizer, x, append_eos=True), batched=True)
    test_ds = test_ds.map(lambda x: tokenize_function(tokenizer, x, append_eos=True), batched=True)

    # check if first token is CLS
    print(tokenizer.eos_token_id)
    print(tokenizer.eos_token)
    # check if first query instance in train_ds has CLS token
    print(train_ds[0]["query_input_ids"][-1])

    # docs_train = ds["train"]["docid_text"]
    # queries_train = ds["train"]["query"]
    # doc_ids_train = ds["train"]["docid"]
    # query_ids_train = ds["train"]["id"]

    # docs_test = ds["test"]["docid_text"]
    # queries_test = ds["test"]["query"]
    # doc_ids_test = ds["test"]["docid"]
    # query_ids_test = ds["test"]["id"]

    # # Define Data Collator for On-the-Fly Tokenization
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir="./results",           # Directory to save checkpoints
        evaluation_strategy="steps",     # Evaluate at the end of each epoch
        eval_steps=500,                  # Evaluate every 500 steps
        learning_rate=1e-5,              # Learning rate
        per_device_train_batch_size=2,  # Batch size for training
        per_device_eval_batch_size=2,   # Batch size for evaluation
        gradient_accumulation_steps=4,
        num_train_epochs=3,              # Number of epochs
        weight_decay=0.01,               # Weight decay
        max_grad_norm=30,                # Maximum gradient norm
        save_strategy="epoch",           # Save model checkpoints at the end of each epoch
        logging_dir="./logs",            # Directory for logs
        logging_steps=10,                # Log every 10 steps
        save_total_limit=1,              # Save only the last checkpoint
        remove_unused_columns=False,
        warmup_ratio=0.1,
    )

    # Create Trainer Instance
    trainer = InfoNCERetrievalTrainerHNLLM(
        model=model,
        args=training_args,
        train_dataset=train_ds,     # Raw dataset for training
        eval_dataset=test_ds,       # Raw dataset for evaluation
        tokenizer=tokenizer,             # Tokenizer for on-the-fly tokenization
        data_collator=custom_data_collator,     # Handles dynamic padding and tokenization
    )

    # Train the Model
    trainer.train()

    # # Evaluate the Model
    # metrics = trainer.evaluate()
    # print(metrics)

    # Save the Model
    trainer.save_model("./saved_model")


if __name__ == "__main__":
    train()