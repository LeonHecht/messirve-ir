from models.model_setup import get_mamba_model, get_auto_model
from datasets import load_from_disk
from trainers.info_nce_trainer import InfoNCERetrievalTrainerHN
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import torch
from torch.nn.utils.rnn import pad_sequence
import pickle


def add_hard_negatives(example, docid_to_text, hard_negatives):
    # Example logic to add hard negatives:
    # You can retrieve hard negatives from some precomputed data or logic
    query_id = example["id"]
    hard_neg_docids = hard_negatives[str(query_id)]
    # Get hard negative texts using the precomputed dictionary
    hard_neg_doc_texts = [docid_to_text[docid] for docid in hard_neg_docids if docid in docid_to_text]
    
    # Return the updated example
    example["hard_negatives"] = hard_neg_doc_texts
    return example


def preprocess_dataset(train_ds, hard_negatives):
    # Precompute a mapping from docid to docid_text
    docid_to_text = {item["docid"]: item["docid_text"] for item in train_ds}

    # Map the function over the dataset
    train_ds = train_ds.map(
        lambda example: add_hard_negatives(example, docid_to_text, hard_negatives),
        batched=False  # If hard_negatives are added per example, keep this False
    )
    return train_ds


def limit_hard_negatives(example):
    # Limit hard negatives to 5 per query
    example["hard_negatives"] = example["hard_negatives"][:5]
    return example


def tokenize_with_hard_negatives(tokenizer, examples):
    # Flatten the list of hard negatives for tokenization
    flattened_negatives = [doc for hard_neg_docs in examples["hard_negatives"] for doc in hard_neg_docs]
    
    # Tokenize queries
    tokenized_queries = tokenizer(
        examples["query"], truncation=True, padding=True, max_length=128
    )
    
    # Tokenize positive documents
    tokenized_docs = tokenizer(
        examples["docid_text"], truncation=True, padding=True, max_length=512
    )
    
    # Tokenize hard negatives (flattened)
    tokenized_negatives = tokenizer(
        flattened_negatives, truncation=True, padding=True, max_length=512
    )
    
    # Reshape the tokenized negatives back to the original structure
    tokenized_negatives = {
        key: [
            tokenized_negatives[key][i : i + len(hard_neg_docs)]
            for i, hard_neg_docs in enumerate(examples["hard_negatives"])
        ]
        for key in tokenized_negatives
    }
    
    return {
        "query_input_ids": tokenized_queries["input_ids"],
        "query_attention_mask": tokenized_queries["attention_mask"],
        "doc_input_ids": tokenized_docs["input_ids"],
        "doc_attention_mask": tokenized_docs["attention_mask"],
        "neg_input_ids": tokenized_negatives["input_ids"],
        "neg_attention_mask": tokenized_negatives["attention_mask"],
    }


def tokenize_function(tokenizer, examples, append_eos=False):
    # Append the EOS token to queries and documents
    if append_eos:
        examples["query"] = [q + tokenizer.eos_token for q in examples["query"]]
        examples["docid_text"] = [d + tokenizer.eos_token for d in examples["docid_text"]]

    tokenized_queries = tokenizer(examples["query"], truncation=True, padding=True, max_length=128)
    tokenized_docs = tokenizer(examples["docid_text"], truncation=True, padding=True, max_length=512)

    # Return tokenized queries and documents
    return {
        "query_input_ids": tokenized_queries["input_ids"],
        "query_attention_mask": tokenized_queries["attention_mask"],
        "doc_input_ids": tokenized_docs["input_ids"],
        "doc_attention_mask": tokenized_docs["attention_mask"],
    }


def custom_data_collator(batch):
    # Pad query input IDs and attention masks
    query_input_ids = [torch.tensor(example["query_input_ids"]) for example in batch]
    query_attention_mask = [torch.tensor(example["query_attention_mask"]) for example in batch]

    # Pad positive document input IDs and attention masks
    doc_input_ids = [torch.tensor(example["doc_input_ids"]) for example in batch]
    doc_attention_mask = [torch.tensor(example["doc_attention_mask"]) for example in batch]

    # Pad hard negatives input IDs and attention masks (if present)
    if "neg_input_ids" in batch[0] and "neg_attention_mask" in batch[0]:
        neg_input_ids = [
            torch.tensor(hard_negative) for example in batch for hard_negative in example["neg_input_ids"]
        ]
        neg_attention_mask = [
            torch.tensor(hard_negative_mask) for example in batch for hard_negative_mask in example["neg_attention_mask"]
        ]
        
        neg_input_ids_padded = pad_sequence(neg_input_ids, batch_first=True, padding_value=0)
        neg_attention_mask_padded = pad_sequence(neg_attention_mask, batch_first=True, padding_value=0)
    else:
        neg_input_ids_padded = None
        neg_attention_mask_padded = None

    # Return padded inputs
    result = {
        "query_input_ids": pad_sequence(query_input_ids, batch_first=True, padding_value=0),
        "query_attention_mask": pad_sequence(query_attention_mask, batch_first=True, padding_value=0),
        "doc_input_ids": pad_sequence(doc_input_ids, batch_first=True, padding_value=0),
        "doc_attention_mask": pad_sequence(doc_attention_mask, batch_first=True, padding_value=0),
    }

    if neg_input_ids_padded is not None and neg_attention_mask_padded is not None:
        result["neg_input_ids"] = neg_input_ids_padded
        result["neg_attention_mask"] = neg_attention_mask_padded

    return result


def train():
    country = "ar"
    print(f"Training on {country} dataset...")
    ds = load_from_disk(f"messirve_{country}")
    train_ds = ds["train"]
    print("Dataset loaded")
    # ds.save_to_disk(f"messirve_{country}")
    # print("Dataset saved to disk")
    
    # read hard negatives from disk
    print("Loading hard negatives from pickle file...", end="")
    with open(f"hard_negatives_bge_{country}.pkl", "rb") as f:
        hard_negatives = pickle.load(f)
        print("Done")

    train_ds = preprocess_dataset(train_ds, hard_negatives)
    train_ds = train_ds.map(limit_hard_negatives)

    # save ds to disk
    train_ds.save_to_disk(f"messirve_train_{country}_hard_negatives")

    # # load train_df from disk
    # train_ds = load_from_disk(f"messirve_train_{country}_hard_negatives")

    train_ds = train_ds.select(range(5000))
    test_ds = ds["test"]

    checkpoint = "distilbert/distilbert-base-multilingual-cased"

    # model, tokenizer = get_mamba_model()
    model, tokenizer = get_auto_model(checkpoint)

    # Apply tokenization to the dataset
    train_ds = train_ds.map(lambda x: tokenize_with_hard_negatives(tokenizer, x), batched=True)
    test_ds = test_ds.map(lambda x: tokenize_function(tokenizer, x, append_eos=False), batched=True)

    # check if first token is CLS
    print(tokenizer.cls_token_id)
    print(tokenizer.cls_token)
    # check if first query instance in train_ds has CLS token
    print(train_ds[0]["query_input_ids"][0])

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
        evaluation_strategy="epoch",     # Evaluate at the end of each epoch
        learning_rate=1e-5,              # Learning rate
        per_device_train_batch_size=2,  # Batch size for training
        per_device_eval_batch_size=2,   # Batch size for evaluation
        num_train_epochs=3,              # Number of epochs
        weight_decay=0.01,               # Weight decay
        save_strategy="epoch",           # Save model checkpoints at the end of each epoch
        logging_dir="./logs",            # Directory for logs
        logging_steps=10,                # Log every 10 steps
        save_total_limit=1,              # Save only the last checkpoint
        remove_unused_columns=False,
        warmup_steps=500
    )

    # Create Trainer Instance
    trainer = InfoNCERetrievalTrainerHN(
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