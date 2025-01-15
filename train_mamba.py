from models.model_setup import get_mamba_model, get_auto_model
from datasets import load_dataset
from trainers.info_nce_trainer import InfoNCERetrievalTrainer
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import torch
from torch.nn.utils.rnn import pad_sequence


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
    # Pad query input IDs
    query_input_ids = [torch.tensor(example["query_input_ids"]) for example in batch]
    query_attention_mask = [torch.tensor(example["query_attention_mask"]) for example in batch]

    # Pad document input IDs
    doc_input_ids = [torch.tensor(example["doc_input_ids"]) for example in batch]
    doc_attention_mask = [torch.tensor(example["doc_attention_mask"]) for example in batch]

    return {
        "query_input_ids": pad_sequence(query_input_ids, batch_first=True, padding_value=0),  # Pad to the max length in batch
        "query_attention_mask": pad_sequence(query_attention_mask, batch_first=True, padding_value=0),
        "doc_input_ids": pad_sequence(doc_input_ids, batch_first=True, padding_value=0),
        "doc_attention_mask": pad_sequence(doc_attention_mask, batch_first=True, padding_value=0),
    }


def train():
    country = "ar"
    ds = load_dataset("spanish-ir/messirve", country)

    train_ds = ds["train"]
    test_ds = ds["test"]

    checkpoint = "distilbert/distilbert-base-multilingual-cased"

    # model, tokenizer = get_mamba_model()
    model, tokenizer = get_auto_model(checkpoint)

    # Apply tokenization to the dataset
    train_ds = train_ds.map(lambda x: tokenize_function(tokenizer, x, append_eos=False), batched=True)
    test_ds = test_ds.map(lambda x: tokenize_function(tokenizer, x, append_eos=False), batched=True)

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
        learning_rate=2e-5,              # Learning rate
        per_device_train_batch_size=8,  # Batch size for training
        per_device_eval_batch_size=8,   # Batch size for evaluation
        num_train_epochs=3,              # Number of epochs
        weight_decay=0.01,               # Weight decay
        save_strategy="epoch",           # Save model checkpoints at the end of each epoch
        logging_dir="./logs",            # Directory for logs
        logging_steps=10,                # Log every 10 steps
        save_total_limit=1,              # Save only the last checkpoint
        remove_unused_columns=False
    )

    # Create Trainer Instance
    trainer = InfoNCERetrievalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,     # Raw dataset for training
        eval_dataset=test_ds,       # Raw dataset for evaluation
        tokenizer=tokenizer,             # Tokenizer for on-the-fly tokenization
        data_collator=custom_data_collator,     # Handles dynamic padding and tokenization
    )

    # Train the Model
    trainer.train()

    # Evaluate the Model
    metrics = trainer.evaluate()
    print(metrics)

    # Save the Model
    trainer.save_model("./saved_model")


if __name__ == "__main__":
    train()