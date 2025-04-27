import sys
print("Executable", sys.executable)

import os
import config.config as config

STORAGE_DIR = config.STORAGE_DIR
print(f"STORAGE_DIR: {STORAGE_DIR}")    # STORAGE_DIR: /media/discoexterno/leon/messirve-ir

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
# import library for timestamp
from transformers import EarlyStoppingCallback
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Makes only GPU1 visible

import logging
logging.basicConfig(
        format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
    )
logger = logging.getLogger(__name__)


def create_datasets(tokenizer,
                    train_queries, val_queries, test_queries,
                    pos_train_texts, pos_val_texts, pos_test_texts,
                    neg_train_texts, neg_val_texts, neg_test_texts,
                    max_length=256):
    # Tokenize and encode the text data
    def tokenize_data(queries, positive_docs, negative_docs):
        positive_encodings = tokenizer(queries, positive_docs, truncation=True, padding=True, max_length=max_length)
        negative_encodings = tokenizer(queries, negative_docs, truncation=True, padding=True, max_length=max_length)
        positive_encodings["labels"] = [1] * len(queries)
        negative_encodings["labels"] = [0] * len(queries)
        return positive_encodings, negative_encodings

    logger.info("Tokenizing and encoding train data...")
    pos_train_encodings, neg_train_encodings = tokenize_data(train_queries, pos_train_texts, neg_train_texts)
    logger.info("Tokenizing and encoding val data...")
    pos_val_encodings, neg_val_encodings = tokenize_data(val_queries, pos_val_texts, neg_val_texts)
    logger.info("Tokenizing and encoding test data...")
    pos_test_encodings, neg_test_encodings = tokenize_data(test_queries, pos_test_texts, neg_test_texts)

    # Function to merge and shuffle encodings
    def merge_and_shuffle(pos_encodings, neg_encodings):
        """Combines positive and negative encodings, then shuffles them."""
        combined_encodings = {key: pos_encodings[key] + neg_encodings[key] for key in pos_encodings.keys()}
        
        # Shuffle the dataset while keeping inputs and labels aligned
        combined_data = list(zip(combined_encodings["input_ids"], 
                                 combined_encodings["attention_mask"], 
                                 combined_encodings["labels"]))
        random.shuffle(combined_data)

        # Unzip the shuffled data
        shuffled_input_ids, shuffled_attention_mask, shuffled_labels = zip(*combined_data)

        # Convert back to dataset-friendly format
        return {
            "input_ids": list(shuffled_input_ids),
            "attention_mask": list(shuffled_attention_mask),
            "labels": list(shuffled_labels),
        }

    logger.info("Merging and shuffling datasets...")
    train_encodings = merge_and_shuffle(pos_train_encodings, neg_train_encodings)
    val_encodings = merge_and_shuffle(pos_val_encodings, neg_val_encodings)
    test_encodings = merge_and_shuffle(pos_test_encodings, neg_test_encodings)

    # Convert to Hugging Face Dataset format
    logger.info("Converting to Hugging Face Dataset...")
    train_dataset = Dataset.from_dict(train_encodings)
    val_dataset = Dataset.from_dict(val_encodings)
    test_dataset = Dataset.from_dict(test_encodings)
    logger.info("Done converting to HF Dataset.")

    return train_dataset, val_dataset, test_dataset       


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def training_arguments(hyperparameters):
    epochs = hyperparameters.get("epochs", 100)
    batch_size = hyperparameters.get("batch_size", 16)
    weight_decay = hyperparameters.get("weight_decay", 0.01)
    learning_rate = hyperparameters.get("learning_rate", 5e-6)
    warmup_ratio = hyperparameters.get("warmup_ratio", 0.1)
    metric = hyperparameters.get("metric_for_best_model", "f1")
    output_dir = hyperparameters.get("output_dir", "./results")

    print("Training arguments")
    print("Batch size:", batch_size)
    print("Weight decay:", weight_decay)
    print("Learning rate:", learning_rate)
    print("Warmup ratio:", warmup_ratio)

    training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=warmup_ratio,
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=40,
    save_strategy="steps",
    save_steps=40,
    save_total_limit=1,  # limit the number of saved checkpoints
    load_best_model_at_end=True,
    metric_for_best_model=metric,
    remove_unused_columns=False,  # Keep all columns
    )
    return training_args


def get_trainer(model, training_args, train_dataset, val_dataset, hyperparameters):
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=hyperparameters.get("early_stopping_patience", 4))]
    # callbacks=[LossDifferenceCallback(loss_diff_threshold=loss_diff_threshold)]
    )
    return trainer


def predict(trainer, test_dataset):
    predictions = trainer.predict(test_dataset)
    return predictions


def get_labels(predictions):
    test_pred_labels = np.argmax(predictions.predictions, axis=-1)
    print("Predicted Labels", test_pred_labels)
    return test_pred_labels

def run(model_checkpoint, num_labels,
        train_queries, val_queries, test_queries,
        pos_train_texts, pos_val_texts, pos_test_texts,
        neg_train_texts, neg_val_texts, neg_test_texts,
        hyperparameters=not None):
    
    max_length = hyperparameters.get("max_length", 256)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    logger.info("Tokenizing and encoding the data...")
    train_dataset, val_dataset, test_dataset = create_datasets(
        tokenizer,
        train_queries, val_queries, test_queries,
        pos_train_texts, pos_val_texts, pos_test_texts,
        neg_train_texts, neg_val_texts, neg_test_texts,
        max_length=max_length)
    logger.info("Done tokenizing and encoding the data.")

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    training_args = training_arguments(hyperparameters)
    trainer = get_trainer(model, training_args, train_dataset, val_dataset, hyperparameters)

    # Train the model
    trainer.train()

    # Evaluate the model
    predictions = predict(trainer, test_dataset)
    test_pred_labels = get_labels(predictions)
    
    # Generate and print the classification report
    y_test = test_dataset["labels"]
    if num_labels == 2:
        print(classification_report(y_test, test_pred_labels, target_names=['Class 0', 'Class 1']))
    elif num_labels == 3:
        print(classification_report(y_test, test_pred_labels, target_names=['Class 1', 'Class 2', 'Class 3']))
    else:
        pass
        print(classification_report(y_test, test_pred_labels, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3']))
    
    # save model
    model.save_pretrained(hyperparameters.get("output_dir", "./results"))
    tokenizer.save_pretrained(hyperparameters.get("output_dir", "./results"))
    print("Model saved to:", hyperparameters.get("output_dir", "./results"))
    return test_pred_labels


def convert_ds(ds):
    # Convert to a dict-of-lists to avoid repeated expensive indexing.
    data = ds.to_dict()
    queries = data["anchor"]
    positives = data["positive"]
    negatives = data["negative_1"]
    return queries, positives, negatives


def prepare_dataset(ds_path):
    # train_ds_path = os.path.join(ds_path, "messirve_train_ar_hard_negatives1_sbert")
    # test_ds_path = os.path.join(ds_path, "messirve_test_ar_hard_negatives1_sbert")
    train_ds_path = os.path.join(ds_path, "train_ds_sbert1")
    test_ds_path = os.path.join(ds_path, "test_ds_sbert1")

    train_ds = load_from_disk(train_ds_path)
    test_ds = load_from_disk(test_ds_path)

    train_queries, pos_train_texts, neg_train_texts = convert_ds(train_ds)
    test_queries, pos_test_texts, neg_test_texts = convert_ds(test_ds)

    return train_queries, pos_train_texts, neg_train_texts, test_queries, pos_test_texts, neg_test_texts


def main():
    # model_checkpoint = "distilbert/distilbert-base-multilingual-cased"
    # model_checkpoint = "FacebookAI/xlm-roberta-base"
    model_checkpoint = "mrm8488/legal-longformer-base-8192-spanish"

    ds_path = os.path.join(STORAGE_DIR, "legal_ir", "data")
    train_queries, pos_train_texts, neg_train_texts, test_queries, pos_test_texts, neg_test_texts = prepare_dataset(ds_path)

    # # split the training data into training and validation sets
    # train_queries, val_queries, pos_train_texts, pos_val_texts, neg_train_texts, neg_val_texts = train_test_split(
    #     train_queries, pos_train_texts, neg_train_texts, test_size=0.2, random_state=42
    # )

    hyperparameters = {
        'epochs': 7,
        'batch_size': 8,
        'weight_decay': 0.01,
        'learning_rate': 2e-5,
        'warmup_ratio': 0.1,
        'metric_for_best_model': "f1",
        'early_stopping_patience': 6,
        'max_length': 2048,
        'output_dir': os.path.join(STORAGE_DIR, "legal_ir", "results", "cross_encoder_2048"),
    }

    val_queries = test_queries
    pos_val_texts = pos_test_texts
    neg_val_texts = neg_test_texts

    test_pred_labels = run(
        model_checkpoint, 2,
        train_queries, val_queries, test_queries,
        pos_train_texts, pos_val_texts, pos_test_texts,
        neg_train_texts, neg_val_texts, neg_test_texts,
        hyperparameters=hyperparameters
    )


if __name__ == "__main__":
    main()