import sys
print("Executable", sys.executable)

import os

STORAGE_DIR = os.getenv("STORAGE_DIR")
print(f"STORAGE_DIR: {STORAGE_DIR}")    # STORAGE_DIR: /media/discoexterno/leon/messirve-ir

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerControl, TrainerState
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
# import library for timestamp
from datetime import datetime
from transformers import EarlyStoppingCallback
import torch
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Makes only GPU1 visible

import logging
logging.basicConfig(
        format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
    )
logger = logging.getLogger(__name__)
#### /print debug information to stdout

import torch.nn as nn
import torch.nn.functional as F


class WeightedSmoothCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights, smoothing=0.1):
        super(WeightedSmoothCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        n_classes = inputs.size(1)
        assert len(self.class_weights) == n_classes, "Class weights should match number of classes"

        # Create the smoothed label
        targets = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.smoothing) * targets + self.smoothing / n_classes
        
        # Apply class weights
        if self.class_weights is not None:
            class_weights = torch.tensor(self.class_weights, dtype=inputs.dtype, device=inputs.device)
            targets = targets * class_weights.unsqueeze(0)
        
        # Calculate weighted smooth loss
        loss = F.log_softmax(inputs, dim=1).mul(targets).sum(dim=1).mean() * -1
        return loss


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight  # class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class WeightedAutoModel(AutoModelForSequenceClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.class_weights = class_weights
        self.focal_loss = FocalLoss(weight=self.class_weights, gamma=2.0)
        # self.criterion = FocalLoss(alpha=0.25, gamma=2.0)

    def compute_loss_cross_ent(self, model_output, labels):
        # model_output: tuple of (logits, ...)
        logits = model_output[0]
        # Assuming using CrossEntropyLoss, adjust accordingly if using a different loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        return loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
    def compute_loss(self, model_output, labels):
        logits = model_output[0]
        loss_fct = WeightedSmoothCrossEntropyLoss(class_weights=self.class_weights)
        smoothed_loss = loss_fct(logits, labels)
        cross_ent_loss = self.compute_loss_cross_ent(model_output, labels)
        focal_loss = self.focal_loss(logits, labels)
        return (smoothed_loss + cross_ent_loss + focal_loss)/3

    # def compute_loss_combined(self, model_output, labels):
    #     cross_ent_lost = self.compute_loss_cross_ent(model_output, labels)
    #     logits = model_output[0]
    #     focal_loss = self.focal_loss(logits, labels)
    #     return (cross_ent_lost + focal_loss)/2

    # def compute_loss_weighed_focal(self, model_output, labels):
    #     # model_output: tuple of (logits, ...)
    #     logits = model_output[0]
    #     # Assuming using FocalLoss, adjust accordingly if using a different loss
    #     loss_fct = FocalLoss()
    #     return loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


def create_datasets(tokenizer, train_texts, val_texts, test_texts, y_train, y_val, y_test=None, max_length=256):
    # Tokenize and encode the text data
    def tokenize_data(texts, labels=None):
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All elements in 'texts' must be strings.")
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        if labels is not None:
            encodings["labels"] = labels
        return encodings

    logger.info("Tokenizing and encoding train data...")
    train_encodings = tokenize_data(train_texts, y_train)
    logger.info("Tokenizing and encoding val data...")
    val_encodings = tokenize_data(val_texts, y_val)
    logger.info("Tokenizing and encoding test data...")
    test_encodings = tokenize_data(test_texts, y_test)

    # Convert to Hugging Face Dataset
    logger.info("Converting to Hugging Face Dataset...")
    train_dataset = Dataset.from_dict(train_encodings)
    val_dataset = Dataset.from_dict(val_encodings)
    test_dataset = Dataset.from_dict(test_encodings)
    logger.info("Done converting to HF Dataset.")

    return train_dataset, val_dataset, test_dataset        


def load_model(model_checkpoint, num_labels, classes, y_train):
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    # class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    # class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    # weighted_model = WeightedAutoModel.from_pretrained(model_checkpoint, num_labels=num_labels)
    # weighted_model.class_weights = class_weights_tensor
    # # weighted_model.load_state_dict(model.state_dict())  # Copy the weights from the original model
    # # Wrap the model with DataParallel to use multiple GPUs
    # # if torch.cuda.device_count() > 1:
    # #     print(f"Using {torch.cuda.device_count()} GPUs!")
    # #     model = CustomDataParallel(model)
    # # model.cuda()  # Ensure the model is on the correct device
    # return weighted_model
    return model


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
    warmup_steps = hyperparameters.get("warmup_steps", 1000)
    metric = hyperparameters.get("metric_for_best_model", "f1")

    print("Training arguments")
    print("Batch size:", batch_size)
    print("Weight decay:", weight_decay)
    print("Learning rate:", learning_rate)
    print("Warmup steps:", warmup_steps)

    training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=warmup_steps,
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    logging_dir='./logs',
    logging_steps=20,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
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
        train_texts, val_texts, test_texts,
        y_train, y_val, y_test=None,
        hyperparameters=not None):
    
    max_length = hyperparameters.get("max_length", 256)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    logger.info("Tokenizing and encoding the data...")
    train_dataset, val_dataset, test_dataset = create_datasets(tokenizer, train_texts, val_texts, test_texts, y_train, y_val, y_test=None, max_length=max_length)
    logger.info("Done tokenizing and encoding the data.")
    classes = np.unique(y_train)
    model = load_model(model_checkpoint, num_labels, classes, y_train)
    training_args = training_arguments(hyperparameters)
    trainer = get_trainer(model, training_args, train_dataset, val_dataset, hyperparameters)
    # Train the model
    trainer.train()
    predictions = predict(trainer, test_dataset)
    test_pred_labels = get_labels(predictions)
    # Generate and print the classification report
    if num_labels == 2:
        print(classification_report(y_test, test_pred_labels, target_names=['Class 0', 'Class 1']))
    elif num_labels == 3:
        print(classification_report(y_test, test_pred_labels, target_names=['Class 1', 'Class 2', 'Class 3']))
    else:
        pass
        print(classification_report(y_test, test_pred_labels, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3']))
    return test_pred_labels


def convert_ds(ds, num_negatives, sep_token):
    """
    Convert a HuggingFace Dataset into a list of InputExample objects.

    Each row is expected to have an "anchor", a "positive", and negative columns
    named "negative_1", ..., "negative_N" where N equals num_negatives.

    Parameters
    ----------
    ds : datasets.Dataset
        HuggingFace Dataset with columns "anchor", "positive", and negative columns.
    num_negatives : int
        Number of negative examples per sample.

    Returns
    -------
    list
        List of InputExample objects. Each row produces one positive sample and
        num_negatives negative samples.
    """
    # Convert to a dict-of-lists to avoid repeated expensive indexing.
    data = ds.to_dict()
    anchors = data["anchor"]
    positives = data["positive"]
    negatives_lists = [data[f"negative_{j}"] for j in range(1, num_negatives + 1)]

    texts = []
    labels = []
    # Zip through the columns
    for anchor, positive, *negatives in tqdm(zip(anchors, positives, *negatives_lists), total=len(anchors), desc="Converting ds"):
        texts.append(f"{anchor}{sep_token}{positive}")
        labels.append(1)
        for neg in negatives:
            texts.append(f"{anchor}{sep_token}{neg}")
            labels.append(0)
    return texts, labels


def prepare_dataset(ds_path, sep_token):
    train_ds_path = os.path.join(ds_path, "messirve_train_ar_hard_negatives_sbert")
    test_ds_path = os.path.join(ds_path, "messirve_test_ar_hard_negatives_sbert")

    train_ds = load_from_disk(train_ds_path)
    test_ds = load_from_disk(test_ds_path)

    num_negatives = 5

    train_texts, y_train = convert_ds(train_ds, num_negatives, sep_token)
    test_texts, y_test = convert_ds(test_ds, num_negatives, sep_token)

    return train_texts, y_train, test_texts, y_test


def get_sep_token(model_checkpoint: str) -> str:
    if 'distilbert' in model_checkpoint or 'roberta-base' in model_checkpoint:
        sep_token = '[SEP]'
    elif 'xlnet' in model_checkpoint:
        sep_token = '<sep>'
    elif 'xlm-roberta-large' in model_checkpoint:
        sep_token = '</s>'
    return sep_token


def main():
    # model_checkpoint = "distilbert/distilbert-base-multilingual-cased"
    model_checkpoint = "FacebookAI/xlm-roberta-base"
    sep_token = get_sep_token(model_checkpoint)

    ds_path = os.path.join(STORAGE_DIR, "datasets")
    train_texts, y_train, test_texts, y_test = prepare_dataset(ds_path, sep_token)

    train_texts, val_texts, y_train, y_val = train_test_split(
        train_texts, y_train,
        test_size=0.1, random_state=42
    )

    hyperparameters = {
        'epochs': 1,
        'batch_size': 8,
        'weight_decay': 0.01,
        'learning_rate': 2e-5,
        'warmup_steps': 1000,
        'metric_for_best_model': "f1",
        'early_stopping_patience': 4,
        'max_length': 512,
    }

    test_pred_labels = run(
        model_checkpoint, 2,
        train_texts, val_texts, test_texts,
        y_train, y_val, y_test,
        hyperparameters=hyperparameters
    )


if __name__ == "__main__":
    main()