import os
import random
import logging
import numpy as np
import sys
from datasets import Dataset, load_from_disk
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          Trainer,
                          TrainingArguments,
                          EarlyStoppingCallback)
from sklearn.metrics import (accuracy_score,
                             precision_recall_fscore_support,
                             classification_report)
from config.config import STORAGE_DIR
from src.utils.log_experiment import log_csv, log_md, log_plot
import uuid
import torch
import torch.nn as nn
import pandas as pd
import json
from src.eval_class import Evaluator


class CrossEncoderTrainer:
    """
    A class for training and evaluating a cross-encoder model for sequence classification
    with a variable number of labels.

    Parameters
    ----------
    model_checkpoint : str
        Identifier for the pretrained model checkpoint.
    num_labels : int
        The number of classification labels.
    hyperparameters : dict
        A dictionary containing hyperparameter settings such as epochs, batch_size,
        learning_rate, warmup_ratio, early_stopping_patience, max_length, and output_dir.

    Attributes
    ----------
    tokenizer : AutoTokenizer
        The tokenizer loaded from the pretrained model.
    model : AutoModelForSequenceClassification
        The sequence classification model.
    training_args : TrainingArguments
        The training arguments for the Trainer.
    trainer : Trainer or None
        The Hugging Face Trainer instance (set up after datasets are prepared).
    logger : logging.Logger
        Logger for info messages.
    """

    def __init__(self, model_checkpoint, num_labels, hyperparameters):
        self.model_checkpoint = model_checkpoint
        self.num_labels = num_labels
        self.hyperparameters = hyperparameters
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint, num_labels=num_labels
        )
        self.training_args = self._get_training_arguments()
        self.trainer = None
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO
        )

    def _get_training_arguments(self):
        """
        Create training arguments from hyperparameters.

        Returns
        -------
        TrainingArguments
            The training arguments object.
        """
        epoch_batches = 2648
        epoch_steps = epoch_batches // self.hyperparameters["batch_size"]

        return TrainingArguments(
            output_dir=self.hyperparameters.get("output_dir", "./results"),
            num_train_epochs=self.hyperparameters.get("epochs", 100),
            per_device_train_batch_size=self.hyperparameters.get("batch_size", 16),
            per_device_eval_batch_size=2,
            warmup_ratio=self.hyperparameters.get("warmup_ratio", 0.1),
            weight_decay=self.hyperparameters.get("weight_decay", 0.01),
            learning_rate=self.hyperparameters.get("learning_rate", 5e-6),
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=epoch_steps//10,
            save_strategy="steps",
            save_steps=epoch_steps//10,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model=self.hyperparameters.get("metric_for_best_model", "f1"),
            remove_unused_columns=False,
            prediction_loss_only=False,
            gradient_accumulation_steps=self.hyperparameters.get("gradient_accumulation_steps", 1),
            eval_accumulation_steps=2,
        )
    
    @staticmethod
    def compute_pairwise_accuracy(eval_pred):
        """
        Computes pairwise accuracy for a RankNet model.
        
        Parameters
        ----------
        eval_pred : tuple
            A tuple (preds, labels), where:
            - preds is an array of shape (batch_size,) = s^+ - s^- 
            - labels is an array of shape (batch_size,) in {0, 1},
                where 1 means pos doc is more relevant than neg doc,
                0 means otherwise (should never happen in normal usage, but can handle it).
        
        Returns
        -------
        dict
            Dictionary containing 'pairwise_accuracy'.
        """
        preds, labels = eval_pred
        # Ensure tensors are detached and moved to CPU before converting to NumPy
        preds = preds.detach().cpu().numpy() if hasattr(preds, "detach") else preds
        labels = labels.detach().cpu().numpy() if hasattr(labels, "detach") else labels

        # Predicted label = 1 if s^+ - s^- > 0
        pred_labels = (preds > 0).astype(int)
        correct = (pred_labels == labels).sum()
        total = len(labels)
        pairwise_acc = correct / total if total > 0 else 0.0

        return {"pairwise_accuracy": pairwise_acc}

    @staticmethod
    def compute_metrics(eval_pred):
        """
        Compute evaluation metrics from logits and labels.

        Parameters
        ----------
        eval_pred : tuple
            A tuple (logits, labels).

        Returns
        -------
        dict
            Dictionary containing accuracy, f1, precision, and recall scores.
        """
        logits, labels = eval_pred
        # If `logits` is a torch tensor:
        if hasattr(logits, "detach"):
            logits = logits.detach().cpu().numpy()
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="macro"
        )
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
    
    def set_dataset_dicts(self, corpus_path):
        if corpus_path.endswith(".csv"):
            # open corpus_py.csv with pandas
            corpus = pd.read_csv(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus_py.csv"), usecols=["Codigo", "text"])

            docid_to_text = dict(zip(corpus["Codigo"], corpus["text"]))
        elif corpus_path.endswith(".json"):
            # Load corpus from JSON file.
            with open(corpus_path, 'r', encoding='utf-8') as f:
                docid_to_text = json.load(f)
            
        self.doc_dict = docid_to_text

        # open queries_57.csv with pandas
        queries = pd.read_csv(os.path.join(STORAGE_DIR, "legal_ir", "data", "queries_57.csv"), usecols=["topic_id", "Query"])

        # create a dictionary with topic_id as key and query as value
        qid_to_query = dict(zip(queries["topic_id"], queries["Query"]))
        self.query_dict = qid_to_query

    def load_and_split_dataset(self, ds_path, test_size=0.2, val_size=0.1, max_length=None):
        """
        Load a dataset from disk that contains only IDs, map these IDs to texts using
        the trainer's dictionaries, split the dataset into training, validation, and test sets,
        and tokenize each example.

        The input dataset is expected to have at least the following keys:
        - "query_id": identifier for the query.
        - "doc_id": identifier for the document.
        - "label": the relevance label.

        The function uses `self.query_dict` and `self.doc_dict` to convert IDs to actual texts,
        tokenizes the (query, doc) pair using the trainer's tokenizer, and attaches the label.

        Parameters
        ----------
        ds_path : str
            The path to the dataset on disk (a Hugging Face Dataset saved via `save_to_disk`).
        test_size : float, optional
            Fraction of the data to use as the test set (default is 0.2).
        val_size : float, optional
            Fraction of the training data (after splitting off test) to use as the validation set
            (default is 0.1).
        max_length : int, optional
            Maximum sequence length for tokenization. If None, the value from hyperparameters is used.

        Returns
        -------
        tuple
            A tuple of tokenized Hugging Face Datasets in the order:
            (train_dataset, validation_dataset, test_dataset).
            Each dataset contains the following keys:
            - "input_ids": token IDs for the (query, document) pair.
            - "attention_mask": attention mask.
            - "labels": the corresponding relevance label.
        """
        max_length = max_length or self.hyperparameters.get("max_length", 256)
        ds = load_from_disk(ds_path)
        
        # Map IDs to texts using self.query_dict and self.doc_dict
        def map_ids(example):
            example["query"] = self.query_dict.get(example["query_id"], "")
            example["doc"] = self.doc_dict.get(example["doc_id"], "")
            return example

        ds = ds.map(map_ids)
        
        # Split into train+val and test
        split_ds = ds.train_test_split(test_size=test_size, shuffle=True, seed=42)
        test_ds = split_ds["test"]
        train_val_ds = split_ds["train"]

        # Further split train_val into train and validation sets
        split_train_val = train_val_ds.train_test_split(test_size=val_size, shuffle=True, seed=42)
        train_ds = split_train_val["train"]
        val_ds = split_train_val["test"]

        def tokenize_fn(examples):
            encodings = self.tokenizer(
                examples["query"],
                examples["doc"],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
            encodings["labels"] = examples["label"]
            return encodings

        train_ds = train_ds.map(tokenize_fn, batched=True)
        val_ds = val_ds.map(tokenize_fn, batched=True)
        test_ds = test_ds.map(tokenize_fn, batched=True)

        return train_ds, val_ds, test_ds
    
    def prepare_ranknet_ds(self, ds_path, test_size=0.2, val_size=0.1, max_length=2048):
        """
        Prepare the dataset from triplets that contain only query and document IDs.
        It converts IDs to texts using self.query_dict and self.doc_dict, and then tokenizes
        the resulting query, positive document, and negative document.

        Parameters
        ----------
        ds_path : path to Dataset
            A Hugging Face Dataset with keys:
            "query_id", "pos_doc_id", "neg_doc_id", "pos_label", "neg_label".
        max_length : int, optional
            Maximum sequence length for tokenization. Default is 256.

        Returns
        -------
        Dataset
            A tokenized Dataset with keys:
            "input_ids_pos", "attention_mask_pos",
            "input_ids_neg", "attention_mask_neg",
            "pos_label", "neg_label".
        """
        # Map IDs to texts.
        def map_ids(example):
            pos_text = self.doc_dict[str(example["pos_doc_id"])]
            neg_text = self.doc_dict[str(example["neg_doc_id"])]
            if pos_text == neg_text:
                print(f"Identical texts found for query_id {example['query_id']}: pos_doc_id {example['pos_doc_id']} equals neg_doc_id {example['neg_doc_id']}")
            example["query"] = self.query_dict.get(example["query_id"], "")
            example["pos_doc"] = pos_text
            example["neg_doc"] = neg_text
            return example

        ds = load_from_disk(ds_path)
        ds = ds.map(map_ids)
        
        # Split into train+val and test
        split_ds = ds.train_test_split(test_size=test_size, shuffle=True, seed=42)
        test_ds = split_ds["test"]
        train_val_ds = split_ds["train"]

        # Further split train_val into train and validation sets
        split_train_val = train_val_ds.train_test_split(test_size=val_size, shuffle=True, seed=42)
        train_ds = split_train_val["train"]
        val_ds = split_train_val["test"]
        
        # Tokenize the query with positive and negative documents.
        def tokenize_fn(examples):
            pos_enc = self.tokenizer(
                examples["query"],
                examples["pos_doc"],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
            neg_enc = self.tokenizer(
                examples["query"],
                examples["neg_doc"],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
            return {
                "input_ids_pos": pos_enc["input_ids"],
                "attention_mask_pos": pos_enc["attention_mask"],
                "input_ids_neg": neg_enc["input_ids"],
                "attention_mask_neg": neg_enc["attention_mask"],
                "pos_label": examples["pos_label"],
                "neg_label": examples["neg_label"]
            }

        train_ds = train_ds.map(tokenize_fn, batched=True)
        val_ds = val_ds.map(tokenize_fn, batched=True)
        test_ds = test_ds.map(tokenize_fn, batched=True)

        # Remove unnecessary columns
        train_ds = train_ds.remove_columns(["query_id", "pos_doc_id", "neg_doc_id", "query", "pos_doc", "neg_doc"])
        val_ds = val_ds.remove_columns(["query_id", "pos_doc_id", "neg_doc_id", "query", "pos_doc", "neg_doc"])
        test_ds = test_ds.remove_columns(["query_id", "pos_doc_id", "neg_doc_id", "query", "pos_doc", "neg_doc"])

        return train_ds, val_ds, test_ds

    def compute_class_weights_from_dataset(self, train_dataset):
        """
        Compute class weights automatically based on training dataset labels.

        Parameters
        ----------
        train_dataset : Dataset
            The training dataset.

        Returns
        -------
        list
            List of computed class weights.
        """
        import numpy as np
        all_labels = [example["labels"] for example in train_dataset]
        counts = np.bincount(all_labels, minlength=self.num_labels)
        total = np.sum(counts)
        weights = total / (self.num_labels * counts)
        return weights.tolist()

    def setup_trainer(self, train_dataset, val_dataset):
        """
        Setup the Hugging Face Trainer, automatically computing class weights if not provided,
        and using a weighted loss function.
        """
        # At the beginning of setup_trainer, add:
        hp = self.hyperparameters
        
        # Automatically compute class weights if not provided
        if hp.get("loss_type") == "weighted" or hp.get("loss_type") == "focal":
            if "class_weights" not in self.hyperparameters:
                computed_weights = self.compute_class_weights_from_dataset(train_dataset)
                self.hyperparameters["class_weights"] = computed_weights

            # Convert class weights to a tensor on the model's device.
            class_weights = self.hyperparameters["class_weights"]
            class_weights_tensor = torch.tensor(class_weights, device=self.model.device, dtype=torch.float)

            # Define a custom Trainer subclass that overrides compute_loss
            class WeightedLossTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    """
                    Compute the weighted (or focal) cross-entropy loss, filtering out extra keys from inputs.
                    
                    Parameters
                    ----------
                    model : torch.nn.Module
                        The model to train.
                    inputs : dict
                        Dictionary containing model inputs. Expected keys include:
                            - "input_ids", "attention_mask", "labels", etc.
                        Extra keys such as "query_id", "doc_id", "query", "doc", "pos_doc_id", "neg_doc_id"
                        are removed here.
                    return_outputs : bool, optional
                        If True, returns a tuple (loss, outputs), otherwise returns loss.
                    **kwargs : dict
                        Additional keyword arguments.
                        
                    Returns
                    -------
                    torch.Tensor or (torch.Tensor, dict)
                        The computed loss, with optional model outputs.
                    """
                    # Remove extra keys that the model's forward method does not expect.
                    for key in ["query_id", "doc_id", "query", "doc"]:
                        inputs.pop(key, None)
                    
                    labels = inputs.get("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    num_labels = model.config.num_labels if hasattr(model, "config") else model.module.config.num_labels
                    device = logits.device
                    loss_type = hp.get("loss_type", "weighted")
                    
                    if loss_type == "focal":
                        focal_gamma = hp.get("focal_gamma", 2.0)
                        # Always use computed class weights as alpha
                        alpha = class_weights_tensor.to(device)
                        ce_loss = nn.functional.cross_entropy(
                            logits.view(-1, num_labels),
                            labels.view(-1),
                            reduction='none',
                            weight=alpha
                        )
                        pt = torch.exp(-ce_loss)
                        loss = ((1 - pt) ** focal_gamma * ce_loss).mean()
                    else:
                        weight = class_weights_tensor.to(device)
                        loss_fct = nn.CrossEntropyLoss(weight=weight)
                        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                    
                    return (loss, outputs) if return_outputs else loss


            TrainerClass = WeightedLossTrainer

        elif hp.get("loss_type") == "ranknet":
            class RankNetTrainer(Trainer):
                """
                A Trainer subclass that implements the RankNet pairwise loss.
                """

                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    """
                    Compute the RankNet pairwise loss.

                    Parameters
                    ----------
                    model : torch.nn.Module
                        The model to train.
                    inputs : dict
                        A dictionary containing the batch data. Must include:
                        - "input_ids_pos": token IDs for the positive docs
                        - "input_ids_neg": token IDs for the negative docs
                        - "attention_mask_pos": attention mask for the positive docs
                        - "attention_mask_neg": attention mask for the negative docs
                        (Adjust these names as needed if your dataset is organized differently.)
                    return_outputs : bool
                        Whether to return (loss, outputs) or just loss.
                    **kwargs : dict
                        Additional arguments.

                    Returns
                    -------
                    torch.Tensor or (torch.Tensor, dict)
                        The RankNet loss, with optional model outputs if return_outputs=True.
                    """
                    pos_label = inputs.pop("pos_label")
                    neg_label = inputs.pop("neg_label")

                    # Check that positive and negative input_ids are different.
                    diff_mask = (inputs["input_ids_pos"] != inputs["input_ids_neg"]).any(dim=1)
                    if not diff_mask.all():
                        raise ValueError("Some positive and negative input_ids are identical!")

                    # 1) Forward pass on positive docs
                    pos_output = model(
                        input_ids=inputs["input_ids_pos"],
                        attention_mask=inputs["attention_mask_pos"]
                    )
                    # 2) Forward pass on negative docs
                    neg_output = model(
                        input_ids=inputs["input_ids_neg"],
                        attention_mask=inputs["attention_mask_neg"]
                    )

                    # 3) Extract the logits; shape [batch_size, num_labels].
                    #    For a cross-encoder classification model, you might have multiple logits,
                    #    so select the relevant dimension or reduce to a single score per doc.
                    #    Suppose we take the first (or only) logit as the ranking score:
                    pos_scores = pos_output.logits[:, 0]
                    neg_scores = neg_output.logits[:, 0]

                    # 4) Compute the pairwise RankNet loss:
                    #    Loss_i = log(1 + exp( - sigma * (pos_score_i - neg_score_i) ))
                    sigma = kwargs.get("ranknet_sigma", 1.0)  # Or store as a class attr.
                    score_diff = pos_scores - neg_scores
                    ranknet_loss = torch.log1p(torch.exp(-sigma * score_diff)).mean()

                    if return_outputs:
                        return (ranknet_loss, (pos_output, neg_output))
                    return ranknet_loss
                
                def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
                    with torch.no_grad():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                        pos_out, neg_out = outputs

                        # Compute score difference as a tensor.
                        score_diff_tensor = pos_out.logits[:, 0] - neg_out.logits[:, 0]
                        # Create labels tensor with the same shape.
                        labels = torch.ones(score_diff_tensor.shape, dtype=torch.int32, device=score_diff_tensor.device)
                    
                    if prediction_loss_only:
                        return (loss, None, None)
                    
                    # Return tensors directly.
                    return (loss, score_diff_tensor.detach(), labels)


            
            TrainerClass = RankNetTrainer

        elif hp.get("loss_type") == "graded ranknet":
            class GradedRankNetTrainer(Trainer):
                """
                A Trainer subclass that implements a graded RankNet loss.
                
                The loss is defined as:
                
                    Loss = |pos_label - neg_label| * log(1 + exp(-sigma * (s_pos - s_neg)))
                
                where s_pos and s_neg are the scores for the positive and negative documents,
                and sigma is a hyperparameter (default 1.0).
                
                The input batch is expected to have:
                - "input_ids_pos", "attention_mask_pos" for the positive document.
                - "input_ids_neg", "attention_mask_neg" for the negative document.
                - "pos_label" and "neg_label" (numeric tensors).
                """
                
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    pos_label = inputs.pop("pos_label")
                    neg_label = inputs.pop("neg_label")
                    # Forward pass for positive and negative examples.
                    pos_output = model(
                        input_ids=inputs["input_ids_pos"],
                        attention_mask=inputs["attention_mask_pos"]
                    )
                    neg_output = model(
                        input_ids=inputs["input_ids_neg"],
                        attention_mask=inputs["attention_mask_neg"]
                    )
                    
                    # Extract ranking scores (assumes the score is the first logit)
                    pos_scores = pos_output.logits[:, 0]
                    neg_scores = neg_output.logits[:, 0]
                    
                    # Get sigma from kwargs (default 1.0)
                    sigma = kwargs.get("ranknet_sigma", 1.0)
                    
                    # Compute the score difference
                    score_diff = pos_scores - neg_scores
                    
                    # Get the graded labels and compute the absolute difference
                    pos_label = pos_label.float()
                    neg_label = neg_label.float()
                    grade_diff = torch.abs(pos_label - neg_label)
                    
                    # Compute the weighted RankNet loss for each pair
                    loss = (grade_diff * torch.log1p(torch.exp(-sigma * score_diff))).mean()
                    
                    return (loss, (pos_output, neg_output)) if return_outputs else loss
                
                def prediction_step(self, model, inputs, prediction_loss_only=True, ignore_keys=None):
                    """
                    Uses the same pairwise logic for evaluation by calling `compute_loss`.
                    """
                    # Call compute_loss with return_outputs=True
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

                    if prediction_loss_only:
                        return (loss, None, None)

                    # The outputs are (pos_out, neg_out). Let's produce some dummy predictions:
                    pos_out, neg_out = outputs
                    # For logging, we can treat the difference in scores as 'preds':
                    preds = (pos_out.logits[:, 0] - neg_out.logits[:, 0]).detach()
                    # And the difference in labels (if you want) as 'labels'
                    # or just set to None if you don't want them.
                    labs = None

                    return (loss, preds, labs)
                
            TrainerClass = GradedRankNetTrainer

        else:
            TrainerClass = Trainer

        self.trainer = TrainerClass(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            # compute_metrics=self.compute_metrics,
            compute_metrics=self.compute_pairwise_accuracy,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.hyperparameters.get("early_stopping_patience", 4)
                )
            ]
        )

    def train(self):
        """
        Train the model using the Trainer.
        """
        if self.trainer is None:
            raise ValueError(
                "Trainer not set up. Call setup_trainer() with datasets before training."
            )
        self.logger.info("Starting training...")
        self.trainer.train()

    def evaluate_classification(self, test_dataset):
        """
        Evaluate the model on a test dataset and print a classification report.

        Parameters
        ----------
        test_dataset : Dataset
            The test dataset.

        Returns
        -------
        dict
            Evaluation results from the Trainer.
        """
        self.logger.info("Evaluating on test dataset...")
        predictions = self.trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=-1)
        target_names = [f"Class {i}" for i in range(self.num_labels)]
        report = classification_report(test_dataset["labels"], preds, target_names=target_names)
        print(report)
        return predictions

    def evaluate(self, test_dataset):
        """
        Evaluate the model on a test dataset and print evaluation metrics.
        For a RankNet cross-encoder, we compute pairwise accuracy (the percentage of
        document pairs in which the positive is scored higher than the negative).
        
        Parameters
        ----------
        test_dataset : Dataset
            The test dataset.
        
        Returns
        -------
        dict
            Evaluation metrics as returned by the Trainer.
        """
        self.logger.info("Evaluating on test dataset...")
        # Use the Trainer's evaluate, which will call our compute_metrics (e.g. compute_pairwise_accuracy)
        metrics = self.trainer.evaluate(test_dataset)
        print("Evaluation metrics:", metrics)
        return metrics

    def predict(self, dataset):
        """
        Make predictions on a given dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset on which to perform predictions.

        Returns
        -------
        np.ndarray
            Array of predicted labels.
        """
        predictions = self.trainer.predict(dataset)
        preds = np.argmax(predictions.predictions, axis=-1)
        return preds

    def save(self):
        """
        Save the trained model and tokenizer to the specified output directory.
        """
        output_dir = self.hyperparameters.get("output_dir", "./results")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("Model and tokenizer saved to:", output_dir)

    def log_experiment(self, exp_id, dataset_name, loss_name, gpu_name, ir_metrics):
        """
        Log experiment details using Markdown, CSV, and plot logging.

        Parameters
        ----------
        exp_id : str
            Experiment identifier.
        dataset_name : str
            Name of the dataset used.
        loss_name : str
            Name of the loss function used.
        gpu_name : str
            Name of the GPU used.
        ir_metrics : dict
            Dictionary containing IR metrics (e.g., nDCG, Recall, MRR).
        """
        # Retrieve training results from trainer state
        training_results = self.trainer.state.log_history
        # Convert training arguments to a dictionary
        training_args_dict = self.training_args.to_dict()
        # Determine experiment directory from output_dir
        exp_dir = training_args_dict.get("output_dir", "./results")
        model_name = self.model_checkpoint

        # Log Markdown file
        log_md(exp_dir, exp_id, model_name, dataset_name, loss_name,
               training_args_dict, gpu_name, training_results, ir_metrics)
        # Log CSV file
        log_csv(exp_id, model_name, dataset_name, loss_name,
                training_args_dict, gpu_name, training_results, ir_metrics)
        # Log training plot
        log_plot(exp_dir, exp_id, training_results)


def main():
    """
    Main function to load the dataset from the hub, split it, and run training and evaluation.
    """
    model_checkpoint = "mrm8488/legal-longformer-base-8192-spanish"
    # model_checkpoint = "joelniklaus/legal-xlm-roberta-base"
    hyperparameters = {
        "epochs": 1,
        "batch_size": 4,
        "weight_decay": 0.01,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "metric_for_best_model": "eval_pairwise_accuracy",
        "early_stopping_patience": 6,
        "max_length": 2048,
        'output_dir': os.path.join(STORAGE_DIR, "legal_ir", "results", "cross_encoder_fine_2048_ranknet_GPT_cleaned"),
        # 'gradient_accumulation_steps': 4,
        "loss_type": "ranknet",   # "weighted" or "focal" or "ranknet" or "graded ranknet"
        # "focal_gamma": 1.0,
    }
    # Set the number of labels; adjust as needed
    num_labels = 4

    # Initialize the trainer.
    cross_encoder = CrossEncoderTrainer(model_checkpoint, num_labels, hyperparameters)

    cross_encoder.set_dataset_dicts(corpus_path=os.path.join(STORAGE_DIR, "legal_ir", "data", "external", "BatchAPI_outputs", "cleanup", "corpus_Gpt4o-mini_cleaned.json"))

    # Load and split the dataset
    if hyperparameters.get("loss_type") == "focal" or hyperparameters.get("loss_type") == "weighted":
        ds_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "cross_encoder", "cross_encoder_ds_finegrained")
        train_ds, val_ds, test_ds = cross_encoder.load_and_split_dataset(
            ds_path, test_size=0.2, val_size=0.1, max_length=hyperparameters["max_length"]
        )
    elif hyperparameters.get("loss_type") == "ranknet":
        ds_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "triplet_ds_standard_ranknet")
        train_ds, val_ds, test_ds = cross_encoder.prepare_ranknet_ds(
            ds_path, test_size=0.2, val_size=0.1, max_length=hyperparameters["max_length"]
        )
    elif hyperparameters.get("loss_type") == "graded ranknet":
        ds_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "triplet_ds_graded_ranknet")
        train_ds, val_ds, test_ds = cross_encoder.prepare_ranknet_ds(
            ds_path, test_size=0.2, val_size=0.1, max_length=hyperparameters["max_length"]
        )

    # cut train_ds to limit
    train_ds = train_ds.select(range(100))

    # Setup the trainer with training and validation sets.
    cross_encoder.setup_trainer(train_ds, val_ds)

    # Train the model.
    cross_encoder.train()

    # Evaluate the model.
    cross_encoder.evaluate(test_ds)

    # Save the model and tokenizer.
    cross_encoder.save()

    # evaluate IR metrics
    evaluator = Evaluator(ds="legal",
                          model_name="bm25",
                          metric_names={'ndcg', 'ndcg_cut.10', 'recall_1000', 'recall_100', 'recall_10', 'recip_rank'},
                          rerank=True,
                          reranker_model=cross_encoder.model,
                          tokenizer=cross_encoder.tokenizer,
                        #   model_instance=model
                )
    evaluator.evaluate()

    ir_metrics = {
        "ndcg": evaluator.metrics["ndcg"],
        "ndcg_cut_10": evaluator.metrics["ndcg_cut_10"],
        "recall_1000": evaluator.metrics["recall_1000"],
        "recall_100": evaluator.metrics["recall_100"],
        "recall_10": evaluator.metrics["recall_10"],
        "recip_rank": evaluator.metrics["recip_rank"],
    }

    # Replace 'gpu_name' with your actual GPU name if available.
    gpu_name = "RTX A5000"
    exp_id = "exp_" + uuid.uuid4().hex[:8]
    dataset_name = "cross_encoder_ds_finegrained"
    loss_name = hyperparameters.get("loss_type", "")
    cross_encoder.log_experiment(exp_id, dataset_name, loss_name, gpu_name, ir_metrics)


if __name__ == "__main__":
    main()
