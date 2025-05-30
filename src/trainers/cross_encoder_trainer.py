import os
# make only gpu1 visible
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import logging
import numpy as np
import math
from datasets import Dataset, load_from_disk
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          Trainer,
                          TrainingArguments,
                          EarlyStoppingCallback)
from sklearn.metrics import (accuracy_score,
                             precision_recall_fscore_support,
                             classification_report)
import torch
import torch.nn as nn
import pandas as pd
import json
import uuid
import pandas as pd
import sys


def configure_python_path():
    """
    Add the project root directory to sys.path.

    This function finds the directory two levels up from this file
    (the repo root) and inserts it at the front of sys.path so that
    `config.config` can be imported without errors.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Apply the path tweak before any project imports
configure_python_path()

from config.config import STORAGE_DIR
from src.utils.log_experiment import log_csv, log_md, log_plot
from src.utils.retrieval_utils import get_legal_queries, get_legal_dataset
from src.eval_class import Evaluator


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, hyperparameters=None, **kwargs):
        """
        Parameters
        ----------
        *args
            Positional args passed to transformers.Trainer.
        hyperparameters : dict, optional
            The dict of training hyperparameters, must include "loss_type" etc.
        class_weights : Sequence[float], optional
            Class weights computed once before training.
        """
        super().__init__(*args, **kwargs)
        self.hp = hyperparameters
        
        class_weights = self.hp.get("class_weights")
        if class_weights is not None:
            cw = torch.tensor(class_weights,
                              device=self.model.device,
                              dtype=torch.float)
            # For BCE: pos_weight is weight for the positive class
            self.pos_weight = cw[1]
            # For CE: weight vector for all classes
            self.ce_weight = cw
        else:
            print("[Warn]: Using WeightedLossTrainer w/o class weights. Only do so for focal loss or standard CE.")
            self.pos_weight = None
            self.ce_weight = None
        
        # Instantiate loss functions (reduction='none' for focal)
        if self.hp.get("loss_type") == "weighted":
            self.bce_fct = nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight,
                reduction="none"
            )
            self.ce_fct = nn.CrossEntropyLoss(
                weight=self.ce_weight,
                reduction="none"
            )
        else:
            # Unweighted BCE or CE
            self.bce_fct = nn.BCEWithLogitsLoss(reduction="none")
            self.ce_fct = nn.CrossEntropyLoss(reduction="none")
        
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
        labels = inputs.pop("labels")
        for k in ["qid", "doc_id", "query", "doc"]:
            inputs.pop(k, None)
        
        problem = model.config.problem_type
        
        # Forward pass (no built-in loss) and compute base loss
        if problem == "multi_label_classification":
            # BCE expects float labels and a single logit
            labels = labels.float()
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1)
            loss_tensor = self.bce_fct(logits, labels)
        else:
            # Cross-entropy for >1 classes
            outputs = model(**inputs)
            logits = outputs.logits
            loss_tensor = self.ce_fct(
                logits.view(-1, model.config.num_labels),
                labels.view(-1)
            )

        # Apply focal if requested
        if self.hp.get("loss_type") == "focal":
            gamma = self.hp.get("focal_gamma", 2.0)
            alpha = self.hp.get("focal_alpha", 0.25)
            pt = torch.exp(-loss_tensor)
            alpha_t = labels * alpha + (1 - labels) * (1 - alpha)
            loss = (alpha_t * (1 - pt) ** gamma * loss_tensor).mean()
        else:
            loss = loss_tensor.mean()

        if return_outputs:
            return loss, {"logits": logits, "labels": labels}
        return loss


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


class GradedRankNetTrainer(RankNetTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pos_label = inputs["pos_label"].float()
        neg_label = inputs["neg_label"].float()
        
        # Forward pass on positive docs
        pos_output = model(
            input_ids=inputs["input_ids_pos"],
            attention_mask=inputs["attention_mask_pos"]
        )
        # Forward pass on negative docs
        neg_output = model(
            input_ids=inputs["input_ids_neg"],
            attention_mask=inputs["attention_mask_neg"]
        )
        
        # Get the scores
        pos_scores = pos_output.logits[:, 0]
        neg_scores = neg_output.logits[:, 0]
        
        # Get sigma and compute score differences.
        sigma = kwargs.get("ranknet_sigma", 1.0)
        score_diff = pos_scores - neg_scores
        
        # Compute the elementwise RankNet loss (no mean yet)
        elementwise_loss = torch.log1p(torch.exp(-sigma * score_diff))
        
        # Compute grade difference for each pair.
        grade_diff = torch.abs(pos_label - neg_label)
        
        # Weight the loss per example by the grade difference
        weighted_loss = grade_diff * elementwise_loss
        
        # Finally, compute the mean over the batch.
        loss = weighted_loss.mean()
        
        outputs = (pos_output, neg_output)
        if return_outputs:
            return (loss, outputs)
        return loss


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
            model_checkpoint, num_labels=num_labels, problem_type=hyperparameters["problem_type"]
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
        bs  = self.hyperparameters["batch_size"]
        ga  = self.hyperparameters["gradient_accumulation_steps"]

        return TrainingArguments(
            output_dir=self.hyperparameters.get("output_dir", "./results"),
            num_train_epochs=self.hyperparameters.get("epochs", 100),
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs//2,
            warmup_ratio=self.hyperparameters.get("warmup_ratio", 0.1),
            weight_decay=self.hyperparameters.get("weight_decay", 0.01),
            learning_rate=self.hyperparameters.get("learning_rate", 5e-6),
            logging_dir='./logs',
            logging_steps=20,

            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,

            load_best_model_at_end=True,
            metric_for_best_model=self.hyperparameters.get("metric_for_best_model", "eval_f1"),

            remove_unused_columns=True,
            # prediction_loss_only=True,
            gradient_accumulation_steps=ga,
            # eval_accumulation_steps=1,
            max_grad_norm=20,

            save_total_limit=4,
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

    def compute_metrics(self, eval_pred):
        """
        Compute Accuracy, Precision, Recall and F1 for the current eval batch.

        Parameters
        ----------
        eval_pred : Tuple[Union[np.ndarray, torch.Tensor, Sequence], np.ndarray]
            The pair *(logits, labels)* provided by 🤗 `Trainer`.
            In evaluation mode the `Trainer` may wrap the logits in an
            additional tuple if your custom `compute_loss` returned both
            loss and outputs.

        Returns
        -------
        dict
            Keys ``accuracy``, ``precision``, ``recall``, ``f1``.
            During evaluation the ``Trainer`` prefixes them with ``eval_``
            (e.g. ``eval_f1``) so they can be used in
            ``TrainingArguments(metric_for_best_model=...)``.
        """
        logits, labels = eval_pred

        # ── unwrap (loss, {logits, labels}) artefact ──────────────────────
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        # ── to NumPy ──────────────────────────────────────────────────────
        if hasattr(logits, "detach"):
            logits = logits.detach().cpu().numpy()
        if hasattr(labels, "detach"):
            labels = labels.detach().cpu().numpy()

        # ── collapse one-hot label vectors to class ids (if any) ─────────
        if labels.ndim > 1:
            labels = labels.argmax(axis=-1)

        # ── predictions ──────────────────────────────────────────────────
        if self.model.config.problem_type == "multi_label_classification":
            # logits shape = (batch,)
            probs = torch.sigmoid(torch.from_numpy(logits))
            preds = (probs > 0.5).long().numpy()
        else:
            preds = logits.argmax(axis=-1)

        # print warning if preds are only 1s or only 0s
        if np.all(preds == 1):
            print("Warning: All predictions are 1s.")
        elif np.all(preds == 0):
            print("Warning: All predictions are 0s.")

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="macro",
            zero_division=0,
        )
        acc = accuracy_score(labels, preds)

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    
    def set_dataset_dicts(self, corpus_path, queries_path):
        dids, docs = get_legal_dataset(corpus_path)
        self.doc_dict = dict(zip(dids, docs))
        qids, queries = get_legal_queries(queries_path)
        self.query_dict = dict(zip(qids, queries))

    def tokenize_with_stride(self, examples):
        """
        Tokenize query/document pairs with sliding-window chunking.

        This uses the tokenizer's `return_overflowing_tokens` feature to split
        long documents into multiple chunks of length `max_len`, each overlapping
        by `stride` tokens.

        Parameters
        ----------
        examples : dict
            Batch dict with keys
            - "query": list[str]
            - "doc": list[str]
            - "label": list[int]

        Returns
        -------
        dict
            A dict with keys
            - "input_ids", "attention_mask", etc. for each chunk
            - "labels": list[int] of same length, mapped via overflow_to_sample_mapping
        """
        # hyperparameters you can tweak
        max_len = self.hyperparameters.get("max_length", 256)
        stride = self.hyperparameters.get("stride", max_len // 2)

        # ask tokenizer to chunk the *second* sequence (the document)
        tokenized = self.tokenizer(
            examples["query"],
            examples["doc"],
            truncation="only_second",
            max_length=max_len,
            stride=stride,
            return_overflowing_tokens=True,
            padding="max_length"
        )

        # map each chunk back to its original example to assign labels
        overflow_mapping = tokenized.pop("overflow_to_sample_mapping")
        labels = []
        for sample_idx in overflow_mapping:
            labels.append(examples["label"][sample_idx])
        tokenized["labels"] = labels

        return tokenized

    def load_tsv_dataset(self, tsv_path, max_length=None):
        """
        Carga un TSV qid,docid,label (sin split),
        lo mapea a texto y lo tokeniza.
        """
        df = pd.read_csv(tsv_path, sep="\t", usecols=["qid", "doc_id", "label"])
        # convert qid and doc_id to string
        df["qid"] = df["qid"].astype(str)
        df["doc_id"] = df["doc_id"].astype(str)

        ds = Dataset.from_pandas(df)

        # cut ds to half
        # ds = ds.select(range(len(ds)//10))
        # ds = ds.select(range(200))

        # map IDs to text
        def map_ids(ex):
            ex["query"] = self.query_dict[ex["qid"]]
            ex["doc"]   = self.doc_dict[str(ex["doc_id"])]
            return ex
        ds = ds.map(map_ids)

        # tokenize
        max_len = max_length or self.hyperparameters.get("max_length", 256)
        print(f"Tokenizing with max_len = {max_len}")
        def tokenize(exs):
            enc = self.tokenizer(
                exs["query"], exs["doc"],
                truncation=True, padding="max_length", max_length=max_len
            )
            enc["labels"] = exs["label"]
            return enc
        
        if self.hyperparameters["use_stride"] == True:
            ds = ds.map(self.tokenize_with_stride, batched=True, remove_columns=["qid", "doc_id", "query", "doc", "label"])
        else:
            ds = ds.map(tokenize, batched=True, num_proc=4, remove_columns=["qid", "doc_id", "query", "doc", "label"])

        # shuffle the entire dataset
        seed = self.hyperparameters.get("seed", 42)
        ds = ds.shuffle(seed=seed)

        return ds

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

    def compute_class_weights_from_dataset(self, train_dataset, alpha=1.0):
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
        weights = weights ** alpha
        # renormalize so mean(weight)=1
        weights = weights * self.num_labels / weights.sum()
        return weights.tolist()

    def setup_trainer(self, train_dataset, val_dataset):
        """
        Setup the Hugging Face Trainer, automatically computing class weights if not provided,
        and using the provided loss function.
        """
        hp = self.hyperparameters
        # Automatically compute class weights if not provided
        if hp.get("loss_type") == "weighted" or hp.get("loss_type") == "focal":
            if hp.get("loss_type") == "weighted":
                if "class_weights" not in hp:
                    computed_weights = self.compute_class_weights_from_dataset(train_dataset)
                    hp["class_weights"] = computed_weights
            TrainerClass = WeightedLossTrainer

        elif hp.get("loss_type") == "ranknet":
            TrainerClass = RankNetTrainer
        elif hp.get("loss_type") == "graded ranknet":
            TrainerClass = GradedRankNetTrainer
        else:
            raise Exception("Unknown loss type:", hp.get("loss_type"))

        self.trainer = TrainerClass(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            # compute_metrics=self.compute_pairwise_accuracy,
            hyperparameters=self.hyperparameters,
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
        self.logger.info("Evaluating on test dataset...")
        output = self.trainer.predict(test_dataset)
        logits = output.predictions
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        # ── decide BCE vs CE by shape ──────────────────────────────────
        if logits.ndim == 1:
            # already flat: shape (N,)
            flat = logits
        elif logits.ndim == 2 and logits.shape[1] == 1:
            # squeezed logit: shape (N,1)
            flat = logits[:, 0]
        else:
            # multi-class: shape (N, C)
            preds = np.argmax(logits, axis=-1)
            labels = (output.label_ids 
                    if hasattr(output, "label_ids") 
                    else np.array(test_dataset["labels"]))
            target_names = [f"Class {i}" for i in range(self.num_labels)]
            report = classification_report(labels, preds, target_names=target_names)
            self.logger.info("\n" + report)
            return output.metrics

        # ── binary case: sigmoid + threshold ──────────────────────────
        probs = 1 / (1 + np.exp(-flat))      # sigmoid
        preds = (probs > 0.5).astype(int)

        # ── true labels ───────────────────────────────────────────────
        labels = (output.label_ids 
                if hasattr(output, "label_ids") 
                else np.array(test_dataset["labels"]))

        n_classes = len(np.unique(labels))
        target_names = [f"Class {i}" for i in range(n_classes)]
        report = classification_report(labels, preds, target_names=target_names)
        self.logger.info("\n" + report)
        return output.metrics

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
    # Base configuration.
    # model_checkpoint = "mrm8488/legal-longformer-base-8192-spanish"
    model_checkpoint = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    hyperparameters = {
        "epochs": 1,
        "batch_size": 64,
        "weight_decay": 0.01,
        "learning_rate": 3e-5,  # placeholder; will be overridden
        "warmup_ratio": 0.1,
        "metric_for_best_model": "eval_f1",
        "early_stopping_patience": 20,
        "max_length": 512,
        "use_stride": True,
        "stride": 256,
        "output_dir": os.path.join(STORAGE_DIR, "legal_ir", "results"),
        "loss_type": "weighted",
        # "focal_gamma": 2.0,
        # "focal_alpha": 0.75,
        "gradient_accumulation_steps": 1,
        "corpus_type": "original",
        "seed": 42,
        "problem_type": "multi_label_classification",   # single_label_classification
    }

    loss_type    = hyperparameters["loss_type"]
    problem_type = hyperparameters["problem_type"]

    if loss_type in ("ranknet", "graded ranknet"):
        num_labels = 1
        hyperparameters["problem_type"] = "regression"

    elif loss_type in ("weighted", "focal"):
        if problem_type == "multi_label_classification":
            num_labels = 1
        elif problem_type == "single_label_classification":
            num_labels = 2
        else:
            raise ValueError(
                f"Invalid problem_type={problem_type!r} for loss_type={loss_type!r}"
            )

    else:
        raise ValueError(
            "loss_type must be one of "
            "'weighted', 'focal', 'ranknet', or 'graded ranknet'."
        )
    
    # List of learning rates to try.
    learning_rates = [3e-5]

    for lr in learning_rates:
        # Update hyperparameters for this run.
        hyperparameters = hyperparameters.copy()
        hyperparameters["learning_rate"] = lr

        # Update output_dir to include the learning rate in the folder name.
        hyperparameters["output_dir"] = os.path.join(
            STORAGE_DIR, "legal_ir", "results", f"cross_encoder_weighted_stride_synthetic"
        )

        print(f"\n=== Running cross-encoder training with learning_rate = {lr} ===")

        # print all hyperparameters
        print("Hyperparameters:", hyperparameters)

        # Initialize the trainer.
        cross_encoder = CrossEncoderTrainer(model_checkpoint, num_labels, hyperparameters)

        if hyperparameters["corpus_type"] == "original":
            cross_encoder.set_dataset_dicts(
                corpus_path=os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus.jsonl"),
                queries_path=os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "consultas_sinteticas_380_filtered.tsv")
            )        
        elif hyperparameters["corpus_type"] == "cleaned":
            cross_encoder.set_dataset_dicts(
                corpus_path=os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus_Gpt4o-mini_cleaned.json"),
                queries_path=os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "consultas_sinteticas_380_filtered.tsv")
            )

        # Load and split the dataset
        if hyperparameters["loss_type"] in ["focal", "weighted"]:
            # Version for training only on annotations:
            # ds_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "cross_encoder", "cross_encoder_ds_finegrained")
            # train_ds, val_ds, test_ds = cross_encoder.load_and_split_dataset(
            #     ds_path,
            #     test_size=0.2,
            #     val_size=0.1,
            #     max_length=hyperparameters["max_length"]
            # )

            ds_train_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "cross_encoder", f"bce_6x_synthetic_train.tsv")
            ds_dev_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "cross_encoder", f"bce_6x_synthetic_dev.tsv")
            ds_test_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "cross_encoder", f"bce_6x_synthetic_test.tsv")
            
            train_ds = cross_encoder.load_tsv_dataset(ds_train_path, max_length=hyperparameters["max_length"])
            val_ds = cross_encoder.load_tsv_dataset(ds_dev_path, max_length=hyperparameters["max_length"])
            test_ds = cross_encoder.load_tsv_dataset(ds_test_path, max_length=hyperparameters["max_length"])

        elif hyperparameters["loss_type"] == "ranknet":
            ds_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "triplet_ds_standard_ranknet")
            train_ds, val_ds, test_ds = cross_encoder.prepare_ranknet_ds(
                ds_path,
                test_size=0.2,
                val_size=0.1,
                max_length=hyperparameters["max_length"]
            )
        elif hyperparameters["loss_type"] == "graded ranknet":
            ds_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "triplet_ds_graded_ranknet")
            train_ds, val_ds, test_ds = cross_encoder.prepare_ranknet_ds(
                ds_path,
                test_size=0.2,
                val_size=0.1,
                max_length=hyperparameters["max_length"]
            )
        
        # Optionally limit the training size for debugging.
        # train_ds = train_ds.select(range(50))
        # val_ds = val_ds.select(range(50))
        # test_ds = test_ds.select(range(50))

        # Setup the trainer.
        cross_encoder.setup_trainer(train_ds, val_ds)

        # Train the model.
        cross_encoder.train()

        # Evaluate the model.
        eval_metrics = cross_encoder.evaluate_classification(test_ds)
        print(f"Eval metrics for learning_rate = {lr}: {eval_metrics}")

        # Save the model and tokenizer.
        cross_encoder.save()

        # Evaluate IR metrics.
        evaluator = Evaluator(
            ds="legal-54",
            model_name="bm25",
            metric_names={'ndcg', 'ndcg_cut.10', 'recall_1000', 'recall_100', 'recall_10', 'recip_rank'},
            rerank=True,
            reranker_model=cross_encoder.model,
            reranker_model_type="binary",
            tokenizer=cross_encoder.tokenizer,
            max_length=hyperparameters["max_length"],
            rerank_chunkwise=hyperparameters["use_stride"],
        )
        evaluator.evaluate()

        ir_metrics = {
            "pairwise_acc": eval_metrics.get("eval_pairwise_accuracy", "N/A"),
            "eval_f1": eval_metrics.get("eval_f1", "N/A"),
            "ndcg": evaluator.metrics["ndcg"],
            "ndcg_cut_10": evaluator.metrics["ndcg_cut_10"],
            "recall_1000": evaluator.metrics["recall_1000"],
            "recall_100": evaluator.metrics["recall_100"],
            "recall_10": evaluator.metrics["recall_10"],
            "recip_rank": evaluator.metrics["recip_rank"],
        }

        gpu_name = "RTX A5000"
        exp_id = "exp_" + uuid.uuid4().hex[:8]
        dataset_name = "bce_ds"
        loss_name = hyperparameters["loss_type"]
        cross_encoder.log_experiment(exp_id, dataset_name, loss_name, gpu_name, ir_metrics)


# def main2():
#     """
#     Ejecuta entrenamientos comparativos del cross‐encoder con salida binaria y pérdida weighted.
#     Itera sobre las configuraciones S1–S4 manteniendo los mismos hiperparámetros.
#     """
#     # Base configuration identical a main()
#     model_checkpoint = "mrm8488/legal-longformer-base-8192-spanish"
#     base_hparams = {
#         "epochs": 10,
#         "batch_size": 16,
#         "weight_decay": 0.01,
#         "learning_rate": 3e-5,
#         "warmup_ratio": 0.1,
#         "metric_for_best_model": "eval_f1",
#         "early_stopping_patience": 4,
#         "max_length": 2048,
#         "output_dir": os.path.join(STORAGE_DIR, "legal_ir", "results"),
#         "loss_type": "weighted",
#         "gradient_accumulation_steps": 1,
#         "corpus_type": "cleaned",
#     }
#     num_labels = 2

#     # Definimos los escenarios
#     # scenarios = ["S1", "S2", "S3", "S4_r2", "S4_r3"]
#     # scenarios = ["S2", "S4_r3"]
#     scenarios = ["S3"]

#     for scenario in scenarios:
#         # Ajustamos output_dir para este escenario
#         hparams = base_hparams.copy()
#         hparams["output_dir"] = os.path.join(
#             STORAGE_DIR, "legal_ir", "results", f"{scenario}_weighted"
#         )

#         print(f"\n=== Escenario {scenario} ===")
#         print("Hiperparámetros:", hparams)

#         # Inicializar CrossEncoderTrainer
#         trainer = CrossEncoderTrainer(model_checkpoint, num_labels, hparams)
#         trainer.set_dataset_dicts(
#             corpus_path=os.path.join(
#                 STORAGE_DIR, "legal_ir", "data",
#                 "corpus", "corpus_Gpt4o-mini_cleaned.json"
#             )
#         )

#         # Rutas a TSV ya generados por build_ce_dataset + splits
#         train_tsv = os.path.join(
#             STORAGE_DIR, "legal_ir", "data", "datasets", "cross_encoder",
#             f"bce_{scenario}_train.tsv"
#         )
#         dev_tsv = train_tsv.replace("_train.tsv", "_dev.tsv")
#         test_tsv = train_tsv.replace("_train.tsv", "_test.tsv")

#         # Carga y tokenización
#         train_ds = trainer.load_tsv_dataset(train_tsv, max_length=hparams["max_length"])
#         print("Len train_ds:", len(train_ds))
#         val_ds   = trainer.load_tsv_dataset(dev_tsv,   max_length=hparams["max_length"])
#         test_ds  = trainer.load_tsv_dataset(test_tsv,  max_length=hparams["max_length"])

#         # Setup, train y evaluar
#         trainer.setup_trainer(train_ds, val_ds)
#         trainer.train()
#         metrics = trainer.evaluate_classification(test_ds)
#         print(f"Resultados clasificación ({scenario}):\n", metrics)

#         # Guardar
#         trainer.save()

#         # Evaluar IR metrics rerankeando con el modelo entrenado
#         evaluator = Evaluator(
#             ds="legal",
#             model_name="bm25",
#             metric_names={'ndcg', 'ndcg_cut.10', 'recall_1000', 'recall_100', 'recall_10', 'recip_rank'},
#             rerank=True,
#             reranker_model=trainer.model,
#             tokenizer=trainer.tokenizer,
#         )
#         evaluator.evaluate()
#         ir_metrics = {
#             "eval_f1": metrics.get("eval_f1", "N/A"),
#             "ndcg": evaluator.metrics["ndcg"],
#             "ndcg_cut_10": evaluator.metrics["ndcg_cut_10"],
#             "recall_10": evaluator.metrics["recall_10"],
#             "recip_rank": evaluator.metrics["recip_rank"],
#         }

#         # Logging
#         exp_id = f"{scenario}_" + uuid.uuid4().hex[:6]
#         trainer.log_experiment(
#             exp_id=exp_id,
#             dataset_name=scenario,
#             loss_name=hparams["loss_type"],
#             gpu_name="RTX A5000",
#             ir_metrics=ir_metrics
#         )


if __name__ == "__main__":
    main()
