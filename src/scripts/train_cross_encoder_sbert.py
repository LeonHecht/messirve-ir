import logging

import torch

from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)

# import CrossEncoderClassificationEvaluator
from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator

# import Binary Cross Entropy Loss
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

import pandas as pd
from datasets import Dataset, concatenate_datasets
from pathlib import Path
import os

import sys
sys.path.append("home/leon/tesis/messirve-ir")
from config.config import STORAGE_DIR
from src.utils.retrieval_utils import get_legal_dataset, get_legal_queries

# make only gpu 1 visible
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def load_tsv_dataset(tsv_path, query_dict, doc_dict, max_length=None):
    """
    Carga un TSV qid,docid,label (sin split),
    lo mapea a texto y lo tokeniza.
    """
    df = pd.read_csv(tsv_path, sep="\t", usecols=["qid", "doc_id", "label"])
    # convert qid and doc_id to string
    df["qid"] = df["qid"].astype(str)
    df["doc_id"] = df["doc_id"].astype(str)

    ds = Dataset.from_pandas(df)

    # map IDs to text
    def map_ids(ex):
        ex["query"] = query_dict[ex["qid"]]
        ex["doc"]   = doc_dict[str(ex["doc_id"])]
        return ex
    ds = ds.map(map_ids)

    # remove qid and doc_id
    ds = ds.remove_columns(["qid", "doc_id"])
    return ds


def main():
    # model_name = "mrm8488/legal-longformer-base-8192-spanish"
    model_name = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    train_batch_size = 64
    num_epochs = 8
    num_hard_negatives = 6  # How many hard negatives should be mined for each question-answer pair

    # 1a. Load a model to finetune with
    model = CrossEncoder(model_name, max_length=512, num_labels=1)
    print("Model max length:", model.max_length)
    print("Model num labels:", model.num_labels)

    # 2a. Load the dataset:
    logging.info("Read the dataset")
    # TODO
    doc_ids, docs = get_legal_dataset(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus_py.csv"))
    doc_dict = {str(doc_id): doc for doc_id, doc in zip(doc_ids, docs)}
    query_ids, queries = get_legal_queries(Path(STORAGE_DIR) / "legal_ir" / "data" / "corpus" / "queries_57.csv")
    query_dict = {str(query_id): query for query_id, query in zip(query_ids, queries)}

    ds_train_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "cross_encoder", f"bce_train.tsv")
    ds_dev_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "cross_encoder", f"bce_dev.tsv")
    ds_test_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "cross_encoder", f"bce_test.tsv")

    train_dataset = load_tsv_dataset(ds_train_path, query_dict, doc_dict)
    eval_dataset = load_tsv_dataset(ds_dev_path, query_dict, doc_dict)
    test_dataset = load_tsv_dataset(ds_test_path, query_dict, doc_dict)

    # merge eval and test datasets
    eval_dataset = concatenate_datasets([eval_dataset, test_dataset])

    # rename the columns to query, response, label
    train_dataset = train_dataset.rename_column("doc", "response")
    eval_dataset = eval_dataset.rename_column("doc", "response")

    # reorder the columns into the order query, response, label
    train_dataset = train_dataset.select_columns(["query", "response", "label"])
    eval_dataset = eval_dataset.select_columns(["query", "response", "label"])

    # select only the first 1000 samples for train and eval
    # train_dataset = train_dataset.select(range(100))
    # eval_dataset = eval_dataset.select(range(100))

    logging.info(train_dataset)
    logging.info(eval_dataset)

    # 3. Define our training loss.
    # pos_weight is recommended to be set as the ratio between positives to negatives, a.k.a. `num_hard_negatives`
    loss = BinaryCrossEntropyLoss(model=model, pos_weight=torch.tensor(num_hard_negatives))

    # Initialize the evaluator
    pairs = list(zip(eval_dataset["query"], eval_dataset["response"]))
    labels = list(eval_dataset["label"])
    cls_evaluator = CrossEncoderClassificationEvaluator(
        sentence_pairs=pairs,
        labels=labels,
        name="bce_6x_test",
    )
    results = cls_evaluator(model)
    print(cls_evaluator.primary_metric)
    print(results[cls_evaluator.primary_metric])

    # 5. Define the training arguments
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"reranker-{short_model_name}-bce"
    args = CrossEncoderTrainingArguments(
        # Required parameter:
        output_dir=os.path.join(STORAGE_DIR, "legal_ir", "results", f"models/{run_name}"),
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        # dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_bce_6x_test_average_precision",
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        logging_steps=20,
        logging_first_step=True,
        seed=42,
    )

    # 6. Create the trainer & start training
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=cls_evaluator,
    )
    trainer.train()

    # 7. Evaluate the final model, useful to include these in the model card
    cls_evaluator(model)

if __name__ == "__main__":
    main()