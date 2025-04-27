import config.config as config
STORAGE_DIR = config.STORAGE_DIR
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import pandas as pd
from tqdm import tqdm
import pickle
import random
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from datasets import load_from_disk
import logging
logging.basicConfig(
        format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
    )
logger = logging.getLogger(__name__)


def create_datasets(tokenizer, queries, docs, max_length, batch_size=1000):
    """
    Create a Hugging Face Dataset from queries and docs in batches.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        The Hugging Face tokenizer to use for tokenizing text.
    queries : list of str
        A list of query strings.
    docs : list of str
        A list of document strings corresponding to each query.
    max_length : int, optional
        The maximum sequence length for each tokenized example.
    batch_size : int, optional
        The number of samples to process at once during tokenization.

    Returns
    -------
    dataset : datasets.Dataset
        A Hugging Face Dataset containing tokenized input_ids, attention_mask, etc.
    """
    logger.info("Building raw Dataset from queries and docs...")

    if os.path.exists(os.path.join(STORAGE_DIR, "legal_ir", "data", "tokenized_pair_ds.pkl")):
        with open(os.path.join(STORAGE_DIR, "legal_ir", "data", "tokenized_pair_ds.pkl"), "rb") as f:
            return pickle.load(f)
    
    # 1) Create a raw dataset from parallel lists
    raw_dataset = Dataset.from_dict({
        "query": queries,
        "doc": docs
    })

    logger.info("Mapping tokenizer over the dataset in batches...")

    # 2) Define a function that tokenizes a batch of queries and docs
    def tokenize_batch(batch):
        return tokenizer(
            batch["query"],
            batch["doc"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    # 3) Use .map() with batched=True and specify a batch size
    tokenized_dataset = raw_dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=batch_size
    )

    logger.info("Tokenization complete. Returning tokenized dataset...")

    # save to pickle file
    with open(os.path.join(STORAGE_DIR, "legal_ir", "data", "tokenized_pair_ds.pkl"), "wb") as f:
        pickle.dump(tokenized_dataset, f)
    return tokenized_dataset


def create_pair_dataset():
    corpus_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus_py.csv")
    queries_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "queries_57.csv")
    corpus = pd.read_csv(corpus_path, usecols=["Codigo"])
    queries = pd.read_csv(queries_path, usecols=["topic_id"])
    # create all possible pairs of queries and corpus and write to a tsv file with columns: query_id, doc_id
    pairs = []
    for query_id in tqdm(queries["topic_id"]):
        for doc_id in corpus["Codigo"]:
            pairs.append([query_id, doc_id])
    assert len(pairs) == 57 * 5000
    pairs = pd.DataFrame(pairs, columns=["query_id", "doc_id"])
    pairs.to_csv(os.path.join(STORAGE_DIR, "legal_ir", "data", "all_pairs.csv"), index=False)
    print("Pairs created and saved to all_pairs.csv")


def main():
    model_path = os.path.join(STORAGE_DIR, "legal_ir", "results", "cross_encoder_2048")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    pairs_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "all_pairs.csv")
    pairs = pd.read_csv(pairs_path)

    # load train_ds to filter out pairs that are already in the training set
    train_ds = load_from_disk(os.path.join(STORAGE_DIR, "legal_ir", "data", "train_ds"))
    train_queries = set(train_ds["query"])

    print("number of pairs before filtering:", len(pairs))
    # filter pairs to only contain queries that are in the training set
    pairs = pairs[pairs["query_id"].isin(train_queries)]
    print("number of pairs after filtering:", len(pairs))

    n_per_query = 200
    pairs = pairs.groupby("query_id", group_keys=False).apply(
        lambda group: group.sample(n=min(n_per_query, len(group)), random_state=42)
    )
    print("number of pairs after sampling:", len(pairs))
    assert len(pairs) == len(train_queries) * n_per_query, f"Expected {len(train_queries) * n_per_query} pairs, got {len(pairs)}"

    corpus_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus_py.csv")
    queries_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "queries_57.csv")
    corpus = pd.read_csv(corpus_path, usecols=["Codigo", "text"])
    queries = pd.read_csv(queries_path)
    
    docid_to_text = dict(zip(corpus["Codigo"], corpus["text"]))
    qid_to_query = dict(zip(queries["topic_id"], queries["Query"]))

    queries = [qid_to_query[qid] for qid in pairs["query_id"].tolist()]
    docs = [docid_to_text[docid] for docid in pairs["doc_id"].tolist()]

    dataset = create_datasets(tokenizer, queries, docs, max_length=2048)
    dataset.set_format("torch", columns=["input_ids", "attention_mask"])

    print("Dataset created")

    # Run inference on the dataset
    logger.info("Running inference on the dataset...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Create a DataLoader for the dataset
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    # extend pairs with their scores
    results = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)
            probabilities = F.softmax(outputs.logits, dim=-1)
            teacher_scores = probabilities[:, 1]  # Use the positive class score
            results.extend(teacher_scores.cpu().numpy())
    
    pairs["score"] = results
    pairs.to_csv(os.path.join(STORAGE_DIR, "legal_ir", "data", "10k_42r_pairs_scores.csv"), index=False)
    print("Inference done and scores saved to all_pairs_scores.csv")


if __name__ == "__main__":
    main()
