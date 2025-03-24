"""
This script mines hard negatives with the bge-m3 model.
Output: hard_negatives_{split}_{model}_{country}.pkl
Required input: None

This is Script 1) of the pipeline
1) mine_hard_negatives.py
2) preprocess_dataset.py
3) prepare_ds_for_sbert.py (if later training with Sentence Transformers)
"""
from utils.retrieval_utils import embed_bge
from datasets import load_dataset
import torch
import os
from models.model_setup import get_bge_m3_model
import pickle
import sys
print("Executable:", sys.executable)
from tqdm import tqdm

STORAGE_DIR = os.getenv("STORAGE_DIR", "/media/discoexterno/leon/messirve-ir/data")
# STORAGE_DIR = os.getenv("STORAGE_DIR", "/tmpu/helga_g/leonh_a/messirve-ir/data")


def mine_hard_negatives(model, country, k, split):
    ds = load_dataset("spanish-ir/messirve", country)
    ds = ds[split]
    docs = ds["docid_text"]
    queries = ds["query"]
    doc_ids = ds["docid"]
    query_ids = ds["id"]
    print("Data prepared.")

    device = torch.device("cuda")

    if model == "bge":
        run_path = os.path.join(STORAGE_DIR, f"run_bge_{split}_{country}.pkl")
        if not os.path.exists(run_path):
            checkpoint = 'BAAI/bge-m3'
            # checkpoint = 'BAAI/bge-m3-unsupervised'
            # checkpoint = 'BAAI/bge-m3-retromae'
            model = get_bge_m3_model(checkpoint)
            run = embed_bge(model, docs, queries, doc_ids, query_ids, reuse_run=False)
            # save run to disk using pickle
            with open(run_path, "wb") as f:
                print("Dumping run to pickle file...", end="")
                pickle.dump(run, f)
                print("Done")
        else:
            with open(run_path, "rb") as f:
                print("Loading run from pickle file...", end="")
                run = pickle.load(f)
                print("Done")
    
    # Sort the values by relevance (descending)
    sorted_run = {
        query_id: dict(sorted(docs.items(), key=lambda item: item[1], reverse=True))
        for query_id, docs in tqdm(run.items())
    }

    hard_negatives = {}

    for i, (query_id, docs) in tqdm(enumerate(sorted_run.items()), total=len(sorted_run)):
        neg_doc_ids = []
        for doc_id, relevance in docs.items():
            if doc_ids[i] != doc_id:
                neg_doc_ids.append(doc_id)
            if len(neg_doc_ids) == k:
                hard_negatives[query_id] = neg_doc_ids
                break
    
    return hard_negatives


if __name__ == "__main__":
    model = "bge"
    country = "ar"
    k = 15
    split = "train"
    hard_negatives = mine_hard_negatives(model, country, k, split)
    with open(os.path.join(STORAGE_DIR, f"hard_negatives_{split}_{model}_{country}.pkl"), "wb") as f:
        print("Dumping hard negatives to pickle file...", end="")
        pickle.dump(hard_negatives, f)
        print("Done")