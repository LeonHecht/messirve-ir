import sys
print("Executable: ", sys.executable)
from models import model_setup
from datasets import load_dataset
import pickle
import os
import pytrec_eval
import torch
from utils.retrieval_utils import (
    embed_bge,
    embed_jinja,
    embed_mamba,
    retrieve_bm25,
    embed_s_transformers
)

from models.model_setup import get_bge_m3_model, get_jinja_model

def run(model, metrics, country, model_instance=None, reuse_run=False):
    print("Starting Evaluation")
    ds = load_dataset("spanish-ir/messirve", country)
    docs = ds["test"]["docid_text"]
    queries = ds["test"]["query"]
    doc_ids = ds["test"]["docid"]
    query_ids = ds["test"]["id"]
    print("Data prepared.")

    device = torch.device("cuda")

    if model == "bm25":
        run_path = f"run_bm25_{country}.pkl"
        if not os.path.exists(run_path) or not reuse_run:
            run = retrieve_bm25(docs, queries, doc_ids, query_ids)
            # save qrels_model to disk using pickle
            with open(run_path, "wb") as f:
                print("Dumping run to pickle file...", end="")
                pickle.dump(run, f)
                print("Done")
        else:
            with open(run_path, "rb") as f:
                run = pickle.load(f)
        print("BM25 retrieval done.")
    elif model == "bge":
        run_path = f"run_bge_{country}.pkl"
        if not os.path.exists(run_path) or not reuse_run:
            checkpoint = 'BAAI/bge-m3'
            # checkpoint = 'BAAI/bge-m3-unsupervised'
            # checkpoint = 'BAAI/bge-m3-retromae'
            model = get_bge_m3_model(checkpoint)
            run = embed_bge(model, docs, queries, doc_ids, query_ids)
            # save run to disk using pickle
            with open(run_path, "wb") as f:
                print("Dumping run to pickle file...")
                pickle.dump(run, f)
        else:
            with open(run_path, "rb") as f:
                run = pickle.load(f)
    elif model == "jinja":
        run_path = f"run_jinja_{country}.pkl"
        if not os.path.exists(run_path) or not reuse_run:
            model = get_jinja_model()
            model.to(device)
            run = embed_jinja(model, docs, queries, doc_ids, query_ids)
            # save run to disk using pickle
            with open(run_path, "wb") as f:
                print("Dumping run to pickle file...")
                pickle.dump(run, f)
        else:
            with open(run_path, "rb") as f:
                run = pickle.load(f)
    elif model == "mamba":
        run_path = f"run_mamba_{country}.pkl"
        if not os.path.exists(run_path) or not reuse_run:
            model, tokenizer = model_setup.get_mamba_model()
            model.to(device)
            run = embed_mamba(model, tokenizer, docs, queries, doc_ids, query_ids)
            # save run to disk using pickle
            with open(run_path, "wb") as f:
                print("Dumping run to pickle file...")
                pickle.dump(run, f)
        else:
            with open(run_path, "rb") as f:
                run = pickle.load(f)
    elif model == "bge-finetuned":
        run_path = f"run_bge_finetuned_{country}.pkl"
        if not os.path.exists(run_path) or not reuse_run:
            checkpoint = 'test_encoder_only_m3_bge-m3_sd'
            # checkpoint = 'BAAI/bge-m3-unsupervised'
            # checkpoint = 'BAAI/bge-m3-retromae'
            model = get_bge_m3_model(checkpoint)
            run = embed_bge(model, docs, queries, doc_ids, query_ids)
            # save run to disk using pickle
            with open(run_path, "wb") as f:
                print("Dumping run to pickle file...")
                pickle.dump(run, f)
        else:
            with open(run_path, "rb") as f:
                run = pickle.load(f)
    elif model == "sentence-transformer":
        run_path = f"run_sentence_transformer_{country}.pkl"
        if not os.path.exists(run_path) or not reuse_run:
            model = model_instance
            run = embed_s_transformers(model, docs, queries, doc_ids, query_ids)
            # save run to disk using pickle
            with open(run_path, "wb") as f:
                print("Dumping run to pickle file...")
                pickle.dump(run, f)
        else:
            with open(run_path, "rb") as f:
                run = pickle.load(f)
    else:
        raise ValueError("Model not supported.")
    
    # Evaluate BM25
    # qrels = {query_id: {doc_id: relevance, ...},
    #          query_id: {doc_id: relevance, ...}, ...},
    qrels = {}
    for query_id, rel_doc_id in zip(query_ids, doc_ids):
        # put all qrels to 0
        qrels[str(query_id)] = {str(doc_id): 0 for doc_id in doc_ids}
        # set the qrel for the relevant doc to 1 (assuming 1 relevant doc per query)
        qrels[str(query_id)][str(rel_doc_id)] = 1
    print("Qrels prepared.")

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    results = evaluator.evaluate(run)
    print("Evaluation done.")

    result_values = list(results.values())
    metric_names = list(result_values[0].keys())      # because some result names change e.g. from ndcg_cut.10 to ndcg_cut_10
    metric_sums = {metric_name: 0 for metric_name in metric_names}
    for metrics_ in results.values():
        for metric in metric_names:
            metric_sums[metric] += metrics_[metric]
    
    # Average metrics over all queries
    assert len(results) == len(queries)
    avg_metrics = {metric_name: metric_sums[metric_name]/len(queries) for metric_name in metric_names}
    
    print("\nResults:")
    for metric_name, metric_value in avg_metrics.items():
        print(f"Average {metric_name}: {metric_value}")

    print("\n")
    
    return avg_metrics


if __name__ == "__main__":
    # run(model="bm25", metrics={'ndcg', 'ndcg_cut.10', 'recall_100', 'recip_rank'}, country="ar")
    # run(model="bge", metrics={'ndcg', 'ndcg_cut.10', 'recall_100', 'recip_rank'}, country="ar")
    run(model="bge-finetuned", metrics={'ndcg', 'ndcg_cut.10', 'recall_100', 'recall_10', 'recip_rank'}, country="ar")
    # run(model="jinja", metrics={'ndcg', 'ndcg_cut.10', 'recall_100', 'recip_rank'}, country="ar")
    # run(model="mamba", metrics={'ndcg', 'ndcg_cut.10', 'recall_100', 'recip_rank'}, country="ar")