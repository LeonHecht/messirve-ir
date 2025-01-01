from models import model_setup
from data import dataset_preparation
from tqdm import tqdm
import pickle
import os
import pytrec_eval


def load_model(checkpoint):
    return model_setup.load_model(checkpoint)


def load_data(data):
    ds = dataset_preparation.load_messirve_dataset(country="ar")
    return ds


def retrieve_bm25(docs, queries, doc_ids, query_ids):
    """
    Embed the queries and documents using the BM25 model and compute the similarity between queries and documents.

    Args:
        docs (dict): Dictionary with document_id as key and text as value.
        queries (list): List of queries.
        doc_ids (list): List of document IDs.
        top_k (int): Number of most similar documents to retrieve.

    Returns:
        dict: Dictionary with (query_id, doc_id) as key and a list of tuples of score as value.
    """
    from rank_bm25 import BM25Okapi
    import numpy as np

    texts = docs

    # Simple space-based tokenization
    tokenized_corpus = [text.lower().split() for text in texts]

    # Create BM25 model
    bm25 = BM25Okapi(tokenized_corpus)

    # Queries
    tokenized_queries = [query.lower().split() for query in queries]

    # key: query, value: (similarity, text, doc_id)
    qrels = {}
    for tokenized_query, query_id in tqdm(zip(tokenized_queries, query_ids), total=len(tokenized_queries)):
        # Calcular las puntuaciones BM25 para la consulta en cada documento
        scores = bm25.get_scores(tokenized_query)

        # Ordenar los documentos por relevancia
        # sorted_docs = np.argsort(scores)[::-1]  # Orden descendente
        for i, score in enumerate(scores):
            qrels[(query_id, doc_ids[i])] = score
    return qrels


def retrieve():
    import sys
    assert sys.executable == "/home/leon/tesis/Environments/IR_env/bin/python"

    ds = load_data("ar")
    docs = ds["test"]["docid_text"]
    queries = ds["test"]["query"]
    doc_ids = ds["test"]["docid"]
    doc_id_set = set(doc_ids)   # (len=3829) There are duplicates, so it seems as if on doc is the answer for several queries
    query_ids = ds["test"]["id"]
    # query_id_set = set(query_ids)
    print(ds)
    print("Data prepared.")

    if not os.path.exists("qrels_bm25.pkl"):
        qrels_bm25 = retrieve_bm25(docs, queries, doc_ids, query_ids)
        # save qrels_bm25 to disk using pickle
        with open("qrels_bm25.pkl", "wb") as f:
            pickle.dump(qrels_bm25, f)
    else:
        with open("qrels_bm25.pkl", "rb") as f:
            qrels_bm25 = pickle.load(f)
    print("BM25 retrieval done.")

    # Evaluate BM25
    # qrels = {query_id: {doc_id: relevance, ...},
    #          query_id: {doc_id: relevance, ...},
    #          ...}
    qrels = {}
    run = {}
    for query_id, rel_doc_id in zip(query_ids, doc_ids):
        qrel_dict = {str(rel_doc_id): 1}
        run_dict = {}
        for doc_id in doc_id_set:
            run_dict[str(doc_id)] = qrels_bm25[(query_id, doc_id)]
            if doc_id != doc_id:
                qrel_dict[str(doc_id)] = 0
        qrels[str(query_id)] = qrel_dict
        run[str(query_id)] = run_dict
    print("Qrels and runs prepared.")

    # run = {}
    # for query_id in query_ids:
    #     qrel_dict = {}
    #     for doc_id in doc_ids:
    #         qrel_dict[str(doc_id)] = qrels_bm25[(query_id, doc_id)]
    #     run[str(query_id)] = qrel_dict
    # print("Run prepared.")

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg', 'ndcg_cut.10', 'recall_100', 'recip_rank'})
    results = evaluator.evaluate(run)

    print("Evaluation done.")
    metric_sums = {"ndcg": 0, "ndcg_cut_10": 0, "recall_100": 0, "recip_rank": 0}
    for query, metrics in results.items():
        metric_sums["ndcg"] += metrics["ndcg"]
        metric_sums["ndcg_cut_10"] += metrics["ndcg_cut_10"]
        metric_sums["recall_100"] += metrics["recall_100"]
        metric_sums["recip_rank"] += metrics["recip_rank"]

    avg_ndcg_10 = metric_sums["ndcg_cut_10"] / len(results)
    avg_ndcg = metric_sums["ndcg"] / len(results)
    avg_recall = metric_sums["recall_100"] / len(results)
    avg_recip_rank = metric_sums["recip_rank"] / len(results)

    print(f"\Average nDCG@10: {avg_ndcg_10}")
    print(f"\nAverage nDCG: {avg_ndcg}")
    print(f"Average Recall@100: {avg_recall}")
    print(f"Average Reciprocal Rank (MRR): {avg_recip_rank}\n")
    # Average nDCG@10: 0.4992929543547334
    # Average Recall@100: 0.7276044517423828
    # Average Reciprocal Rank: 0.4679763392013155


if __name__ == "__main__":
    retrieve()