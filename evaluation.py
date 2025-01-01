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


def run(model, metrics):
    import sys
    print("Executable: ", sys.executable)

    ds = load_data("ar")
    docs = ds["test"]["docid_text"]
    queries = ds["test"]["query"]
    doc_ids = ds["test"]["docid"]
    doc_id_set = set(doc_ids)   # (len=3829) There are duplicates, so it seems as if on doc is the answer for several queries
    query_ids = ds["test"]["id"]
    print("Data prepared.")

    if model == "bm25":    
        if not os.path.exists("qrels_bm25.pkl"):
            qrels_model = retrieve_bm25(docs, queries, doc_ids, query_ids)
            # save qrels_model to disk using pickle
            with open("qrels_bm25.pkl", "wb") as f:
                pickle.dump(qrels_model, f)
        else:
            with open("qrels_bm25.pkl", "rb") as f:
                qrels_model = pickle.load(f)
        print("BM25 retrieval done.")
    else:
        raise ValueError("Model not supported.")
    
    # Evaluate BM25
    # qrels = {query_id: {doc_id: relevance, ...},
    #          query_id: {doc_id: relevance, ...}, ...},
    qrels = {}
    run = {}
    for query_id, rel_doc_id in zip(query_ids, doc_ids):
        qrel_dict = {str(rel_doc_id): 1}
        run_dict = {}
        for doc_id in doc_id_set:
            run_dict[str(doc_id)] = qrels_model[(query_id, doc_id)]
            if doc_id != doc_id:
                qrel_dict[str(doc_id)] = 0
        qrels[str(query_id)] = qrel_dict
        run[str(query_id)] = run_dict
    print("Qrels and runs prepared.")

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


if __name__ == "__main__":
    run(model="bm25", metrics={'ndcg', 'ndcg_cut.10', 'recall_100', 'recip_rank'})