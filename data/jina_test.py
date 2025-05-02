import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ndcg_score

import os
import sys

# --- Path Configuration ---
def configure_python_path():
    """Add the project root directory to sys.path."""
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

configure_python_path()

from config.config import STORAGE_DIR
from src.utils.retrieval_utils import get_legal_queries, get_legal_dataset


def read_qrels(
    qrels_tsv: str,
    query_ids: list,
    doc_ids: list
) -> np.ndarray:
    """
    Read a TREC-style qrels TSV file and build a relevance
    matrix y_true of shape (n_queries, n_docs).

    Parameters
    ----------
    qrels_tsv : str
        Path to qrels TSV file. Expected columns without header:
        query_id, doc_id, relevance (tab-separated).
    query_ids : list of str
        List of query IDs in the same order as embeddings_queries.
    doc_ids : list of str
        List of document IDs in the same order as embeddings_docs.

    Returns
    -------
    y_true : numpy.ndarray, shape (n_queries, n_docs)
        Ground-truth relevance levels.
    """
    df = pd.read_csv(
        qrels_tsv,
        sep="\t",
        header=None,
        names=["query_id", "run", "doc_id", "relevance"],
        dtype={"query_id": str, "run": int, "doc_id": str, "relevance": int}
    )

    # drop run column
    df = df.drop(columns=["run"])

    q_index = {qid: idx for idx, qid in enumerate(query_ids)}
    d_index = {did: idx for idx, did in enumerate(doc_ids)}

    y_true = np.zeros((len(query_ids), len(doc_ids)), dtype=int)

    for _, row in df.iterrows():
        qid = row["query_id"]
        did = row["doc_id"]
        rel = row["relevance"]
        if qid in q_index and did in d_index:
            y_true[q_index[qid], d_index[did]] = rel

    return y_true


def batch_encode_jina(model, data, batch_size, task, max_length):
    """
    Encode data in batches using the provided Jina model.
    Returns a list of all embeddings (on CPU).
    """
    from tqdm import trange

    embeddings = []
    for start_idx in trange(0, len(data), batch_size, desc="Batch encoding data"):
        batch = data[start_idx : start_idx + batch_size]
        batch_embeddings = model.encode(batch, task=task, max_length=max_length)
        batch_embeddings = np.array(batch_embeddings, dtype=np.float32)
        embeddings.extend(batch_embeddings)
    return embeddings


def compute_similarity_and_ndcg(
    embeddings_queries: torch.Tensor,
    embeddings_docs: torch.Tensor,
    y_true: np.ndarray,
    k: int = 10
) -> dict:
    """
    Compute cosine similarity matrix and average nDCG@k.

    Parameters
    ----------
    embeddings_queries : torch.Tensor, shape (n_queries, dim)
        Query embedding vectors.
    embeddings_docs : torch.Tensor, shape (n_docs, dim)
        Document embedding vectors.
    y_true : numpy.ndarray, shape (n_queries, n_docs)
        Ground-truth relevance labels (0/1 or graded).
    k : int, optional
        Rank cutoff for nDCG@k (default is 10).

    Returns
    -------
    result : dict
        {
            'similarity': np.ndarray, shape (n_queries, n_docs),
            'ndcg': float
        }
        similarity is the cosine similarity matrix,
        ndcg is the average nDCG@k score.
    """
    q_emb = embeddings_queries.detach().cpu().numpy()
    d_emb = embeddings_docs.detach().cpu().numpy()

    # Normalize embeddings
    q_norm = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    d_norm = d_emb / np.linalg.norm(d_emb, axis=1, keepdims=True)

    # Cosine similarity
    similarity = np.dot(q_norm, d_norm.T)

    # Compute average nDCG@k
    ndcg = ndcg_score(y_true, similarity, k=k)

    return {"similarity": similarity, "ndcg": ndcg}


def embed_jina_with_ndcg(
    model,
    docs: list,
    queries: list,
    doc_ids: list,
    query_ids: list,
    qrels_tsv: str,
    k: int = 10
) -> dict:
    """
    Embed queries and documents using Jina, read qrels,
    and compute similarity + nDCG@k.

    Parameters
    ----------
    model : JinaEmbeddingModel
        The Jina embeddings model.
    docs : list of str
        List of document texts.
    queries : list of str
        List of query texts.
    doc_ids : list of str
        Ordered list of document IDs.
    query_ids : list of str
        Ordered list of query IDs.
    qrels_tsv : str
        Path to a TSV file with qrels.
    k : int, optional
        Rank cutoff for nDCG@k (default is 10).

    Returns
    -------
    dict
        {
            'similarity': np.ndarray,
            'ndcg': float
        }
    """
    # Encode queries and docs
    embeddings_queries = torch.tensor(
        np.array(batch_encode_jina(
            model, queries, batch_size=128,
            task="retrieval.query", max_length=48
        )),
        dtype=torch.float32
    )
    embeddings_docs = torch.tensor(
        np.array(batch_encode_jina(
            model, docs, batch_size=128,
            task="retrieval.passage", max_length=4096
        )),
        dtype=torch.float32
    )

    # Read ground-truth relevance
    y_true = read_qrels(qrels_tsv, query_ids, doc_ids)

    # Compute similarity and nDCG
    return compute_similarity_and_ndcg(
        embeddings_queries, embeddings_docs, y_true, k=k
    )


def get_jinja_model():
    """ Load Jinja embeddings model."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
    return model


if __name__ == "__main__":
    model = get_jinja_model().to(device="cuda:0")

    doc_ids, docs = get_legal_dataset(
        os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus_py.csv"),
    )

    qids, queries = get_legal_queries(
        os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "queries_57.csv"),
    )

    y_true = read_qrels(
        os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "qrels_py.tsv"),
        query_ids=qids,
        doc_ids=doc_ids
    )

    results_dict = embed_jina_with_ndcg(
        model,
        docs=docs,
        queries=queries,
        doc_ids=doc_ids,
        query_ids=qids,
        qrels_tsv=os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "qrels_py.tsv"),
        k=10
    )

    print("results:", results_dict)
    print("similarity shape:", results_dict["similarity"].shape)
    print("ndcg:", results_dict["ndcg"])
    print("ndcg shape:", results_dict["ndcg"].shape)
    print("ndcg mean:", results_dict["ndcg"].mean())



