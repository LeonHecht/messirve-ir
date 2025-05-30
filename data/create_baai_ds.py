from tqdm import tqdm
import os
import sys
import json
import csv
from typing import List, Dict
import pandas as pd
import random


def configure_python_path():
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    print(f"Adding {project_root} to sys.path")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

configure_python_path()

from config.config import STORAGE_DIR
from src.utils.retrieval_utils import get_legal_dataset, get_legal_queries


def convert_tsv_to_json_chunked(
    tsv_path: str,
    query_ids: List[str],
    queries: List[str],
    chunk_texts: Dict[str, str],
    output_path: str,
    prompt: str,
    negatives_per_positive: int = 6,
) -> None:
    """
    Convert a chunked TSV dataset (qid, chunk_id, label) to JSON lines format.
    
    Each JSON object contains one positive chunk and a fixed number of unique
    negative chunks for the same query. Negatives are not reused across positives.
    
    Parameters
    ----------
    tsv_path : str
        Path to TSV with columns ['qid', 'chunk_id', 'label'] separated by tabs.
    query_ids : list of str
        List of query IDs (strings).
    queries : list of str
        List of query texts corresponding to `query_ids`.
    chunk_texts : dict of str -> str
        Mapping from chunk ID to chunk text.
    output_path : str
        Path to write output JSON lines file.
    prompt : str
        Prompt template to include in each JSON object.
    negatives_per_positive : int, optional
        Number of unique negatives per positive (default 6).
    
    Returns
    -------
    None
    """
    # Map query IDs to their text
    query_map: Dict[str, str] = dict(zip(query_ids, queries))

    # Group texts by query and label
    grouped: Dict[str, Dict[str, List[str]]] = {}
    with open(tsv_path, 'r', encoding='utf-8', newline='') as tsv_in:
        reader = csv.DictReader(tsv_in, delimiter='\t')
        for row in reader:
            qid = row['qid']
            cid = row['chunk_id']
            label = row['label']
            grouped.setdefault(qid, {'pos': [], 'neg': []})
            text = chunk_texts.get(cid, '')
            if label == '1':
                grouped[qid]['pos'].append(text)
            else:
                grouped[qid]['neg'].append(text)

    # Write JSON lines
    with open(output_path, 'w', encoding='utf-8') as json_out:
        for qid, docs in grouped.items():
            pos_texts = docs['pos']
            neg_texts = docs['neg']
            total_req = len(pos_texts) * negatives_per_positive
            if len(neg_texts) < total_req:
                raise ValueError(
                    f"Not enough negatives for query {qid}: "
                    f"have {len(neg_texts)}, need {total_req}"
                )
            # Sample or trim negatives
            if len(neg_texts) != total_req:
                negs_to_use = random.sample(neg_texts, total_req)
            else:
                negs_to_use = neg_texts.copy()
            random.shuffle(negs_to_use)

            for idx, pos in enumerate(pos_texts):
                start = idx * negatives_per_positive
                end = start + negatives_per_positive
                sampled_negs = negs_to_use[start:end]
                entry = {
                    'query': query_map[qid],
                    'pos': [pos],
                    'neg': sampled_negs,
                    'pos_scores': [],
                    'neg_scores': [],
                    'prompt': prompt,
                    'type': ''
                }
                json_out.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(
        f"Converted TSV to JSON lines with "
        f"{negatives_per_positive} negatives per positive: {output_path}"
    )


def convert_tsv_to_json(
    tsv_path: str,
    query_ids: List[str],
    queries: List[str],
    doc_map: Dict[str, str],
    output_path: str,
    prompt: str,
    negatives_per_positive: int = 6,
) -> None:
    """
    Convert a TSV dataset with qid, doc_id, and label to JSON lines format,
    creating one training group per positive document with a fixed number of
    unique negatives.

    Each JSON object contains exactly one positive passage and
    `negatives_per_positive` distinct negative passages for the same query,
    and negatives are not reused across positives.

    Parameters
    ----------
    tsv_path : str
        Path to the input TSV file containing 'qid', 'doc_id', and 'label'
        columns separated by tabs.
    query_ids : list of str
        List of query IDs obtained from get_legal_queries.
    queries : list of str
        List of query strings corresponding to `query_ids`.
    doc_map : dict of str -> str
        Mapping from document ID to full document text.
    output_path : str
        Path to write the output JSON lines file.
    prompt : str
        Prompt string to include in each JSON object.
    negatives_per_positive : int, optional
        Number of unique negatives to assign to each positive (default is 6).

    Returns
    -------
    None
    """
    # Map query IDs to query strings
    query_map: Dict[str, str] = dict(zip(query_ids, queries))

    # Group document texts by qid and label
    grouped: Dict[str, Dict[str, List[str]]] = {}
    with open(tsv_path, 'r', newline='', encoding='utf-8') as tsv_in:
        reader = csv.DictReader(tsv_in, delimiter='\t')
        for row in reader:
            qid = row['qid']
            doc_id = row['doc_id']
            label = row['label']
            grouped.setdefault(qid, {'pos': [], 'neg': []})
            text = doc_map.get(doc_id, '')
            if label == '1':
                grouped[qid]['pos'].append(text)
            else:
                grouped[qid]['neg'].append(text)

    # Write JSON lines with unique negatives per positive
    with open(output_path, 'w', encoding='utf-8') as json_out:
        for qid, docs in grouped.items():
            pos_texts = docs['pos']
            neg_texts = docs['neg']
            total_required = len(pos_texts) * negatives_per_positive
            if len(neg_texts) < total_required:
                raise ValueError(
                    f"Not enough negatives for query {qid}: "
                    f"have {len(neg_texts)}, need {total_required}"
                )
            # Choose or trim negatives to match exactly P * N
            if len(neg_texts) != total_required:
                negs_to_use = random.sample(neg_texts, total_required)
            else:
                negs_to_use = neg_texts.copy()
            random.shuffle(negs_to_use)
            for idx, pos in enumerate(pos_texts):
                start = idx * negatives_per_positive
                end = start + negatives_per_positive
                sampled_negs = negs_to_use[start:end]
                entry = {
                    'query': query_map[qid],
                    'pos': [pos],
                    'neg': sampled_negs,
                    'pos_scores': [],
                    'neg_scores': [],
                    'prompt': prompt,
                    'type': ''
                }
                json_out.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(
        f"Converted TSV to JSON lines with {negatives_per_positive} "
        f"unique negatives per positive: {output_path}"
    )


def chunk_text(
    text: str,
    max_length: int,
    stride: int = None
) -> List[str]:
    """
    Split text into overlapping chunks of at most `max_length` tokens.

    If `stride` is not provided, defaults to 50% overlap (max_length // 2).

    Parameters
    ----------
    text : str
        The full document text.
    max_length : int
        Maximum number of tokens per chunk.
    stride : int, optional
        Number of tokens to advance for each new chunk.

    Returns
    -------
    List[str]
        List of text chunks.
    """
    tokens = text.split()
    if stride is None:
        stride = max_length // 2
    chunks = [
        " ".join(tokens[i : i + max_length])
        for i in range(0, len(tokens), stride)
        if tokens[i : i + max_length]
    ]
    return chunks


def build_baai_ds():
    base_dir = os.path.join(STORAGE_DIR, "legal_ir", "data")
    corpus_dir = os.path.join(base_dir, "corpus")
    query_path = os.path.join(corpus_dir, "inpars_mistral-small-2501_queries_Q1.tsv")
    corpus_path = os.path.join(corpus_dir, "corpus.jsonl")

    qids, queries = get_legal_queries(query_path)
    dids, docs = get_legal_dataset(corpus_path)
    doc_dict = dict(zip(dids, docs))

    in_paths = [
        "bce_6x_inpars_chunked_train.tsv",
        "bce_6x_inpars_chunked_dev.tsv",
        "bce_6x_inpars_chunked_test.tsv",
    ]
    out_paths = [
        "bce_6x_inpars_train_chunked_baai.jsonl",
        "bce_6x_inpars_dev_chunked_baai.jsonl",
        "bce_6x_inpars_test_chunked_baai.jsonl",
    ]

    max_length = 512
    stride = 256

    chunk_map: Dict[str, str] = {}
    chunk_texts: Dict[str, str] = {}

    for doc_id, text in tqdm(doc_dict.items(), desc="Chunking docs"):
        for idx, chunk in enumerate(chunk_text(text, max_length, stride)):
            cid = f"{doc_id}__chunk{idx}"
            chunk_map[cid] = doc_id
            chunk_texts[cid] = chunk

    for in_path, out_path in zip(in_paths, out_paths):
        convert_tsv_to_json_chunked(
            os.path.join(base_dir, "datasets", "cross_encoder", in_path),
            qids,
            queries,
            chunk_texts,
            os.path.join(base_dir, "datasets", "dual_encoder", out_path),
            prompt="Documento legal con el siguiente tema: "
        )


def build_baai_eval_ds(query_path, corpus_path, qrels_path,
                       baai_corpus_path, baai_queries_path, baai_qrels_path,
                       qrel_num_cols, relevance_range):
    
    qids, queries = get_legal_queries(query_path, header=None)
    dids, docs = get_legal_dataset(corpus_path)
    if qrel_num_cols == 3:
        qrels_dev_df = pd.read_csv(
                qrels_path,
                sep="\t",                # TREC qrels are usually tab-separated
                names=["query_id", "doc_id", "relevance"],
                header=0,            # There's no header in qrels files
                dtype={"query_id": str, "doc_id": str, "relevance": int}
            )
    elif qrel_num_cols == 4:
        qrels_dev_df = pd.read_csv(
                qrels_path,
                sep="\t",                # TREC qrels are usually tab-separated
                names=["query_id", "iteration", "doc_id", "relevance"],
                header=0,            # There's no header in qrels files
                dtype={"query_id": str, "iteration": int, "doc_id": str, "relevance": int}
            )
        qrels_dev_df = qrels_dev_df.drop(columns=["iteration"])
    else:
        raise ValueError("qrel_num_cols must be 3 or 4")

    qrel_qids = qrels_dev_df["query_id"].tolist()
    qrel_dids = qrels_dev_df["doc_id"].tolist()
    qrel_relevance = qrels_dev_df["relevance"].tolist()

    # baai corpus structure
    # {"id": "566392", "title": "", "text": "Have the check reissued to the proper payee."}
    baai_corpus = [
        {
            "id": did,
            "title": "",
            "text": doc
        }
        for did, doc in zip(dids, docs)
    ]

    with open(baai_corpus_path, "w", encoding="utf-8") as f:
        for line in baai_corpus:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    
    # baai queries structure
    # {"id": "15", "text": "Can I send a money order from USPS as a business?"}
    baai_queries = [
        {
            "id": qid.replace("_Q", "0000"),
            "text": query
        }
        for qid, query in zip(qids, queries) if qid in qrel_qids
    ]

    with open(baai_queries_path, "w", encoding="utf-8") as f:
        for line in baai_queries:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    
    # baai qrels structure
    # {"qid": "8", "docid": "566392", "relevance": 1}
    baai_qrels = [
        {
            "qid": qid.replace("_Q", "0000"),
            "docid": did,
            "relevance": 1
        }
        for qid, did, relevance in zip(qrel_qids, qrel_dids, qrel_relevance) if relevance in relevance_range
    ]

    with open(baai_qrels_path, "w", encoding="utf-8") as f:
        for line in baai_qrels:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    print("Done building BAAI eval dataset")


if __name__ == "__main__":
    build_baai_ds()

    # # in paths
    # base_dir = os.path.join(STORAGE_DIR, "legal_ir", "data")
    # corpus_dir = os.path.join(base_dir, "corpus")
    # # query_path = os.path.join(corpus_dir, "inpars_mistral-small-2501_queries.tsv")
    # query_path = os.path.join(corpus_dir, "queries_57.csv")
    # corpus_path = os.path.join(corpus_dir, "corpus_py.csv")
    # # qrels_path = os.path.join(base_dir, "datasets", "cross_encoder", "bce_6x_inpars_test.tsv")
    # qrels_path = os.path.join(base_dir, "annotations", "qrels_py.tsv")
    
    # # out paths
    # baai_dir = os.path.join(base_dir, "baai_57")
    # baai_corpus_path = os.path.join(baai_dir, "corpus.jsonl")
    # baai_queries_path = os.path.join(baai_dir, "test_queries.jsonl")
    # baai_qrels_path = os.path.join(baai_dir, "test_qrels.jsonl")

    # build_baai_eval_ds(query_path, corpus_path, qrels_path,
    #                    baai_corpus_path, baai_queries_path, baai_qrels_path,
    #                    qrel_num_cols=4, relevance_range=[2, 3])