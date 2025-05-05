from tqdm import tqdm
import os
import sys
import json
import csv
from typing import List, Dict
import pandas as pd


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


def convert_tsv_to_json(
    tsv_path: str,
    query_ids: List[str],
    queries: List[str],
    doc_map: Dict[str, str],
    output_path: str,
    prompt: str,
) -> None:
    """
    Convert a TSV dataset with qid, doc_id, and label to JSON lines format,
    replacing doc IDs with full document texts.

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
        Mapping from document ID to full document text (as returned by get_legal_dataset).
    output_path : str
        Path to write the output JSON lines file.
    prompt : str
        Prompt string to include in each JSON object.

    Returns
    -------
    None
    """
    # Map query IDs to query strings
    query_map: Dict[str, str] = dict(zip(query_ids, queries))

    # Parse TSV and group document texts by qid and label
    grouped: Dict[str, Dict[str, List[str]]] = {}
    with open(tsv_path, 'r', newline='', encoding='utf-8') as tsv_in:
        reader = csv.DictReader(tsv_in, delimiter='\t')
        for row in reader:
            qid = row['qid']
            doc_id = row['doc_id']
            label = row['label']
            if qid not in grouped:
                grouped[qid] = {'pos': [], 'neg': []}
            text = doc_map[doc_id]
            if label == '1':
                grouped[qid]['pos'].append(text)
            else:
                grouped[qid]['neg'].append(text)

    # Write JSON lines
    with open(output_path, 'w', encoding='utf-8') as json_out:
        for qid, docs in grouped.items():
            entry = {
                'query': query_map[qid],
                'pos': docs['pos'],
                'neg': docs['neg'],
                'pos_scores': [],
                'neg_scores': [],
                'prompt': prompt,
                'type': ""
            }
            json_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
    print("Converted TSV to JSON lines:", output_path)


def build_baai_ds():
    base_dir = os.path.join(STORAGE_DIR, "legal_ir", "data")
    corpus_dir = os.path.join(base_dir, "corpus")
    query_path = os.path.join(corpus_dir, "inpars_mistral-small-2501_queries.tsv")
    corpus_path = os.path.join(corpus_dir, "corpus_py.csv")

    qids, queries = get_legal_queries(query_path, header=None)
    dids, docs = get_legal_dataset(corpus_path)
    doc_dict = dict(zip(dids, docs))

    in_paths = [
        "bce_6x_inpars_train.tsv",
        "bce_6x_inpars_dev.tsv",
        "bce_6x_inpars_test.tsv",
    ]
    out_paths = [
        "bce_6x_inpars_train_baai.jsonl",
        "bce_6x_inpars_dev_baai.jsonl",
        "bce_6x_inpars_test_baai.jsonl",
    ]

    for in_path, out_path in zip(in_paths, out_paths):
        convert_tsv_to_json(
            os.path.join(base_dir, "datasets", "cross_encoder", in_path),
            qids,
            queries,
            doc_dict,
            os.path.join(base_dir, "datasets", "dual_encoder", out_path),
            prompt="Documento legal con el siguiente tema: "
        )


def build_baai_eval_ds():
    # in paths
    base_dir = os.path.join(STORAGE_DIR, "legal_ir", "data")
    corpus_dir = os.path.join(base_dir, "corpus")
    query_path = os.path.join(corpus_dir, "inpars_mistral-small-2501_queries.tsv")
    corpus_path = os.path.join(corpus_dir, "corpus_py.csv")
    qrels_path = os.path.join(base_dir, "datasets", "cross_encoder", "bce_6x_inpars_test.tsv")
    
    # out paths
    baai_dir = os.path.join(base_dir, "baai_bce_inpars_test")
    baai_corpus_path = os.path.join(baai_dir, "corpus.jsonl")
    baai_queries_path = os.path.join(baai_dir, "test_queries.jsonl")
    baai_qrels_path = os.path.join(baai_dir, "test_qrels.jsonl")

    qids, queries = get_legal_queries(query_path, header=None)
    dids, docs = get_legal_dataset(corpus_path)
    qrels_dev_df = pd.read_csv(
            qrels_path,
            sep="\t",                # TREC qrels are usually tab-separated
            names=["query_id", "doc_id", "relevance"],
            header=0,            # There's no header in qrels files
            dtype={"query_id": str, "doc_id": str, "relevance": int}
        )
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
            "relevance": relevance
        }
        for qid, did, relevance in zip(qrel_qids, qrel_dids, qrel_relevance) if relevance == 1
    ]

    with open(baai_qrels_path, "w", encoding="utf-8") as f:
        for line in baai_qrels:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    print("Done building BAAI eval dataset")


if __name__ == "__main__":
    # build_baai_ds()
    build_baai_eval_ds()