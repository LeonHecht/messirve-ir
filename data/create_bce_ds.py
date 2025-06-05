from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import json
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import os
import sys
from typing import List, Dict, Set

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


# utility to load qid lists
def load_qids(path):
    df = pd.read_csv(path, header=None, usecols=[0], dtype={0: str})
    return set(df[0].tolist())


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


def build_ce_dataset_chunked(
    qrels_path: str,
    pos_labels: List[int],
    neg_labels: List[int],
    corpus_path: str,
    queries_path: str,
    output_path: str,
    max_length: int,
    stride: int = None,
    qid_filter: Set[str] = None,
    neg_ratio: int = 6,
    med_cap: int = 3,
    med_offset: int = 10,
    med_top_k: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a binary-label cross-encoder dataset with overlapping chunking.

    Documents are split into chunks of up to `max_length` tokens with
    a sliding window of `stride` tokens (default 50% overlap).

    Query and document IDs are consistently handled as strings.

    Parameters
    ----------
    qrels_path : str
        Path to TSV with columns [qid, run, doc_id, label].
    pos_labels : list of int
        Labels treated as positives.
    neg_labels : list of int
        Labels treated as annotated negatives.
    corpus_path : str
        Path to JSON mapping doc_id to document text.
    queries_path : str
        Path to CSV with ['topic_id', 'Query'] columns.
    output_path : str
        Destination TSV file (qid, chunk_id, label).
    max_length : int
        Maximum token length per document chunk.
    stride : int, optional
        Tokens to slide between chunks. Defaults to max_length // 2.
    qid_filter : set of str, optional
        If provided, only include these query IDs.
    neg_ratio : int, optional
        Number of negatives per positive (default 6).
    med_cap : int, optional
        Max BM25 negatives per positive (default 3).
    med_offset : int, optional
        BM25 ranking offset (default 10).
    med_top_k : int, optional
        BM25 top-k window size (default 150).
    seed : int, optional
        Random seed (default 42).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['qid', 'chunk_id', 'label'].
    """
    random.seed(seed)

    # 1) Load relevance judgments as strings
    df_q = pd.read_csv(
        qrels_path, sep="\t", header=None,
        names=["qid", "run", "doc_id", "label"]
    )
    df_q["qid"] = df_q["qid"].astype(str)
    df_q["doc_id"] = df_q["doc_id"].astype(str)

    positives = (
        df_q[df_q.label.isin(pos_labels)]
        .groupby("qid")["doc_id"].apply(list).to_dict()
    )
    annotated_negs = (
        df_q[df_q.label.isin(neg_labels)]
        .groupby("qid")["doc_id"].apply(list).to_dict()
    )

    # 2) Load and chunk corpus with doc_id as string
    raw_ids, docs = get_legal_dataset(corpus_path)
    corpus = {str(did): text for did, text in zip(raw_ids, docs)}

    chunk_map: Dict[str, str] = {}
    chunk_texts: Dict[str, str] = {}

    for doc_id, text in tqdm(corpus.items(), desc="Chunking docs"):
        for idx, chunk in enumerate(chunk_text(text, max_length, stride)):
            cid = f"{doc_id}__chunk{idx}"
            chunk_map[cid] = doc_id
            chunk_texts[cid] = chunk

    chunk_ids = list(chunk_texts.keys())
    tokenized_chunks = [chunk_texts[cid].split() for cid in chunk_ids]
    bm25 = BM25Okapi(tokenized_chunks)

    # 3) Load queries with qid as string
    raw_qids, queries = get_legal_queries(queries_path)
    qids = [str(q) for q in raw_qids]
    query_dict = dict(zip(qids, queries))

    rows = []
    n_annotated = n_medium = n_easy = 0

    for qid, q_text in tqdm(query_dict.items(), desc="Processing queries"):
        if qid_filter and qid not in qid_filter:
            continue

        pos_docs = positives.get(qid, [])
        if not pos_docs:
            continue

        pos_chunk_ids = [
            cid for cid, did in chunk_map.items() if did in pos_docs
        ]

        desired_neg = neg_ratio * len(pos_chunk_ids)
        if desired_neg == 0:
            desired_neg = len(annotated_negs.get(qid, []))

        hard_docs = annotated_negs.get(qid, [])
        hard_chunk_ids = [
            cid for cid, did in chunk_map.items() if did in hard_docs
        ]

        if len(hard_chunk_ids) >= desired_neg:
            hard_used = random.sample(hard_chunk_ids, desired_neg)
            med_used = []
            easy_used = []
        else:
            hard_used = hard_chunk_ids.copy()
            rem = desired_neg - len(hard_used)

            # Medium negatives via BM25
            tok_q = q_text.split()
            scores = bm25.get_scores(tok_q)
            ranked_idx = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )
            med_candidates = [
                chunk_ids[i] for i in ranked_idx[med_offset : med_offset + med_top_k]
                if chunk_map[chunk_ids[i]] not in pos_docs
                and chunk_ids[i] not in hard_used
            ]
            m_needed = min(med_cap * len(pos_chunk_ids), rem)
            med_used = med_candidates[:m_needed]
            rem -= len(med_used)

            # Easy negatives by query-term exclusion
            q_terms = {t for t in tok_q if len(t) > 3}
            easy_candidates = [
                cid for cid in chunk_ids
                if chunk_map[cid] not in pos_docs
                and cid not in hard_used
                and cid not in med_used
                and all(t.lower() not in chunk_texts[cid].lower() for t in q_terms)
            ]
            take = min(rem, len(easy_candidates))
            easy_used = random.sample(easy_candidates, take) if take > 0 else []
            rem -= len(easy_used)

            # Fallback negatives
            if rem > 0:
                fallback = [
                    cid for cid in chunk_ids
                    if chunk_map[cid] not in pos_docs
                    and cid not in hard_used
                    and cid not in med_used
                    and cid not in easy_used
                ]
                fb_take = min(rem, len(fallback))
                fb = random.sample(fallback, fb_take) if fb_take > 0 else []
                easy_used.extend(fb)
                rem -= len(fb)
                if rem > 0:
                    print(f"[WARN] qid {qid}: short {rem} negatives")

        n_annotated += len(hard_used)
        n_medium += len(med_used)
        n_easy += len(easy_used)

        rows.extend((qid, cid, 1) for cid in pos_chunk_ids)
        rows.extend((qid, cid, 0) for cid in hard_used + med_used + easy_used)

    # 4) Save output
    df_out = pd.DataFrame(rows, columns=["qid", "chunk_id", "label"])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, sep="\t", index=False)

    print(f"[✓] Written {len(df_out)} rows to {output_path}")
    print(
        f"Negatives breakdown: annotated={n_annotated}, "
        f"medium={n_medium}, easy={n_easy}"
    )
    print(f"Total positives={len(df_out[df_out.label == 1])}")

    return df_out


def build_ce_dataset(
    qrels_path: str,
    pos_labels: list,
    neg_labels: list,
    corpus_path: str,
    queries_path: str,
    output_path: str,
    qid_filter: set = None,
    neg_ratio: int = 6,
    med_cap: int = 3,
    med_offset: int = 10,
    med_top_k: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a binary‑label cross‑encoder dataset where `neg_ratio` controls the total
    number of negatives per positive, including both annotated and mined negatives.

    Parameters
    ----------
    qrels_path : str
        Path to TSV with columns [qid, run, doc_id, label].
    corpus_path : str
        Path to JSON mapping doc_id -> full document text.
    queries_path : str
        Path to CSV with columns ['topic_id', 'Query'].
    output_path : str
        Destination TSV file (qid, doc_id, label).
    qid_filter : set, optional
        If provided, only include queries in this set.
    neg_ratio : int, optional
        Total number of negatives (annotated + mined) per positive (default: 6).
    med_cap : int, optional
        Maximum BM25‑mined negatives per positive (default: 3).
    med_top_k : int, optional
        How many top BM25 hits to consider for medium negatives (default: 150).
    seed : int, optional
        RNG seed for reproducibility (default: 42).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['qid', 'doc_id', 'label'].
    """
    random.seed(seed)

    # Counters for negative types
    n_annotated = 0
    n_medium = 0
    n_easy = 0

    # 1) Load relevance judgments
    df_q = pd.read_csv(
        qrels_path, sep="\t", header=None,
        names=["qid", "run", "doc_id", "label"]
    )
    df_q["qid"] = df_q["qid"].astype(str)
    df_q["doc_id"] = df_q["doc_id"].astype(str)

    positives = (
        df_q[df_q.label.isin(pos_labels)]
        .groupby("qid")["doc_id"]
        .apply(list)
        .to_dict()
    )
    annotated_negs = (
        df_q[df_q.label.isin(neg_labels)]
        .groupby("qid")["doc_id"]
        .apply(list)
        .to_dict()
    )

    # 2) Load corpus and build BM25 index
    doc_ids, docs = get_legal_dataset(corpus_path)
    # corpus is a mapping of doc_id -> full document text
    corpus = dict(zip(doc_ids, docs))
    
    tokenized_docs = [corpus[d].split() for d in tqdm(doc_ids, desc="Tokenizing corpus")]
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_docs)
    print("BM25 index built.")

    # 3) Load queries
    qids, queries = get_legal_queries(queries_path)
    query_dict = dict(zip(qids, queries))

    rows = []

    for qid, q_text in tqdm(query_dict.items(), desc="Processing queries"):
        if qid_filter and qid not in qid_filter:
            continue
        pos_list = positives.get(qid, [])
        if not pos_list:
            continue

        # Determine total negatives needed
        desired_neg = neg_ratio * len(pos_list)
        if desired_neg == 0:
            # Take only annotated negatives when desired_neg is 0
            desired_neg = len(annotated_negs.get(qid, []))

        # Fetch all annotated (hard) negatives
        hard_pool = annotated_negs.get(qid, [])
        if len(hard_pool) >= desired_neg:
            # If there are more annotated negatives than needed, sample them
            hard_used = random.sample(hard_pool, desired_neg)
            med_used = []
            easy_used = []
        else:
            # Otherwise take all annotated and mine the rest
            hard_used = hard_pool
            rem = desired_neg - len(hard_used)

            # 3a) Mine medium negatives via BM25
            tok_q = q_text.split()
            scores = bm25.get_scores(tok_q)
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            med_candidates = [
                doc_ids[i]
                for i in ranked[med_offset:med_offset + med_top_k]
                if doc_ids[i] not in pos_list and doc_ids[i] not in hard_used
            ]
            m_needed = min(med_cap * len(pos_list), rem)
            med_used = med_candidates[:m_needed]
            rem -= len(med_used)

            # 3b) Mine easy negatives by excluding query terms
            q_terms = {t for t in tok_q if len(t) > 3}
            easy_candidates = [
                d for d in doc_ids
                if d not in pos_list
                and d not in hard_used
                and d not in med_used
                and all(t.lower() not in corpus[d].lower() for t in q_terms)
            ]
            take = min(rem, len(easy_candidates))
            easy_used = random.sample(easy_candidates, take) if take > 0 else []
            rem -= len(easy_used)

            # 3c) Fallback to any remaining docs if still short
            if rem > 0:
                fallback = [
                    d for d in doc_ids
                    if d not in pos_list
                    and d not in hard_used
                    and d not in med_used
                    and d not in easy_used
                ]
                fb_take = min(rem, len(fallback))
                fb = random.sample(fallback, fb_take) if fb_take > 0 else []
                easy_used.extend(fb)
                rem -= len(fb)
                if rem > 0:
                    print(f"[WARN] qid {qid}: short {rem} negatives after fallback")

        # Update counters
        n_annotated += len(hard_used)
        n_medium += len(med_used)
        n_easy += len(easy_used)

        # Append positives and negatives to rows
        rows.extend((qid, d, 1) for d in pos_list)
        rows.extend((qid, d, 0) for d in hard_used + med_used + easy_used)

    # 4) Create DataFrame and save
    df_out = pd.DataFrame(rows, columns=["qid", "doc_id", "label"])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, sep="\t", index=False)
    print(f"[✓] Written {len(df_out)} rows to {output_path}")
    print(f"Negatives breakdown: annotated={n_annotated}, medium={n_medium}, easy={n_easy}")
    print(f"Total positives={len(df_out[df_out.label == 1])}")

    return df_out


def main_create_scenario_datasets():
    """
    Generate BCE cross‑encoder TSVs for four scenarios:
      S1: random split, only annotated negatives
      S2: random split, annotated + 6× extra negatives
      S3: query‑wise split, only annotated negatives
      S4: query‑wise split, annotated + extras at 2× and 3×
    """
    base = Path(STORAGE_DIR) / "legal_ir" / "data"
    # ann  = base / "annotations" / "qrels_py.tsv"
    ann  = base / "annotations" / "inpars_mistral-small-2501_qrels_Q1.tsv"
    corp = base / "corpus" / "corpus_mistral_summaries_1024.jsonl"
    qry  = base / "corpus" / "inpars_mistral-small-2501_queries_Q1.tsv"
    out  = base / "datasets" / "cross_encoder"
    out.mkdir(parents=True, exist_ok=True)

    seed = 42
    # # --- S1 & S2: random (basic) split ---
    # for scenario, neg_ratio in [("S1", 0), ("S2", 6)]:
    #     # 1) build full dataset
    #     df_full = build_ce_dataset(
    #         qrels_path=str(ann),
    #         corpus_path=str(corp),
    #         queries_path=str(qry),
    #         output_path=str(out / f"bce_{scenario}_full.tsv"),
    #         qid_filter=None,
    #         neg_ratio=neg_ratio,
    #         seed=seed
    #     )

    #     # 2) random train/dev/test split 70/15/15
    #     # Note: this is a random split, not query-wise
    #     df_train, df_temp = train_test_split(df_full, test_size=0.3, random_state=seed, shuffle=True)
    #     df_dev, df_test = train_test_split(df_temp, test_size=0.5, random_state=seed, shuffle=True)

    #     # 3) save splits
    #     df_train.to_csv(out / f"bce_{scenario}_train.tsv", sep="\t", index=False)
    #     df_dev.  to_csv(out / f"bce_{scenario}_dev.tsv",   sep="\t", index=False)
    #     df_test.to_csv(out / f"bce_{scenario}_test.tsv",  sep="\t", index=False)

    # # --- S3: query‑wise split, only annotated negatives ---
    # for split_name, qids in [("train", train_qids), ("dev", dev_qids), ("test", test_qids)]:
    #     build_ce_dataset(
    #         qrels_path=str(ann),
    #         corpus_path=str(corp),
    #         queries_path=str(qry),
    #         output_path=str(out / f"bce_S3_{split_name}.tsv"),
    #         qid_filter=qids,
    #         neg_ratio=0,
    #         seed=seed
    #     )

    # # --- S4: query‑wise split + extras at 2× and 3× ---
    # for ratio in (2, 3):
    #     for split_name, qids in [("train", train_qids), ("dev", dev_qids), ("test", test_qids)]:
    #         build_ce_dataset(
    #             qrels_path=str(ann),
    #             corpus_path=str(corp),
    #             queries_path=str(qry),
    #             output_path=str(out / f"bce_S4_r{ratio}_{split_name}.tsv"),
    #             qid_filter=qids,
    #             neg_ratio=ratio,
    #             seed=seed
    #         )
    
    train_qids = load_qids(base / "qids_inpars_train.txt")
    dev_qids   = load_qids(base / "qids_inpars_dev.txt")
    test_qids  = load_qids(base / "qids_inpars_test.txt")
    for split_name, qids in [("train", train_qids), ("dev", dev_qids), ("test", test_qids)]:
        # build_ce_dataset(
        #     qrels_path=str(ann),
        #     pos_labels=[1],
        #     neg_labels=[0],
        #     corpus_path=str(corp),
        #     queries_path=str(qry),
        #     output_path=str(out / f"bce_6x_synthetic_{split_name}.tsv"),
        #     qid_filter=qids,
        #     neg_ratio=6,
        #     seed=seed
        # )
        build_ce_dataset(
            qrels_path=str(ann),
            pos_labels=[1],
            neg_labels=[0],
            corpus_path=str(corp),
            queries_path=str(qry),
            output_path=str(out / f"bce_6x_inpars_summary_1024_{split_name}.tsv"),
            qid_filter=qids,
            neg_ratio=6,
            seed=seed
        )

if __name__ == "__main__":
    main_create_scenario_datasets()
