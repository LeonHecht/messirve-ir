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

def configure_python_path():
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    print(f"Adding {project_root} to sys.path")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

configure_python_path()

from config.config import STORAGE_DIR
from src.utils.retrieval_utils import get_legal_dataset


# utility to load qid lists
def load_qids(path):
    return set(pd.read_csv(path, header=None)[0].tolist())


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
    corpus = {doc_id: doc for doc_id, doc in zip(doc_ids, docs)}
    
    tokenized_docs = [corpus[d].split() for d in tqdm(doc_ids, desc="Tokenizing corpus")]
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_docs)
    print("BM25 index built.")

    # 3) Load queries
    if queries_path.endswith(".tsv"):
        df_queries = pd.read_csv(queries_path, sep="\t", header=None)
    elif queries_path.endswith(".csv"):
        df_queries = pd.read_csv(queries_path)
    else:
        raise ValueError(f"Unsupported file format: {queries_path}")
    query_dict = dict(zip(df_queries.iloc[:, 0], df_queries.iloc[:, 1]))

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
    corp = base / "corpus" / "corpus_py.csv"
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
    
    train_qids = load_qids(base / "qids_inpars_train_Q1.txt")
    dev_qids   = load_qids(base / "qids_inpars_dev_Q1.txt")
    test_qids  = load_qids(base / "qids_inpars_test_Q1.txt")
    for split_name, qids in [("train", train_qids), ("dev", dev_qids), ("test", test_qids)]:
        build_ce_dataset(
            qrels_path=str(ann),
            pos_labels=[1],
            neg_labels=[0],
            corpus_path=str(corp),
            queries_path=str(qry),
            output_path=str(out / f"bce_1x_inpars_{split_name}_Q1.tsv"),
            qid_filter=qids,
            neg_ratio=1,
            seed=seed
        )

if __name__ == "__main__":
    main_create_scenario_datasets()
