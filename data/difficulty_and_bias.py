#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Difficulty / Bias (D) diagnostics for an IR dataset.

Computes per-query:
- length (tokens), uniq_terms
- BM25 IDF stats: avg_idf, sum_idf, max_idf
- CLARITY score: sum_w p(w|Q) * log( p(w|Q) / p(w|C) )
Optionally joins with per-query nDCG@10 & condensed nDCG@10 from TREC runs,
reports Pearson correlations and quartile analyses.

No argparse; edit CONFIG below.
"""

import os, sys, re, math, csv
from collections import defaultdict, Counter
from statistics import mean
from math import log
from typing import Dict, List, Tuple
import numpy as np

# ---------- CONFIG (edit here) ----------
def configure_python_path():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

configure_python_path()
from config.config import STORAGE_DIR

# Corpus (JSON or JSONL) — uses your helper shape: {"id":..., "text":...} or JSONL lines with those keys
CORPUS_PATH = os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus_NEW.jsonl")

# Queries TSV: qid \t query
QUERIES_PATH = os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "queries_54.tsv")  # change if needed

# Qrels TSV (qid \t run \t doc_id \t label)
QRELS_PATH = os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "qrels_54.tsv")

# Optional TREC run files to compute effectiveness (nDCG@10 and condensed_nDCG@10)
RUN_FILES = [
    # os.path.join("..", "src", "predictions_BM25_ds-54_corpus.tsv"),
    os.path.join("..", "src", "predictions_BGE-v4.tsv"),
    # os.path.join("..", "src", "predictions_BM25_ds-54_corpus_NEW.tsv"),
]

# Output per-query CSV
OUT_CSV = os.path.join(STORAGE_DIR, "legal_ir", "analysis", "difficulty_per_query.csv")
# ---------------------------------------

# --------- Minimal tokenizer (Spanish-friendly-ish) ----------
TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]

# --------- Load corpus (uses your helper format) ----------
def load_corpus(path: str) -> Tuple[List[str], List[str]]:
    """
    Expect .json or .jsonl with fields id / text.
    """
    import json
    ids, texts = [], []
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            corpus_dict = json.load(f)
        for k, v in corpus_dict.items():
            ids.append(str(k))
            texts.append(v if v is not None else "")
    elif path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                did = str(obj.get("id", ""))
                txt = obj.get("text", "") or obj.get("contents", "") or ""
                if not did:
                    continue
                ids.append(did)
                texts.append(txt)
    else:
        raise ValueError("CORPUS_PATH must be .json or .jsonl")
    return ids, texts

# --------- Build corpus stats: collection LM + DF ----------
def build_corpus_stats(doc_texts: List[str]) -> Tuple[Counter, Dict[str,int], int, int]:
    term_counts = Counter()
    df = Counter()
    N_docs = 0
    total_tokens = 0
    for txt in doc_texts:
        toks = tokenize(txt)
        if not toks:
            N_docs += 1
            continue
        total_tokens += len(toks)
        term_counts.update(toks)
        df.update(set(toks))
        N_docs += 1
    return term_counts, df, total_tokens, N_docs

def bm25_idf(df_val: int, N: int) -> float:
    # Robertson/Sparck Jones IDF with +0.5 correction
    return log((N - df_val + 0.5) / (df_val + 0.5) + 1.0)

# --------- Read queries / qrels / runs ----------
def read_queries_tsv(path: str) -> Dict[str, str]:
    q = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            qid, qry = parts[0].strip(), parts[1].strip()
            # skip header-like first row
            if i == 0 and qid.lower() in {"id", "qid"} and qry.lower() in {"query", "queries"}:
                continue
            if not qid:
                continue
            q[str(qid)] = qry
    return q

def read_qrels(path: str) -> Dict[str, Dict[str,int]]:
    qrels = defaultdict(dict)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 4:
                continue
            qid, _, docid, lab = parts[:4]
            try:
                lab = int(lab)
            except:
                continue
            qrels[qid][docid] = lab
    return qrels

def read_run(path: str) -> Dict[str, List[str]]:
    run = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank, score, tag = parts[:6]
            run[qid].append((int(rank), docid, float(score)))
    ranked = {}
    for qid in run:
        run[qid].sort(key=lambda x: x[0])
        ranked[qid] = [d for _, d, _ in run[qid]]
    return ranked

# --------- nDCG and condensed nDCG ----------
def dcg_at_k(labels: List[int], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(labels[:k]):
        gain = (2**rel - 1)
        dcg += gain / math.log2(i+2)
    return dcg

def ndcg_at_10(run_list: List[str], qrels_q: Dict[str,int]) -> float:
    gains = [qrels_q.get(docid, 0) for docid in run_list]
    dcg = dcg_at_k(gains, 10)
    ideal = sorted(qrels_q.values(), reverse=True)
    idcg = dcg_at_k(ideal, 10)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def condensed_ndcg_at_10(run_list: List[str], qrels_q: Dict[str,int]) -> float:
    judged = [d for d in run_list if d in qrels_q]
    if not judged:
        return 0.0
    gains = [qrels_q[d] for d in judged]
    dcg = dcg_at_k(gains, 10)
    ideal = sorted(qrels_q.values(), reverse=True)
    idcg = dcg_at_k(ideal, 10)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

# --------- Clarity score ----------
def clarity_score(query_tokens: List[str], term_counts: Counter, total_tokens: int, eps: float = 1e-12) -> float:
    if not query_tokens:
        return 0.0
    q_counts = Counter(query_tokens)
    L = sum(q_counts.values())
    score = 0.0
    for w, c in q_counts.items():
        p_q = c / L
        p_c = term_counts.get(w, 0) / max(1, total_tokens)
        p_c = max(p_c, eps)
        score += p_q * math.log(p_q / p_c)
    return score  # nats; if you want bits: / math.log(2)

# --------- Correlations ----------
def pearsonr_safe(x: List[float], y: List[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_arr, y_arr = np.array(x, dtype=float), np.array(y, dtype=float)
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0,1])

# --------- Main pipeline ----------
def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    # 1) Load corpus & build stats
    print("[*] Loading corpus and building stats...")
    _, texts = load_corpus(CORPUS_PATH)
    term_counts, df, total_tokens, N_docs = build_corpus_stats(texts)
    print(f"    docs={N_docs:,}  tokens={total_tokens:,}  vocab={len(term_counts):,}")

    # 2) Load queries
    queries = read_queries_tsv(QUERIES_PATH)

    # 3) Precompute per-query features
    rows = {}
    for qid, qtext in queries.items():
        q_toks = tokenize(qtext)
        uniq = set(q_toks)
        idfs = [bm25_idf(df.get(w, 0), N_docs) for w in uniq if w]
        avg_idf = mean(idfs) if idfs else 0.0
        sum_idf = sum(idfs) if idfs else 0.0
        max_idf = max(idfs) if idfs else 0.0
        cl = clarity_score(q_toks, term_counts, total_tokens)
        rows[qid] = {
            "qid": qid,
            "query": qtext,
            "q_len": len(q_toks),
            "uniq_terms": len(uniq),
            "avg_idf": avg_idf,
            "sum_idf": sum_idf,
            "max_idf": max_idf,
            "clarity": cl,
        }

    # 4) Optional: effectiveness from runs (per-query nDCG)
    if RUN_FILES:
        print("[*] Loading qrels and runs for effectiveness...")
        qrels = read_qrels(QRELS_PATH)
        for run_path in RUN_FILES:
            run = read_run(run_path)
            tag = os.path.basename(run_path).replace(".tsv","")
            nds, cnds = [], []
            feat_len, feat_uniq, feat_cl, feat_avgidf = [], [], [], []
            overlap_q = sorted(set(run.keys()) & set(qrels.keys()))
            for qid in overlap_q:
                nd = ndcg_at_10(run[qid], qrels[qid])
                cd = condensed_ndcg_at_10(run[qid], qrels[qid])
                rows[qid][f"nDCG@10[{tag}]"] = nd
                rows[qid][f"condensed_nDCG@10[{tag}]"] = cd

                # collect for correlations
                nds.append(nd); cnds.append(cd)
                feat_len.append(rows[qid]["q_len"])
                feat_uniq.append(rows[qid]["uniq_terms"])
                feat_cl.append(rows[qid]["clarity"])
                feat_avgidf.append(rows[qid]["avg_idf"])

            # Print correlations (feature vs effectiveness)
            r_len = pearsonr_safe(feat_len, nds)
            r_uniq = pearsonr_safe(feat_uniq, nds)
            r_cl  = pearsonr_safe(feat_cl, nds)
            r_idf = pearsonr_safe(feat_avgidf, nds)

            r_len_c = pearsonr_safe(feat_len, cnds)
            r_uniq_c = pearsonr_safe(feat_uniq, cnds)
            r_cl_c  = pearsonr_safe(feat_cl, cnds)
            r_idf_c = pearsonr_safe(feat_avgidf, cnds)

            print(f"\n=== Difficulty correlations with {tag} ===")
            print(f"corr(q_len, nDCG@10)             = {r_len: .3f}")
            print(f"corr(uniq_terms, nDCG@10)        = {r_uniq: .3f}")
            print(f"corr(clarity, nDCG@10)           = {r_cl: .3f}")
            print(f"corr(avg_idf, nDCG@10)           = {r_idf: .3f}")
            print(f"corr(q_len, condensed_nDCG@10)   = {r_len_c: .3f}")
            print(f"corr(uniq_terms, condensed_nDCG) = {r_uniq_c: .3f}")
            print(f"corr(clarity, condensed_nDCG)    = {r_cl_c: .3f}")
            print(f"corr(avg_idf, condensed_nDCG)    = {r_idf_c: .3f}")

            # Quartile buckets by clarity → mean nDCG
            if feat_cl:
                cl_vals = np.array(feat_cl)
                nd_vals = np.array(nds)
                cd_vals = np.array(cnds)
                qs = np.quantile(cl_vals, [0.25, 0.5, 0.75])
                labels = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
                idxs = [
                    cl_vals <= qs[0],
                    (cl_vals > qs[0]) & (cl_vals <= qs[1]),
                    (cl_vals > qs[1]) & (cl_vals <= qs[2]),
                    cl_vals > qs[2],
                ]
                print(f"\n=== nDCG by clarity quartile ({tag}) ===")
                for lab, mask in zip(labels, idxs):
                    if mask.sum() == 0:
                        print(f"{lab}: n/a")
                    else:
                        print(f"{lab}: mean nDCG@10={nd_vals[mask].mean():.3f}  condensed={cd_vals[mask].mean():.3f}")

    # 5) Write per-query CSV
    fieldnames = [
        "qid","query","q_len","uniq_terms","avg_idf","sum_idf","max_idf","clarity"
    ]
    # append any run columns
    # (collect keys from one row)
    extra_cols = []
    for r in rows.values():
        for k in r.keys():
            if k not in fieldnames and k not in extra_cols:
                extra_cols.append(k)
    fieldnames += extra_cols

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for qid in sorted(rows.keys(), key=lambda x: str(x)):
            w.writerow(rows[qid])
    print(f"\n[✓] Wrote per-query difficulty table to {OUT_CSV}")

if __name__ == "__main__":
    main()
