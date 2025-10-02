#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, math, random
from collections import defaultdict
import numpy as np
import os
import sys

def configure_python_path():
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Apply the path tweak before any project imports
configure_python_path()

from config.config import STORAGE_DIR

# ---------- I/O ----------
def read_qrels(path):
    qrels = defaultdict(dict)  # qrels[qid][docid] = label(int)
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

def read_run(path):
    run = defaultdict(list)  # run[qid] = [(rank, docid, score)]
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank, score, _ = parts[:6]
            run[qid].append((int(rank), docid, float(score)))
    # sort and keep docids
    ranked = {}
    for q in run:
        run[q].sort(key=lambda x: x[0])
        ranked[q] = [d for _, d, _ in run[q]]
    return ranked

# ---------- metrics ----------
def dcg_at_k(labels, k):
    dcg = 0.0
    for i, rel in enumerate(labels[:k]):
        gain = (2**rel - 1)
        dcg += gain / math.log2(i+2)  # pos 1.. -> log2(2..)
    return dcg

def ndcg10_for_qid(ranked_docs, qrels_q):
    gains = [qrels_q.get(d, 0) for d in ranked_docs]
    dcg = dcg_at_k(gains, 10)
    ideal = sorted(qrels_q.values(), reverse=True)
    idcg = dcg_at_k(ideal, 10)
    return 0.0 if idcg == 0 else dcg / idcg

# ---------- bootstrap ----------
def bootstrap_ci(scores, B=2000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(42)
    arr = np.array(scores, dtype=float)
    n = len(arr)
    samples = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        samples.append(arr[idx].mean())
    lo, hi = np.quantile(samples, [alpha/2, 1-alpha/2])
    return float(arr.mean()), float(lo), float(hi)

# ---------- paired randomization test (quick) ----------
def paired_randomization_test(deltas, reps=10000, rng=None):
    rng = rng or random.Random(42)
    observed = abs(sum(deltas))
    count = 0
    for _ in range(reps):
        s = 0.0
        for d in deltas:
            s += d if rng.random() < 0.5 else -d
        if abs(s) >= observed:
            count += 1
    return (count + 1) / (reps + 1)  # p-value

def main():
    qrels_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "qrels_54.tsv")
    qrels = read_qrels(qrels_path)

    run_path = "predictions_BM25_ds-54_corpus_NEW.tsv"
    run1 = read_run(os.path.join("..", "src", run_path))

    iterations = 2000

    # per-query scores for run1
    qids = sorted(set(qrels.keys()) & set(run1.keys()))
    s1 = []
    for q in qids:
        s1.append(ndcg10_for_qid(run1[q], qrels[q]))
    m1, lo1, hi1 = bootstrap_ci(s1, B=iterations)

    print(f"Run1: {run_path}")
    print(f"nDCG@10 mean={m1:.4f}  95% CI= [{lo1:.4f}, {hi1:.4f}]  (width ±{(hi1-lo1)/2:.4f})")
    print(f"Queries used: {len(qids)}")

    # if args.run2:
    #     run2 = read_run(args.run2)
    #     qids2 = sorted(set(qids) & set(run2.keys()))
    #     s1 = [ndcg10_for_qid(run1[q], qrels[q]) for q in qids2]
    #     s2 = [ndcg10_for_qid(run2[q], qrels[q]) for q in qids2]
    #     deltas = [b - a for a, b in zip(s1, s2)]
    #     # bootstrap CI on deltas
    #     mD, loD, hiD = bootstrap_ci(deltas, B=args.B)
    #     p = paired_randomization_test(deltas)
    #     print(f"\nRun2: {args.run2}")
    #     print(f"Δ nDCG@10 (run2 - run1): mean={mD:.4f}  95% CI= [{loD:.4f}, {hiD:.4f}]")
    #     print(f"Paired randomization p-value: {p:.4f}  (lower is more significant)")
    #     print(f"Queries used for comparison: {len(qids2)}")

if __name__ == "__main__":
    """
    Run1: predictions_BM25_ds-54_corpus_NEW.tsv
    nDCG@10 mean=0.1896 95% CI= [0.1417, 0.2385] (width ±0.0484)
    Queries used: 54
    """
    main()
