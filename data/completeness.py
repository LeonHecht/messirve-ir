#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, csv, os, sys
from collections import defaultdict

# ---------- CONFIG (edit here) ----------
def configure_python_path():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

configure_python_path()
from config.config import STORAGE_DIR  # uses your project config

QRELS_PATH = os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "qrels_54.tsv")

# TREC run files to evaluate (edit/add as needed)
RUN_FILES = [
    os.path.join("..", "src", "predictions_BM25_ds-54_corpus_NEW.tsv"),
    os.path.join("..", "src", "predictions_BGE-v4.tsv"),
    # os.path.join("..", "src", "predictions_BM25_ds-54_corpus.tsv"),
]

OUT_CSV = os.path.join(STORAGE_DIR, "legal_ir", "analysis", "completeness_per_query.csv")
KS = (10, 20, 50, 100)
# ---------------------------------------

def read_qrels(path):
    # qrels: qid \t run \t doc_id \t label
    qrels = defaultdict(dict)           # qrels[qid][docid] = label (int)
    rel_sets = defaultdict(set)         # judged relevant (label>0)
    nrel_sets = defaultdict(set)        # judged non-relevant (label==0)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 4:
                continue
            qid, _, docid, lab = parts[0], parts[1], parts[2], parts[3]
            try:
                lab = int(lab)
            except:
                continue
            qrels[qid][docid] = lab
            if lab > 0:
                rel_sets[qid].add(docid)
            else:
                nrel_sets[qid].add(docid)
    return qrels, rel_sets, nrel_sets

def read_run(path):
    # TREC run: qid Q0 docid rank score tag
    run = defaultdict(list)  # run[qid] = [docid in rank order]
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank, score, tag = parts[:6]
            run[qid].append((int(rank), docid, float(score)))
    # sort by rank
    ranked = {}
    for qid in run:
        run[qid].sort(key=lambda x: x[0])
        ranked[qid] = [d for _, d, _ in run[qid]]
    return ranked

def judged_at_k(run_list, judged_set, k):
    topk = run_list[:k]
    if not topk:
        return 0.0
    judged = sum(1 for d in topk if d in judged_set)
    return judged / float(k)

def dcg_at_k(labels, k):
    dcg = 0.0
    for i, rel in enumerate(labels[:k]):
        gain = (2**rel - 1)
        denom = math.log2(i+2)  # positions 1..k -> log2(2..k+1)
        dcg += gain / denom
    return dcg

def ndcg_at_10(run_list, qrels_q):
    """Standard nDCG@10 (treats unjudged as 0)."""
    gains = [qrels_q.get(docid, 0) for docid in run_list]
    dcg = dcg_at_k(gains, 10)
    ideal_labels = sorted(qrels_q.values(), reverse=True)
    idcg = dcg_at_k(ideal_labels, 10)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def condensed_ndcg_at_10(run_list, qrels_q):
    """
    Condensed nDCG@10:
    - Drop unjudged docs from the ranking first (keep order of judged).
    - Compute DCG@10 over the condensed list.
    - IDCG is the same ideal DCG built from judged labels only.
    """
    judged_ordered = [d for d in run_list if d in qrels_q]  # keep only judged, preserve order
    if not judged_ordered:
        return 0.0
    gains = [qrels_q[d] for d in judged_ordered]
    dcg = dcg_at_k(gains, 10)
    ideal_labels = sorted(qrels_q.values(), reverse=True)
    idcg = dcg_at_k(ideal_labels, 10)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def bpref(run_list, relset, nrelset):
    """
    bpref as in trec_eval:
      For each judged relevant r, add 1 - (# judged nonrel above r)/min(R, N)
      Only counts judged docs seen in the ranking; ignores unjudged.
    """
    R = len(relset)
    N = len(nrelset)
    if R == 0 or N == 0:
        return 0.0
    denom = min(R, N)
    seen_nonrel = 0
    score = 0.0
    seen_rel = 0
    for doc in run_list:
        if doc in nrelset:
            seen_nonrel += 1
        elif doc in relset:
            seen_rel += 1
            score += 1.0 - (seen_nonrel / denom)
    if seen_rel == 0:
        return 0.0
    return score / R

def evaluate_run(run, qrels, rel_sets, nrel_sets, ks=KS):
    per_query = []
    eval_qids = sorted(set(qrels.keys()) & set(run.keys()))
    for qid in eval_qids:
        ranked = run[qid]
        judged_set = set(qrels[qid].keys())
        row = {"qid": qid}
        # judged@k
        for k in ks:
            row[f"judged@{k}"] = judged_at_k(ranked, judged_set, k)
        # metrics
        row["nDCG@10"] = ndcg_at_10(ranked, qrels[qid])
        row["condensed_nDCG@10"] = condensed_ndcg_at_10(ranked, qrels[qid])
        row["bpref"] = bpref(ranked, rel_sets[qid], nrel_sets[qid])
        row["bpref_minus_nDCG10"] = row["bpref"] - row["nDCG@10"]
        row["condensed_minus_nDCG10"] = row["condensed_nDCG@10"] - row["nDCG@10"]
        per_query.append(row)
    return per_query

def summarize(per_query_rows, ks=KS):
    n = len(per_query_rows)
    if n == 0:
        return {}
    avg = {}
    for k in ks:
        avg[f"judged@{k}"] = sum(r[f"judged@{k}"] for r in per_query_rows)/n
    avg["nDCG@10"] = sum(r["nDCG@10"] for r in per_query_rows)/n
    avg["condensed_nDCG@10"] = sum(r["condensed_nDCG@10"] for r in per_query_rows)/n
    avg["bpref"]   = sum(r["bpref"] for r in per_query_rows)/n
    avg["bpref_minus_nDCG10"] = avg["bpref"] - avg["nDCG@10"]
    avg["condensed_minus_nDCG10"] = avg["condensed_nDCG@10"] - avg["nDCG@10"]
    return avg

def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    qrels, rel_sets, nrel_sets = read_qrels(QRELS_PATH)

    all_rows = []
    for run_path in RUN_FILES:
        run = read_run(run_path)
        rows = evaluate_run(run, qrels, rel_sets, nrel_sets)
        for r in rows:
            r["run"] = os.path.basename(run_path)
        all_rows.extend(rows)

        summ = summarize(rows)
        print(f"\n=== Summary for run: {run_path} ===")
        if not summ:
            print("No overlapping queries between run and qrels.")
            continue
        for k in KS:
            print(f"judged@{k}: {summ.get(f'judged@{k}', 0.0):.3f}")
        print(f"nDCG@10: {summ['nDCG@10']:.3f}")
        print(f"condensed_nDCG@10: {summ['condensed_nDCG@10']:.3f}  (Δ vs nDCG: {summ['condensed_minus_nDCG10']:.3f})")
        print(f"bpref  : {summ['bpref']:.3f}")
        print(f"bpref - nDCG@10: {summ['bpref_minus_nDCG10']:.3f}")

    # write per-query CSV
    fieldnames = [
        "run","qid",
        "judged@10","judged@20","judged@50","judged@100",
        "nDCG@10","condensed_nDCG@10","condensed_minus_nDCG10",
        "bpref","bpref_minus_nDCG10",
    ]
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\n[✓] Wrote per-query completeness table to {OUT_CSV}")

if __name__ == "__main__":
    """
    Completeness evidence to quote:
    - If judged@10 is low and bpref >> nDCG@10, pool incompleteness is high.
    - If condensed_nDCG@10 rebounds toward previous scores, the drop was mostly due to unjudged docs.
    """
    main()
