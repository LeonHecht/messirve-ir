import argparse
import math
import pandas as pd
from pathlib import Path
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


def shannon_entropy(counts):
    """counts: iterable of non-negative ints; returns entropy in bits."""
    total = sum(counts)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent

def load_qrels(path: str) -> pd.DataFrame:
    """
    Loads qrels with columns: qid, run, doc_id, label.
    Robust to a header or no header. Forces types to str except label->int.
    """
    try:
        df = pd.read_csv(path, sep="\t", header=None, names=["qid","run","doc_id","label"], dtype=str)
        # If first row looks like header strings, try reading with header=0
        if df.iloc[0]["qid"] == "qid" or df.iloc[0]["label"] == "label":
            df = pd.read_csv(path, sep="\t", header=0, dtype=str)
            # try to align column names
            cols = list(df.columns)
            rename_map = {}
            # Try common variants
            for c in cols:
                lc = c.lower()
                if lc in ["qid","query_id","topic_id","id"]:
                    rename_map[c] = "qid"
                elif lc in ["doc_id","docid","did","pid"]:
                    rename_map[c] = "doc_id"
                elif lc in ["label","rel","relevance","gain"]:
                    rename_map[c] = "label"
                elif lc in ["run","q0","iter"]:
                    rename_map[c] = "run"
            df = df.rename(columns=rename_map)
            # Ensure missing 'run'
            if "run" not in df.columns:
                df["run"] = "Q0"
            df = df[["qid","run","doc_id","label"]]
    except Exception:
        # Fallback strict read
        df = pd.read_csv(path, sep="\t", header=None, names=["qid","run","doc_id","label"], dtype=str)

    # Types
    df["qid"] = df["qid"].astype(str)
    df["doc_id"] = df["doc_id"].astype(str)
    # Coerce label to int, non-numeric -> NaN -> drop
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    return df

def compute_entropy_table(qrels_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for qid, grp in qrels_df.groupby("qid"):
        counts = grp["label"].value_counts().to_dict()
        # Ensure 0..3 keys present for convenience
        c0 = int(counts.get(0, 0))
        c1 = int(counts.get(1, 0))
        c2 = int(counts.get(2, 0))
        c3 = int(counts.get(3, 0))
        n = c0 + c1 + c2 + c3
        ent = shannon_entropy([c0, c1, c2, c3])
        maj_label = max(range(4), key=lambda r: [c0, c1, c2, c3][r]) if n > 0 else None
        purity = (max(c0, c1, c2, c3) / n) if n > 0 else 0.0
        rows.append({
            "qid": qid,
            "n_judged": n,
            "count_0": c0,
            "count_1": c1,
            "count_2": c2,
            "count_3": c3,
            "entropy_bits": ent,      # 0..log2(4)=2
            "purity": purity,         # 0..1
            "majority_label": maj_label
        })
    return pd.DataFrame(rows).sort_values(by=["entropy_bits","n_judged"], ascending=[False, False])

def summarize(ent_df: pd.DataFrame):
    if ent_df.empty:
        print("No queries found.")
        return
    print("=== Label Entropy per Query (Summary) ===")
    print(f"Queries: {len(ent_df)}")
    print(f"n_judged per query:   mean={ent_df['n_judged'].mean():.2f}  median={ent_df['n_judged'].median():.0f}")
    print(f"entropy (bits):       mean={ent_df['entropy_bits'].mean():.3f}  median={ent_df['entropy_bits'].median():.3f}  max={ent_df['entropy_bits'].max():.3f}")
    print(f"purity:               mean={ent_df['purity'].mean():.3f}  median={ent_df['purity'].median():.3f}")
    # Show a few highest-entropy queries
    top = ent_df.sort_values("entropy_bits", ascending=False).head(5)
    print("\nTop-5 highest entropy queries (potentially noisy):")
    for _, r in top.iterrows():
        print(f"  qid={r['qid']}  H={r['entropy_bits']:.3f}  n={int(r['n_judged'])}  counts=[{int(r['count_0'])},{int(r['count_1'])},{int(r['count_2'])},{int(r['count_3'])}]")

def main():
    qrels_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "qrels_54.tsv")
    df = load_qrels(qrels_path)

    ent = compute_entropy_table(df)

    out_csv_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "qrels_54_entropy.csv")
    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)

    ent.to_csv(out_csv_path, index=False)
    summarize(ent)
    print(f"\n[✓] Wrote per-query entropy table to: {out_csv_path}")

if __name__ == "__main__":
    main()
