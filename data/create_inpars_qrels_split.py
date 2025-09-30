"""Script that takes total inpars qrels as input and splits it into train, dev, or test qrels."""

import os
import sys
import pandas as pd

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


def split_qrels(split: str):
    qrels_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "inpars_mistral-small-2501_qrels.tsv")
    qrels_df = pd.read_csv(
        qrels_path,
        sep="\t",                # TREC qrels are usually tab-separated
        names=["query_id", "iteration", "doc_id", "relevance"],
        header=None,            # There's no header in qrels files
        dtype={"query_id": str, "iteration": int, "doc_id": str, "relevance": int}
    )

    qids_path = os.path.join(STORAGE_DIR, "legal_ir", "data", f"qids_inpars_{split}.txt")

    with open(qids_path, "r") as f:
        qids = f.read().splitlines()
    qids = [qid.strip() for qid in qids]

    qrels_df = qrels_df[qrels_df["query_id"].isin(qids)]

    qrels_df.to_csv(
        os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", f"inpars_mistral-small-2501_qrels_{split}.tsv"),
        sep="\t",
        index=False,
        header=False
    )


if __name__ == "__main__":
    # Split the qrels into train, dev, and test sets
    split = "test"
    split_qrels(split)