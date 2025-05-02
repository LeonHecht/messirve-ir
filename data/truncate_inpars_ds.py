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
from src.utils.retrieval_utils import get_legal_queries


def truncate_qrels():
    qrels_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "inpars_mistral-small-2501_qrels.tsv")

    rows_cleaned = []

    with open(qrels_path, "r") as f:
        rows = f.readlines()
    
    for row in rows:
        qid = row.split("\t")[0]
        if qid.endswith("Q1"):
            rows_cleaned.append(row)

    with open(os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "inpars_mistral-small-2501_qrels_Q1.tsv"), "w") as f:
        for row in rows_cleaned:
            f.write(row)


def truncate_queries():
    queries_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "inpars_mistral-small-2501_queries.tsv")
    qids, queries = get_legal_queries(queries_path)

    with open(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "inpars_mistral-small-2501_queries_Q1.tsv"), "w") as f:
        for qid, query in zip(qids, queries):
            if qid.endswith("Q1"):
                f.write(f"{qid}\t{query}\n")


def truncate_query_split():
    train_queries_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "qids_inpars_train.txt")
    train_queries_path_new = os.path.join(STORAGE_DIR, "legal_ir", "data", "qids_inpars_train_Q1.txt")

    dev_queries_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "qids_inpars_dev.txt")
    dev_queries_path_new = os.path.join(STORAGE_DIR, "legal_ir", "data", "qids_inpars_dev_Q1.txt")

    test_queries_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "qids_inpars_test.txt")
    test_queries_path_new = os.path.join(STORAGE_DIR, "legal_ir", "data", "qids_inpars_test_Q1.txt")

    train_qids = []
    dev_qids = []
    test_qids = []

    with open(train_queries_path, "r") as f:
        train_qids = f.readlines()

    with open(dev_queries_path, "r") as f:
        dev_qids = f.readlines()

    with open(test_queries_path, "r") as f:
        test_qids = f.readlines()

    train_qids = [qid.strip() for qid in train_qids if qid.strip().endswith("Q1")]
    dev_qids = [qid.strip() for qid in dev_qids if qid.strip().endswith("Q1")]
    test_qids = [qid.strip() for qid in test_qids if qid.strip().endswith("Q1")]

    with open(train_queries_path_new, "w") as f:
        for qid in train_qids:
            f.write(f"{qid}\n")

    with open(dev_queries_path_new, "w") as f:
        for qid in dev_qids:
            f.write(f"{qid}\n")

    with open(test_queries_path_new, "w") as f:
        for qid in test_qids:
            f.write(f"{qid}\n")


if __name__ == "__main__":
    truncate_query_split()

