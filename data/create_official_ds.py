import os
import sys
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


def convert_csv_to_tsv(input_csv, output_tsv):
    """
    Convert a CSV file to TSV format.
    """
    df = pd.read_csv(input_csv)
    # column "topic_id" has to be overwritten with the index
    df["topic_id"] = df.index + 1
    df.to_csv(output_tsv, sep="\t", index=False)
    print(f"Converted {input_csv} to {output_tsv}")


def convert_csv_to_jsonl(input_csv, output_jsonl):
    """
    Convert a CSV file to JSONL format.
    """
    df = pd.read_csv(input_csv)
    columns_to_keep = [
        "Codigo",
        "Titulo",
        "text"
    ]
    # remove all columns that are not in columns_to_keep
    df = df[columns_to_keep]
    # rename columns
    df = df.rename(columns={
        "Codigo": "id",
        "Titulo": "title",
        "text": "text"
    })
    # convert id column to string
    df["id"] = df["id"].astype(str)
    df.to_json(output_jsonl, orient="records", lines=True, force_ascii=False)
    print(f"Converted {input_csv} to {output_jsonl}")


def select_qrels_for_meta_annotation():
    qrels_dev_df = pd.read_csv(
        os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "qrels_py_57.tsv"),
        sep="\t",                # TREC qrels are usually tab-separated
        names=["query_id", "iteration", "doc_id", "relevance"],
        header=None,            # There's no header in qrels files
        dtype={"query_id": str, "iteration": int, "doc_id": str, "relevance": int}
    )

    # select 3 random qrels for each query
    qrels_dev_df = qrels_dev_df.groupby("query_id").apply(
        lambda x: x.sample(n=3, random_state=42)
    ).reset_index(drop=True)

    # drop relevance column
    qrels_dev_df = qrels_dev_df.drop(columns=["iteration", "relevance"])

    # sort qrels by query_id
    qrels_dev_df = qrels_dev_df.sort_values(by=["query_id"])

    # save the selected qrels to a new file
    qrels_dev_df.to_csv(
        os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "qrels_57_meta.tsv"),
        sep="\t",
        index=False,
        header=False
    )
    print("Selected qrels for meta annotation saved to qrels_57_meta.tsv")



if __name__ == "__main__":
    # Define the paths for the input and output files
    # input_csv = os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "queries_54.csv")
    # output_tsv = os.path.join(STORAGE_DIR,  "legal_ir", "data", "corpus", "queries_54.tsv")

    # # Convert the CSV file to TSV format
    # convert_csv_to_tsv(input_csv, output_tsv)

    # convert_csv_to_jsonl(
    #     os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus_py.csv"),
    #     os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus.jsonl"),
    # )
    select_qrels_for_meta_annotation()