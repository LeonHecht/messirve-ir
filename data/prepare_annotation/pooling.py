"""
This file serves the purpose to merge the retrieval results of
retrieve_for_annotation.py several retrieval models for annotation.
"""

import pandas as pd
import os
import sys

def configure_python_path():
    """
    Add the project root directory to sys.path.

    This function finds the directory two levels up from this file
    (the repo root) and inserts it at the front of sys.path so that
    `config.config` can be imported without errors.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

configure_python_path()

from src.utils.retrieval_utils import get_legal_queries
from config.config import STORAGE_DIR


def read_retrieved_docs(path):
    df = pd.read_csv(path, dtype=str)
    return df


def main():
    """
    Input: Several different csv files with retrieved top k docs from different models
           for n queries.
           df with columns: Query, doc_sim, Codigo, doc_text
        
    Output: A single csv file with top k retrieved docs for each query.
            df with columns: Query, Rank, Codigo, doc_text
    """
    
    # I want 30 docs for each query for annotation 
    top_k = 40

    # get queries list
    queries_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "consultas_sinteticas_380.tsv")
    qids, _ = get_legal_queries(queries_path, header=0)

    model_names = ["bm25", "bge-m3", "Jinja"]
    paths = []
    for model_name in model_names:
        paths.append(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", f"retrieved_docs_{model_name}.csv"))

    in_dfs = []
    for path in paths:
        df = read_retrieved_docs(path)
        in_dfs.append(df)
    
    # for each query there will be one out_df with top_k entries (docs)
    out_dfs = []
    for qid in qids:
        # Get sub-dfs for current query
        sub_in_dfs = []
        for df in in_dfs:
            sub_df = df[df["query_id"] == qid]
            # assert len of sub_df is top_k
            assert len(sub_df) == top_k
            sub_in_dfs.append(sub_df)

        # Perform intersection between the 3 dfs
        # Convert rows to sets of tuples
        # CAUTION: Only works for 3 dfs!
        # Extract the "Codigo" column as sets
        set1 = set(sub_in_dfs[0]["Codigo"])
        set2 = set(sub_in_dfs[1]["Codigo"])
        set3 = set(sub_in_dfs[2]["Codigo"])

        # Find intersection based on "Codigo"
        common_codigos = set1 & set2 & set3

        # Filter rows in the first DataFrame based on the common "Codigo" values
        out_df = sub_in_dfs[0][sub_in_dfs[0]["Codigo"].isin(common_codigos)]
 
        # Remove overlapping rows from original DataFrames based on "Codigo"
        sub_in_dfs[0] = sub_in_dfs[0][~sub_in_dfs[0]["Codigo"].isin(common_codigos)]
        sub_in_dfs[1] = sub_in_dfs[1][~sub_in_dfs[1]["Codigo"].isin(common_codigos)]
        sub_in_dfs[2] = sub_in_dfs[2][~sub_in_dfs[2]["Codigo"].isin(common_codigos)]
        
        # Calculate docs/rows left to fill out_df up to top_k
        docs_left = top_k - len(out_df)
        if docs_left == 0:
            out_dfs.append(out_df)
            continue
            
        # Perform intersection between df 1 and 2
        set1 = set(sub_in_dfs[0]["Codigo"])
        set2 = set(sub_in_dfs[1]["Codigo"])

        # Find intersection based on "Codigo"
        common_codigos = set1 & set2

        # Filter rows in the first DataFrame based on the common "Codigo" values
        filtered_rows = sub_in_dfs[0][sub_in_dfs[0]["Codigo"].isin(common_codigos)].head(docs_left)
        # Append the filtered rows to out_df
        out_df = pd.concat([out_df, filtered_rows], ignore_index=True)
 
        # Remove overlapping rows from original DataFrames based on "Codigo"
        sub_in_dfs[0] = sub_in_dfs[0][~sub_in_dfs[0]["Codigo"].isin(common_codigos)]
        sub_in_dfs[1] = sub_in_dfs[1][~sub_in_dfs[1]["Codigo"].isin(common_codigos)]

        # Calculate docs/rows left to fill out_df up to top_k
        docs_left = top_k - len(out_df)
        if docs_left == 0:
            out_dfs.append(out_df)
            continue

        # Perform intersection between df 2 and 3
        set2 = set(sub_in_dfs[1]["Codigo"])
        set3 = set(sub_in_dfs[2]["Codigo"])

        # Find intersection based on "Codigo"
        common_codigos = set2 & set3

        # Filter rows in the first DataFrame based on the common "Codigo" values
        filtered_rows = sub_in_dfs[1][sub_in_dfs[1]["Codigo"].isin(common_codigos)].head(docs_left)
        # Append the filtered rows to out_df
        out_df = pd.concat([out_df, filtered_rows], ignore_index=True)
 
        # Remove overlapping rows from original DataFrames based on "Codigo"
        sub_in_dfs[1] = sub_in_dfs[1][~sub_in_dfs[1]["Codigo"].isin(common_codigos)]
        sub_in_dfs[2] = sub_in_dfs[2][~sub_in_dfs[2]["Codigo"].isin(common_codigos)]

        # Calculate docs/rows left to fill out_df up to top_k
        docs_left = top_k - len(out_df)
        if docs_left == 0:
            out_dfs.append(out_df)
            continue

        # Perform intersection between df 1 and 3
        set1 = set(sub_in_dfs[0]["Codigo"])
        set3 = set(sub_in_dfs[2]["Codigo"])

        # Find intersection based on "Codigo"
        common_codigos = set1 & set3

        # Filter rows in the first DataFrame based on the common "Codigo" values
        filtered_rows = sub_in_dfs[0][sub_in_dfs[0]["Codigo"].isin(common_codigos)].head(docs_left)
        # Append the filtered rows to out_df
        out_df = pd.concat([out_df, filtered_rows], ignore_index=True)
 
        # Remove overlapping rows from original DataFrames based on "Codigo"
        sub_in_dfs[0] = sub_in_dfs[0][~sub_in_dfs[0]["Codigo"].isin(common_codigos)]
        sub_in_dfs[2] = sub_in_dfs[2][~sub_in_dfs[2]["Codigo"].isin(common_codigos)]

        docs_left = top_k - len(out_df)

        curr_index = 0
        while docs_left > 0:
            out_df.loc[len(out_df)] = sub_in_dfs[curr_index].iloc[0]
            # Remove the first row from sub_df
            sub_in_dfs[curr_index] = sub_in_dfs[curr_index].iloc[1:]
            docs_left -= 1

            # let curr_index oscillate between 0 and 2
            if curr_index < 2:
                curr_index += 1
            else:
                curr_index = 0

        out_dfs.append(out_df)

    out_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "temp_pooling_output")

    # write out dfs to csv
    for out_df in out_dfs:
        assert len(out_df) == top_k
        qid = out_df["query_id"].iloc[0]
        path = os.path.join(out_path, f"merged_retrieved_docs_{qid}.csv")
        out_df.to_csv(path, index=False)
        print(f"Merged retrieved documents saved to {path}")


if __name__ == '__main__':
    main()