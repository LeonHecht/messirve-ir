import pandas as pd
from pathlib import Path
import sys
import os

def configure_python_path():
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    print(f"Adding {project_root} to sys.path")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

configure_python_path()

from config.config import STORAGE_DIR

import pandas as pd
from pathlib import Path


def merge_csv_files(file_paths, output_path, start_id=58):
    """
    Merge multiple CSV files, drop duplicate queries, and assign new topic IDs.

    Reads CSVs containing 'topic_id' and 'Query', concatenates them,
    removes duplicate 'Query' rows, reassigns sequential 'topic_id'
    starting from `start_id`, and writes the result to `output_path`.

    Parameters
    ----------
    file_paths : list of str or pathlib.Path
        Paths to the input CSV files.
    output_path : str or pathlib.Path
        Destination path for the merged CSV file.
    start_id : int, optional
        Starting value for the first topic_id (default is 58).

    Returns
    -------
    None
        Writes the merged DataFrame to `output_path`.

    Examples
    --------
    >>> files = ['topics_part1.csv', 'topics_part2.csv',
    ...          'topics_part3.csv']
    >>> merge_csv_files_with_ids(files, 'merged_topics.csv', start_id=58)
    """
    paths = [Path(fp) for fp in file_paths]
    df_list = [pd.read_csv(path) for path in paths]

    # Concatenate, drop duplicate queries, and reset index
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=['Query']).reset_index(drop=True)

    # Assign new topic_id starting from `start_id`
    merged_df['topic_id'] = merged_df.index + start_id

    # Reorder columns and write out
    merged_df = merged_df[['topic_id', 'Query']]
    merged_df.to_csv(output_path, index=False)


def merge_jsonl_files(file_paths, output_path):

    with open(output_path, "w", encoding="utf-8") as fout:
        for file in file_paths:
            with open(file, "r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():  # skip empty lines
                        fout.write(line)

    print(f"Merged {len(file_paths)} files into {output_path}")


if __name__ == "__main__":
    # # Define the paths to the CSV files
    # file_paths = [
    #     Path(STORAGE_DIR) / "legal_ir" / "data" / "corpus" / "queries_150_GPT4.5.csv",
    #     Path(STORAGE_DIR) / "legal_ir" / "data" / "corpus" / "queries_150_GPT4o.csv",
    #     Path(STORAGE_DIR) / "legal_ir" / "data" / "corpus" / "queries_150_GPTo3.csv"
    # ]

    # # Define the output path
    # output_path = Path(STORAGE_DIR) / "legal_ir" / "data" / "corpus" / "synthetic_queries2.csv"

    # # Merge the CSV files
    # merge_csv_files(file_paths, output_path)
    
    # Define the paths to the CSV files
    file_paths = [
        Path(STORAGE_DIR) / "mtrag" / "data" / "datasets" / "dual_encoder" / "bge_finetune_6x_all_dev_baai.jsonl",
        Path(STORAGE_DIR) / "mtrag" / "data" / "datasets" / "dual_encoder" / "bge_finetune_6x_all_test_baai.jsonl",
    ]

    # Define the output path
    output_path = Path(STORAGE_DIR) / "mtrag" / "data" / "datasets" / "dual_encoder" / "bge_finetune_6x_all_dev_test_baai.jsonl"

    # Merge the CSV files
    merge_jsonl_files(file_paths, output_path)