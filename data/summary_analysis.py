import re
import os
import pandas as pd
import sys


def configure_python_path():
    project_root = os.path.abspath(
        os.path.join('..')
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Apply the path tweak before any project imports
configure_python_path()

from config.config import STORAGE_DIR
from src.utils.retrieval_utils import get_legal_dataset

headers = ["Identificación", "Hechos clave", "Pretensiones y agravios",
               "Cuestiones sometidas", "Decisión", "Fundamentos esenciales",
               "Resultado y efectos", "Citas relevantes"]


def parse_summary(txt):
    present = {h: bool(h + '\n' in txt) for h in headers}
    ambig = txt.count("AMBIGUO")
    return {**present, "ambiguous": ambig}


def main():
    dids, docs = get_legal_dataset(os.path.join(STORAGE_DIR, 'legal_ir', 'data', 'corpus', 'corpus_mistral_summaries_1024.jsonl'))
    df = pd.DataFrame([parse_summary(doc) for doc in docs],
                      index=dids).reset_index().rename(columns={"index": "did"})
    
    df.describe().to_csv(os.path.join(STORAGE_DIR, 'legal_ir', 'data', 'summary_analysis.csv'), index=True)
    df[headers].mean().to_csv(os.path.join(STORAGE_DIR, 'legal_ir', 'data', 'summary_analysis_mean.csv'), index=True)

    print(df.head())


if __name__ == "__main__":
    main()
    print("Summary analysis completed and saved.")