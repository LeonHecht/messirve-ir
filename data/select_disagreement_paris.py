"""
select_disagreement_pairs.py
============================
Genera la lista de pares *qid–did* cuya divergencia entre el
rango “oro” (basado en las etiquetas del `qrels`) y el rango
del modelo (archivo `predictions.tsv`) es máxima.

Uso
----
$ python select_disagreement_pairs.py \
    --qrels  path/to/qrels.txt \
    --pred   path/to/predictions.tsv \
    --k      10 \
    --out    pairs_to_annotate.tsv
"""
from __future__ import annotations

import argparse
import pandas as pd
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


def read_qrels(path: str) -> pd.DataFrame:
    """
    Lee un qrels de cuatro columnas (qid, run, did, label) y
    asigna un rango 'gold_rank' según la etiqueta.

    Parameters
    ----------
    path : str
        Ruta al archivo qrels.

    Returns
    -------
    pd.DataFrame
        Columnas: qid, did, gold_rank
    """
    cols = ["qid", "run", "did", "label"]
    qrels = pd.read_csv(path, sep=r"\s+", names=cols, dtype={"qid": str, "did": str})
    # Orden estable por qid, label desc, posición original
    qrels["gold_rank"] = (
        qrels.groupby("qid", sort=False)["label"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return qrels[["qid", "did", "gold_rank"]]


def read_preds(path: str) -> pd.DataFrame:
    """
    Lee predictions.tsv de seis columnas (qid, Q0, did, rank, score, run_name)
    y devuelve un DataFrame con el rango del modelo.

    Parameters
    ----------
    path : str
        Ruta al archivo de predicciones.

    Returns
    -------
    pd.DataFrame
        Columnas: qid, did, model_rank
    """
    cols = ["qid", "Q0", "did", "model_rank", "score", "run_name"]
    preds = pd.read_csv(path, sep=r"\s+", names=cols, dtype={"qid": str, "did": str})
    return preds[["qid", "did", "model_rank"]]


def select_pairs(qrels: pd.DataFrame,
                 preds: pd.DataFrame,
                 k: int,
                 max_rank_for_missing: int = 31) -> pd.DataFrame:
    """
    Same idea as antes, but:
    - left-join to keep all gold docs
    - fill missing model ranks with `max_rank_for_missing`
    - guarantees ≥ k candidates per qid
    """
    merged = pd.merge(qrels, preds,
                      on=["qid", "did"],
                      how="left")          # keep all qrels rows
    merged["model_rank"].fillna(max_rank_for_missing, inplace=True)
    merged["abs_diff"] = (merged["gold_rank"] - merged["model_rank"]).abs()

    topk = (
        merged.sort_values(["qid", "abs_diff"], ascending=[True, False])
              .groupby("qid")
              .head(k)
              .reset_index(drop=True)
    )
    return topk[["qid", "did"]]


def main() -> None:
    qrels_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "qrels_54.tsv")
    preds_path = "../src/predictions_BGE-m3.tsv"


    qrels_df = read_qrels(qrels_path)
    preds_df = read_preds(preds_path)
    pairs_df = select_pairs(qrels_df, preds_df, k=10)
    # convert qid column to int
    pairs_df["qid"] = pairs_df["qid"].astype(int)
    # sort by qid
    pairs_df = pairs_df.sort_values(by="qid").reset_index(drop=True)
    pairs_df.to_csv("pairs_to_annotate.tsv", sep="\t", index=False, header=False)
    print(f"✅  Guardado {len(pairs_df):,} pares en pairs_to_annotate.tsv")


if __name__ == "__main__":
    main()
