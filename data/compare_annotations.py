import sys
sys.path.append("home/leon/tesis/messirve-ir")
from collections import Counter
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    cohen_kappa_score
)
from config.config import STORAGE_DIR
from pathlib import Path
import krippendorff


def compute_krippendorff_alpha(y_true, y_pred):
    """
    Compute Krippendorff's alpha for nominal data between two label sequences.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth labels encoded as integers.
    y_pred : array-like, shape (n_samples,)
        Predicted labels encoded as integers.

    Returns
    -------
    float
        Krippendorff's alpha coefficient.
    """
    # Prepare data for krippendorff: list of rater sequences
    data = [
        list(y_true) if not hasattr(y_true, 'tolist') else y_true.tolist(),
        list(y_pred) if not hasattr(y_pred, 'tolist') else y_pred.tolist()
    ]
    # Compute alpha (nominal)
    try:
        return krippendorff.alpha(reliability_data=data, level_of_measurement='nominal')
    except ValueError:
        # happens when there's no variability in y_true/y_pred
        return float('nan')


def load_qrels(path: Path) -> pd.DataFrame:
    """
    Load a qrels file (gold or pred) into a standard DataFrame.

    This function will accept either:
      • Gold files with columns: qid, run, doc_id, label  
      • Pred files with columns: qid, run, doc_id, label, evidence

    It ignores all columns except qid, doc_id (renamed to 'docid'), and label.

    Parameters
    ----------
    path : Path
        Path to the TSV qrels file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['qid', 'docid', 'label'], where
        - qid and docid are strings
        - label is an integer
    """
    # read *all* columns
    df = pd.read_csv(path, sep="\t", header=None)
    ncols = df.shape[1]

    if ncols == 3:
        df.columns = ["qid", "docid", "label"]
    elif ncols == 4:
        df.columns = ["qid", "run", "docid", "label"]
    elif ncols >= 5:
        df.columns = ["qid", "run", "docid", "label", "evidence"] + [
            f"extra_{i}" for i in range(5, ncols)
        ]
    else:
        raise ValueError(f"Expected ≥3 columns in qrels, got {ncols}")

    # normalize types
    df = df.astype({"qid": "string", "docid": "string", "label": "int32"})
    return df[["qid", "docid", "label"]]


def evaluate_qrels(
    gold_path: Path,
    pred_path: Path,
    *,
    graded_pred: bool = True
) -> None:
    """
    Evaluate a single run against qrels, printing both global and per-query
    metrics (F1 and Cohen's kappa), plus the mean per-query values and
    the full classification report.

    Parameters
    ----------
    gold_path : Path
        Path to the gold qrels file.
    pred_path : Path
        Path to the prediction qrels file.
    graded_pred : bool, optional
        If True, binarize predictions by (label >= 2),
        otherwise assume labels are already 0/1.
    """
    # load and binarize
    gold = load_qrels(gold_path)
    gold["label"] = (gold["label"] >= 2).astype(int)

    pred = load_qrels(pred_path)
    if graded_pred:
        pred["label"] = (pred["label"] >= 2).astype(int)

    # join on qid/docid
    merged = gold.merge(
        pred,
        on=["qid", "docid"],
        how="inner",
        suffixes=("_gold", "_pred"),
    )
    if merged.empty:
        raise ValueError("No overlapping (qid,docid) pairs between gold and pred.")

    y_true = merged["label_gold"]
    y_pred = merged["label_pred"]

    # overall metrics
    print("=== Overall metrics ===")
    print(f"Accuracy      : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision     : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall        : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Micro‑F1      : {f1_score(y_true, y_pred):.4f}")
    print(f"Cohen's kappa : {cohen_kappa_score(y_true, y_pred):.4f}")
    print(f"Krippendorff's alpha : {compute_krippendorff_alpha(y_true, y_pred):.4f}")

    # classification report
    print("\n=== Classification report ===")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["non‑relevant(0)", "relevant(1)"],
            zero_division=0,
        )
    )

    per_q_f1 = (
        merged
        .groupby("qid", group_keys=False)[["label_gold", "label_pred"]]
        .apply(lambda g: f1_score(
            g["label_gold"], g["label_pred"],
            average='macro', labels=[0, 1],
            zero_division=0
        ))
    )

    per_q_kappa = (
        merged
        .groupby("qid", group_keys=False)[["label_gold", "label_pred"]]
        .apply(lambda g: cohen_kappa_score(
            g["label_gold"], g["label_pred"], labels=[0, 1]
        ))
    )

    per_q_alpha = (
        merged
        .groupby('qid', group_keys=False)[['label_gold', 'label_pred']]
        .apply(lambda g: compute_krippendorff_alpha(
            g['label_gold'], g['label_pred']
        ))
    )

    print("=== Per‑query metrics ===")
    for qid in per_q_f1.index:
        print(
            f"Query {qid}: "
            f"F1 = {per_q_f1[qid]:.3f}, "
            f"Kappa = {per_q_kappa[qid]:.3f}"
            f"Alpha = {per_q_alpha[qid]:.3f}"
        )

    # mean per-query
    print("\n=== Mean per‑query ===")
    print(f"Mean F1    : {per_q_f1.mean():.4f}")
    print(f"Mean Kappa : {per_q_kappa.mean():.4f}")
    print(f"Mean Alpha : {per_q_alpha.mean():.4f}")

    n_zero_f1 = (per_q_f1 == 0).sum()
    print(f"[INFO] {n_zero_f1} queries with F1=0")

    n_low_kappa = (per_q_kappa < 0.2).sum()
    print(f"[INFO] {n_low_kappa} queries with Kappa < 0.2")

    missing_gold = pred.merge(gold, on=["qid", "docid"], how="left", indicator=True)
    missing_pred_only = missing_gold[missing_gold["_merge"] == "left_only"]
    print(f"[INFO] {len(missing_pred_only)} predicted pairs have no gold annotation")


def evaluate_qrels_new(
    gold_path: Path,
    pred_path: Path,
    *,
    binarize: bool = True,
    threshold: int = 2
) -> None:
    """
    Evaluate a single run against qrels, printing overall and per-query metrics.

    Supports either binary or multi-class (graded) labels. By default, assumes
    graded qrels and binarizes both gold and pred labels with `label >= threshold`.

    Parameters
    ----------
    gold_path : Path
        Path to the gold qrels file.
    pred_path : Path
        Path to the prediction qrels file.
    binarize : bool, optional
        If True, convert graded labels into binary relevance using
        ``label >= threshold``.  If False, keeps the original grades and
        does multi-class evaluation, computing macro-F1, multi-class kappa, etc.
        Default is True.
    threshold : int, optional
        Cutoff for binarization.  Labels >= this will be treated as
        “relevant” (1), others as “non-relevant” (0).  Only applies if
        ``binarize=True``.  Default is 2.
    """
    # -- Load --
    gold = load_qrels(gold_path)
    pred = load_qrels(pred_path)

    # -- Binarize if requested --
    if binarize:
        gold["label"] = (gold["label"] >= threshold).astype(int)
        pred["label"] = (pred["label"] >= threshold).astype(int)
        labels = [0, 1]
        target_names = ["non-relevant(0)", "relevant(1)"]
    else:
        # multi-class: collect all observed grades
        labels = sorted(
            set(gold["label"].unique()).union(pred["label"].unique())
        )
        target_names = [f"grade_{lvl}" for lvl in labels]

    # -- Join --
    merged = gold.merge(
        pred,
        on=["qid", "docid"],
        how="inner",
        suffixes=("_gold", "_pred"),
    )
    if merged.empty:
        raise ValueError(
            "No overlapping (qid,docid) pairs between gold and pred."
        )

    y_true = merged["label_gold"]
    y_pred = merged["label_pred"]

    # -- Overall metrics --
    print("=== Overall metrics ===")
    print(f"Accuracy      : {accuracy_score(y_true, y_pred):.4f}")

    if binarize:
        print(
            f"Precision     : "
            f"{precision_score(y_true, y_pred, zero_division=0):.4f}"
        )
        print(
            f"Recall        : "
            f"{recall_score(y_true, y_pred, zero_division=0):.4f}"
        )
        print(f"Micro-F1      : {f1_score(y_true, y_pred):.4f}")
    else:
        print(
            f"Macro-Precision: "
            f"{precision_score(y_true, y_pred, average='macro', labels=labels):.4f}"
        )
        print(
            f"Macro-Recall   : "
            f"{recall_score(y_true, y_pred, average='macro', labels=labels):.4f}"
        )
        print(
            f"Macro-F1       : "
            f"{f1_score(y_true, y_pred, average='macro', labels=labels):.4f}"
        )

    print(f"Cohen's kappa : {cohen_kappa_score(y_true, y_pred, labels=labels):.4f}")
    print(
        f"Krippendorff's alpha : "
        f"{compute_krippendorff_alpha(y_true, y_pred):.4f}"
    )

    # -- Classification report --
    print("\n=== Classification report ===")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
        )
    )

    # -- Per-query metrics --
    def _per_query_fn(fn, **kwargs):
        return (
            merged
            .groupby("qid", group_keys=False)[["label_gold", "label_pred"]]
            .apply(lambda g: fn(
                g["label_gold"], g["label_pred"], **kwargs
            ))
        )

    per_q_f1 = _per_query_fn(
        f1_score,
        average="macro",
        labels=labels,
        zero_division=0
    )
    per_q_kappa = _per_query_fn(
        cohen_kappa_score,
        labels=labels
    )
    per_q_alpha = _per_query_fn(
        lambda t, p: compute_krippendorff_alpha(t, p)
    )

    print("=== Per-query metrics ===")
    for qid in per_q_f1.index:
        print(
            f"Query {qid}: "
            f"F1 = {per_q_f1[qid]:.3f}, "
            f"Kappa = {per_q_kappa[qid]:.3f}, "
            f"Alpha = {per_q_alpha[qid]:.3f}"
        )

    # -- Mean per-query --
    print("\n=== Mean per-query ===")
    print(f"Mean F1    : {per_q_f1.mean():.4f}")
    print(f"Mean Kappa : {per_q_kappa.mean():.4f}")
    print(f"Mean Alpha : {per_q_alpha.mean():.4f}")

    # -- Info --
    if binarize:
        n_zero_f1 = (per_q_f1 == 0).sum()
        n_low_kappa = (per_q_kappa < 0.2).sum()
        missing = pred.merge(
            gold, on=["qid", "docid"], how="left", indicator=True
        )
        missing_pred_only = missing[missing["_merge"] == "left_only"]
        print(f"[INFO] {n_zero_f1} queries with F1=0")
        print(f"[INFO] {n_low_kappa} queries with Kappa < 0.2")
        print(
            f"[INFO] {len(missing_pred_only)} predicted pairs "
            "have no gold annotation"
        )


def remap_qrels(
    qrel_path: Path,
    mapping: dict[int, int],
    out_path: Path | None = None
) -> pd.DataFrame:
    """
    Remap the relevance labels in a QREL file according to a provided mapping.

    Parameters
    ----------
    qrel_path : Path
        Path to the input QREL file. Must be readable by `load_qrels`, which
        should return a DataFrame with columns ['qid', 'docid', 'label'].
    mapping : dict[int, int]
        A dict mapping original labels to new labels. For example, `{0: 0, 1: 0,
        2: 1, 3: 1}` to collapse a 4-point scale into binary.
    out_path : Path | None, optional
        If given, write the remapped QREL to this path in TSV format with
        no header (columns qid, docid, label). Defaults to None.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the same columns as the input, but with `label`
        replaced according to `mapping`.

    Raises
    ------
    ValueError
        If any original label in the file is missing from `mapping`.
    """
    # Load the QRELs
    df = load_qrels(qrel_path)

    # Check for unmapped labels
    orig_labels = set(df["label"].unique())
    missing_keys = orig_labels - set(mapping.keys())
    if missing_keys:
        raise ValueError(
            f"No mapping provided for original labels: {sorted(missing_keys)}"
        )

    # Apply the mapping
    df["label"] = df["label"].map(mapping)

    # Optionally write out
    if out_path is not None:
        df.to_csv(
            out_path,
            sep="\t",
            header=False,
            index=False,
            columns=["qid", "docid", "label"],
        )

    return df


def majority_vote(row):
    """
    Given a row with multiple 'pred_i' columns, return the majority label (ties→0).
    """
    votes = [row[c] for c in row.index if c.startswith("pred_")]
    cnt = Counter(votes)
    top, count = cnt.most_common(1)[0]
    # if there is a tie for top count, default to 0
    if list(cnt.values()).count(count) > 1:
        return 0
    return top


def evaluate_ensemble(gold_path, pred_paths):
    """
    Ensemble via majority voting over several prediction qrel files.
    """
    gold = load_qrels(gold_path)
    gold["label"] = (gold["label"] >= 2).astype(int)

    # load and binarize each prediction file
    dfs = []
    for i, p in enumerate(pred_paths):
        df = load_qrels(p)
        df[f"pred_{i}"] = df["label"]
        dfs.append(df[["qid","docid",f"pred_{i}"]])

    # merge gold + all preds on (qid,docid)
    df = gold[["qid","docid","label"]].rename(columns={"label":"gold"})
    for dfp in dfs:
        df = df.merge(dfp, on=["qid","docid"], how="inner")

    if df.empty:
        print("No overlapping (qid,docid) pairs for ensemble!")
        return

    # warn about any gold pairs dropped
    missing = gold.merge(df, on=["qid","docid"], how="left", indicator=True)
    missing = missing[missing["_merge"]=="left_only"]
    if not missing.empty:
        print(f"[WARN] {len(missing)} gold pairs missing from ensemble merge")

    # apply majority voting
    df["ensemble"] = df.apply(majority_vote, axis=1)

    y_true = df["gold"]
    y_pred = df["ensemble"]

    # reuse same printing logic as evaluate_qrels:
    print("=== Ensemble Overall metrics ===")
    print(f"Accuracy      : {accuracy_score(y_true,y_pred):.4f}")
    print(f"Precision     : {precision_score(y_true,y_pred,zero_division=0):.4f}")
    print(f"Recall        : {recall_score(y_true,y_pred,zero_division=0):.4f}")
    print(f"F1 score      : {f1_score(y_true,y_pred,zero_division=0):.4f}")
    print(f"Cohen's kappa : {cohen_kappa_score(y_true,y_pred):.4f}")
    print(f"Krippendorff's alpha : {compute_krippendorff_alpha(y_true, y_pred):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true,y_pred))
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred,
        labels=[0,1],
        target_names=["non‐relevant(0)","relevant(1)"],
        zero_division=0
    ))

    print("\n=== Ensemble Per‐query F1 ===")
    per_q_f1 = (
        df.groupby("qid", group_keys=False)[["gold", "ensemble"]]
        .apply(lambda g: f1_score(
            g["gold"], g["ensemble"], average='macro',
            labels=[0, 1], zero_division=0
        ))
    )

    per_q_kappa = (
        df.groupby("qid", group_keys=False)[["gold", "ensemble"]]
        .apply(lambda g: cohen_kappa_score(
            g["gold"], g["ensemble"], labels=[0, 1]
        ))
    )

    per_q_alpha = (
        df.groupby('qid', group_keys=False)[['gold', 'ensemble']]
        .apply(lambda g: compute_krippendorff_alpha(
            g['gold'], g['ensemble']
        ))
    )

    print("\n=== Ensemble Per‑query metrics ===")
    for qid in per_q_f1.index:
        print(
            f"Query {qid}: F1 = {per_q_f1[qid]:.3f}, "
            f"Kappa = {per_q_kappa[qid]:.3f}, "
            f"Alpha = {per_q_alpha[qid]:.3f}"
        )
    
    n_zero_f1 = (per_q_f1 == 0).sum()
    print(f"[INFO] {n_zero_f1} queries with F1=0")

    n_low_kappa = (per_q_kappa < 0.2).sum()
    print(f"[INFO] {n_low_kappa} queries with Kappa < 0.2")
    
    print(f"[INFO] {len(df)} queries in ensemble")

    print("\n=== Mean per‑query ===")
    print(f"Mean F1    : {per_q_f1.mean():.4f}")
    print(f"Mean Kappa : {per_q_kappa.mean():.4f}")
    print(f"Mean Alpha : {per_q_alpha.mean():.4f}")


def align_qrels(
    qrel_a: Path,
    qrel_b: Path,
    *,
    out_path: Path | None = None
) -> pd.DataFrame:
    """
    Align two QREL files (e.g. from two annotators) so that they
    line up by (qid, docid), showing each annotator’s label side by side.

    Parameters
    ----------
    qrel_a : Path
        Path to the first QREL file (annotator A).  Tab-delimited with
        columns: qid, iter, docid, label.
    qrel_b : Path
        Path to the second QREL file (annotator B).  Same format as `qrel_a`.
    out_path : Path | None, optional
        If provided, write the aligned table to this path as TSV (no header,
        index=False).  The columns will be:
        qid, docid, label_a, label_b.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - qid (int)
        - docid (int)
        - label_a (int)
        - label_b (int)
        sorted by qid, then docid.
    
    Raises
    ------
    ValueError
        If there are any (qid, docid) pairs present in one file but not the other.
    """
    # load both files
    df_a = pd.read_csv(
        qrel_a, sep="\t",
        names=["qid", "iter", "docid", "label"],
        dtype={"qid": int, "docid": int, "label": int}
    ).drop(columns="iter").rename(columns={"label": "label_a"})

    df_b = pd.read_csv(
        qrel_b, sep="\t",
        names=["qid", "iter", "docid", "label"],
        dtype={"qid": int, "docid": int, "label": int}
    ).drop(columns="iter").rename(columns={"label": "label_b"})

    # merge on qid + docid
    merged = df_a.merge(
        df_b,
        on=["qid", "docid"],
        how="outer",
        indicator=True
    )

    # check for mismatches
    only_a = merged[merged["_merge"] == "left_only"]
    only_b = merged[merged["_merge"] == "right_only"]
    if not only_a.empty or not only_b.empty:
        missing = []
        if not only_a.empty:
            missing.append(
                f"{len(only_a)} pairs only in {qrel_a.name}"
            )
        if not only_b.empty:
            missing.append(
                f"{len(only_b)} pairs only in {qrel_b.name}"
            )
        raise ValueError(
            "Mismatch in (qid,docid) keys: " + "; ".join(missing)
        )

    # keep only matching rows
    merged = merged[merged["_merge"] == "both"].drop(columns="_merge")

    # sort
    merged.sort_values(["qid", "docid"], inplace=True)

    # optionally write out
    if out_path is not None:
        merged.to_csv(
            out_path,
            sep="\t",
            header=False,
            index=False,
            columns=["qid", "docid", "label_a", "label_b"]
        )

    return merged


if __name__ == "__main__":
    # align_qrels(
    #     qrel_a=Path(STORAGE_DIR) / "legal_ir" / "data" / "annotations" / "meta_qrels.tsv",
    #     qrel_b=Path(STORAGE_DIR) / "legal_ir" / "data" / "annotations" / "filtered_qrels_54.tsv",
    #     out_path=Path(STORAGE_DIR) / "legal_ir" / "data" / "annotations" / "aligned_qrels.tsv"
    # )
    gold_path = Path(STORAGE_DIR) / "legal_ir" / "data" / "annotations" / "meta_qrels.tsv"

    # # Single model evaluation     
    # # gold_path = Path(STORAGE_DIR) / "legal_ir" / "data" / "annotations" / "qrels_mistral-small-2501_v7.tsv"
    # # pred_path = Path(STORAGE_DIR) / "legal_ir" / "data" / "annotations" / "qrels_mistral-small-2501_v7_QC.tsv"
    pred_path = Path(STORAGE_DIR) / "legal_ir" / "data" / "annotations" / "filtered_qrels_54.tsv"
    # # pred_path = Path(STORAGE_DIR) / "legal_ir" / "data" / "annotations" / "qrels_mistral-small-2501_v7.tsv"
    evaluate_qrels_new(gold_path, pred_path, binarize=False)

    # # Ensemble evaluation
    # ensemble_preds = [
    #     Path(STORAGE_DIR)/"legal_ir"/"data"/"annotations"/f"qrels_{m}.tsv"
    #     for m in [
    #     #   "mistral-small-2501_v1",
    #     #   "mistral-small-2501_v2",
    #     #   "mistral-small-2501_v3",
    #     #   "mistral-small-2501_v4",
    #         "mistral-small-2501_v5",
    #         "mistral-small-2501_v6",
    #         "mistral-small-2501_v6",
    #         "mistral-small-2501_v7",
    #         "mistral-small-2501_v7",
    #     #   "mistral-large-2411_v7",
    #     ]
    # ]

    # print("\n>> Ensemble (majority vote):\n")
    # evaluate_ensemble(gold_path, ensemble_preds)

    # remap_qrels(
    #     qrel_path=Path(STORAGE_DIR) / "legal_ir" / "data" / "annotations" / "filtered_qrels_54.tsv",
    #     mapping={0: 0, 1: 1, 2: 1, 3: 2},
    #     out_path=Path(STORAGE_DIR) / "legal_ir" / "data" / "annotations" / "filtered_qrels_54_3_graded.tsv"
    # )