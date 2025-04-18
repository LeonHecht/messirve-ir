import pandas as pd
import random
from config.config import STORAGE_DIR
import os


def freeze_query_split(
    qrels_path: str,
    train_frac: float = 0.7,
    dev_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42
):
    """
    Freeze a train/dev/test split of query IDs from a qrels file.

    Parameters
    ----------
    qrels_path : str
        Path to the TSV qrels file with columns [qid, run, doc_id, label].
    train_frac : float, optional
        Fraction of queries to assign to the training set. Default is 0.7.
    dev_frac : float, optional
        Fraction to assign to the development set. Default is 0.15.
    test_frac : float, optional
        Fraction to assign to the test set. Default is 0.15.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    dict
        A dict with keys 'train', 'dev', 'test' mapping to lists of qids.
    """
    # 1) Load qrels and get unique qids
    df = pd.read_csv(
        qrels_path,
        sep="\t",
        header=None,
        names=["qid", "run", "doc_id", "label"]
    )
    qids = sorted(df["qid"].unique().tolist())

    # 2) Shuffle with seed
    random.seed(seed)
    random.shuffle(qids)

    # 3) Compute split sizes
    n = len(qids)
    n_train = int(train_frac * n)
    n_dev = int(dev_frac * n)

    # 4) Slice
    train_qids = qids[:n_train]
    dev_qids = qids[n_train : n_train + n_dev]
    test_qids = qids[n_train + n_dev :]

    # 5) Save for later
    save_dir = os.path.join(STORAGE_DIR, "legal_ir", "data")
    pd.Series(train_qids).to_csv(os.path.join(save_dir, "qids_train.txt"), index=False, header=False)
    pd.Series(dev_qids).to_csv(os.path.join(save_dir, "qids_dev.txt"), index=False, header=False)
    pd.Series(test_qids).to_csv(os.path.join(save_dir, "qids_test.txt"), index=False, header=False)

    print(f"Total queries: {n}")
    print(f"→ Train: {len(train_qids)} qids saved to qids_train.txt")
    print(f"→ Dev:   {len(dev_qids)} qids saved to qids_dev.txt")
    print(f"→ Test:  {len(test_qids)} qids saved to qids_test.txt")

    return {"train": train_qids, "dev": dev_qids, "test": test_qids}


if __name__ == "__main__":
    splits = freeze_query_split(os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "qrels_py.tsv"))
