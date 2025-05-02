# --- Imports ---
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import sys
import pandas as pd
from collections import Counter
import xgboost as xgb
from scipy.sparse import hstack
from sklearn.metrics import classification_report, ndcg_score

# --- Path Configuration ---
def configure_python_path():
    """Add the project root directory to sys.path."""
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

configure_python_path()

# Assume these imports work
from src.utils.retrieval_utils import get_legal_dataset, get_legal_queries
from config.config import STORAGE_DIR


def ndcg_at_k(preds, labels, group, k=10):
    """
    Mean NDCG@k for either binary or graded labels.
    Parameters
    ----------
    preds  : 1-D array of model scores
    labels : 1-D array of int labels (0/1 or 0-3)
    group  : list[int]  – docs-per-query, sum == len(preds)
    k      : int        – rank cut-off
    """
    scores, labs, i = [], [], 0
    for g in group:
        scores.append(preds[i:i+g].reshape(1, -1))
        labs.append(labels[i:i+g].reshape(1, -1))
        i += g
    return float(np.fromiter(
        (ndcg_score(l, s, k=k) for s, l in zip(scores, labs)),
        dtype=float
    ).mean())


def to_binary(labels):
    """
    For graded labels (0-3) returns 1 if label >= 2.
    For binary (0/1) returns unchanged.
    """
    labels = np.asarray(labels)
    if labels.max() > 1:
        return (labels >= 2).astype(int)
    return labels


class XGBoostPairwiseTrainer:
    def __init__(self,
                 train_tsv: str,
                 dev_tsv: str,
                 test_tsv: str,
                 mode: str = 'classification'):
        """
        Trainer for XGBoost in two modes:
        - 'classification'  : binary:logistic with early stopping on logloss
        - 'ranking'         : rank:ndcg with query grouping

        Parameters
        ----------
        train_tsv, dev_tsv, test_tsv : str
            Paths to TSV splits with columns [qid, doc_id, label].
        mode : {'classification', 'ranking'}
        """
        assert mode in ('classification', 'ranking')
        self.mode = mode

        # 1) Load queries & documents
        base_dir = os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus")
        self.qids,  self.queries = get_legal_queries(
            os.path.join(base_dir, "inpars_mistral-small-2501_queries.tsv")
        )
        self.dids,  self.docs    = get_legal_dataset(
            os.path.join(base_dir, "corpus_py.csv")
        )

        # 2) Build ID→index maps for slicing sparse matrices
        self._qid2idx = {qid: idx for idx, qid in enumerate(self.qids)}
        self._did2idx = {did: idx for idx, did in enumerate(self.dids)}

        # 3) Load splits (shuffle only in classification mode)
        self.train_qids, self.train_dids, self.y_train = self._load_split(
            train_tsv, shuffle=(mode == 'classification')
        )
        self.dev_qids,   self.dev_dids,   self.y_dev   = self._load_split(
            dev_tsv,   shuffle=False
        )
        self.test_qids,  self.test_dids,  self.y_test  = self._load_split(
            test_tsv,  shuffle=False
        )

        # 4) If classification, compute scale_pos_weight
        if self.mode == 'classification':
            n_pos = int(np.sum(self.y_train))
            n_neg = len(self.y_train) - n_pos
            self.scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
            print(f"scale_pos_weight = {self.scale_pos_weight:.4f}")

        # 5) If ranking, build group lists
        if self.mode == 'ranking':
            self.group_train = self._make_group(self.train_qids)
            self.group_dev   = self._make_group(self.dev_qids)
            self.group_test  = self._make_group(self.test_qids)

        # 6) Fit TF-IDF on all text
        self.vectorizer = TfidfVectorizer(max_features=10000)
        self.vectorizer.fit(self.queries + self.docs)

        # 7) Prepare sparse feature matrices
        self._prepare_xgboost_data()

    def _load_split(self, path: str, shuffle: bool = True):
        """Reads TSV split and returns qids, doc_ids, and labels array."""
        df = pd.read_csv(path, sep='\t', dtype=str)

        df['label'] = df['label'].astype(int)
        if shuffle:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        labels = df['label'].to_numpy(dtype=np.int32)
        print(f"Loaded {len(labels)} examples from {path}")
        return df['qid'].tolist(), df['doc_id'].tolist(), labels

    def _make_group(self, qids):
        """
        Build group sizes for ranking: contiguous counts per qid.
        """
        groups, prev, count = [], None, 0
        for q in qids:
            if q == prev or (prev is None and count == 0):
                count += 1
            else:
                groups.append(count)
                count = 1
            prev = q
        groups.append(count)
        return groups

    def _prepare_xgboost_data(self):
        """Builds sparse X_train, X_dev, X_test via TF-IDF + hstack."""
        Q_mat = self.vectorizer.transform(self.queries)  # [n_q, F]
        D_mat = self.vectorizer.transform(self.docs)     # [n_d, F]

        def make_split(qids, dids):
            q_idx = [self._qid2idx[q] for q in qids]
            d_idx = [self._did2idx[d] for d in dids]
            Xq = Q_mat[q_idx]
            Xd = D_mat[d_idx]
            return hstack([Xq, Xd], format='csr')

        self.X_train = make_split(self.train_qids, self.train_dids)
        self.X_dev   = make_split(self.dev_qids,   self.dev_dids)
        self.X_test  = make_split(self.test_qids,  self.test_dids)

    def train(self,
              n_estimators=1000,
              learning_rate=0.1,
              max_depth=6,
              subsample=0.8,
              colsample_bytree=0.8,
              gamma=0,
              early_stopping_rounds=50,
              use_gpu=True):
        """
        Train either XGBClassifier (classification) or XGBRanker (ranking).
        """
        if self.mode == 'classification':
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                scale_pos_weight=self.scale_pos_weight,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                random_state=42,
                n_jobs=-1,
                tree_method='hist'
            )
            evals = [(self.X_train, 'train'), (self.X_dev, 'dev')]
            evals_result = {}
            model.fit(
                self.X_train, self.y_train,
                eval_set=evals,
                early_stopping_rounds=early_stopping_rounds,
                verbose=10,
                evals_result=evals_result
            )
            # Overfitting check
            tr = evals_result['train']['logloss']
            dv = evals_result['dev']['logloss']
            bi = model.best_iteration or len(tr)
            print(f"Best iter {bi} — train loss {tr[bi-1]:.4f}, dev loss {dv[bi-1]:.4f}")
        else:
            # Ranking mode
            model = xgb.XGBRanker(
                objective='rank:ndcg',
                eval_metric='ndcg@10',
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                random_state=42,
                early_stopping_rounds=early_stopping_rounds,
                n_jobs=-1,
                tree_method='hist'
            )
            model.fit(
                self.X_train, self.y_train,
                group=self.group_train,
                eval_set=[(self.X_dev, self.y_dev)],
                eval_group=[self.group_dev],
                verbose=True
            )
        self.model = model
        print("Training complete.\n")

    def evaluate(self):
        """Print classification report on dev and test."""
        if not hasattr(self, 'model'):
            raise RuntimeError("Model not trained yet.")
        
        if self.mode == 'ranking':
            pred_test = self.model.predict(self.X_test)
            mean_ndcg10 = ndcg_at_k(pred_test, self.y_test, self.group_test, k=10)
            print(f"Test NDCG@10: {mean_ndcg10:.4f}")
        elif self.mode == 'classification':
            for X, y, split in [
                (self.X_dev, self.y_dev, 'Dev'),
                (self.X_test, self.y_test, 'Test')
            ]:
                preds = self.model.predict(X)
                bin_preds = (preds >= 0.5).astype(int)
                print(f"=== {split} classification report ===")
                print(classification_report(y, bin_preds, digits=4, zero_division=0))
        else:
            raise ValueError("Invalid mode. Use 'classification' or 'ranking'.")
        print("Evaluation complete.\n")        

    def load_external_split(self, query_tsv, qrel_tsv):
        if query_tsv.endswith('.csv'):
            q_df = pd.read_csv(query_tsv, sep=',', header=0, names=['qid', 'query'], dtype=str)
        else:
            q_df = pd.read_csv(query_tsv, sep='\t', names=['qid', 'query'], dtype=str)
        qid_to_row = {q: i for i, q in enumerate(q_df['qid'])}
        Q_mat = self.vectorizer.transform(q_df['query'])

        r_df = pd.read_csv(qrel_tsv, sep='\t',
                        names=['qid', 'run', 'doc_id', 'label'], dtype=str)
        # drop run column
        r_df = r_df.drop(columns=['run'])
        r_df['label'] = r_df['label'].astype(int)
        r_df['qrow']  = r_df['qid'].map(qid_to_row)
        r_df['drow']  = r_df['doc_id'].map(self._did2idx)

        Xq = Q_mat[r_df['qrow']]
        Xd = self.vectorizer.transform([self.docs[i] for i in r_df['drow']])
        X_ext   = hstack([Xq, Xd], format='csr')
        y_ext   = r_df['label'].to_numpy(np.int32)
        group_e = r_df.groupby('qid').size().tolist()
        return X_ext, y_ext, group_e

    def evaluate_all(self,
                 ext_query_tsv=None,
                 ext_qrel_tsv=None,
                 k=10):
        """
        Evaluate on dev, test, and (optionally) an external graded set.
        """
        if not hasattr(self, 'model'):
            raise RuntimeError("Train first!")

        def one_split(X, y, group, name):
            scores = self.model.predict(X)
            ndcg = ndcg_at_k(scores, y, group, k)
            bin_y = to_binary(y)
            bin_pred = (scores >= 0.5).astype(int)
            print(f"\n=== {name}  (NDCG@{k}: {ndcg:.4f}) ===")
            print(classification_report(bin_y, bin_pred, digits=4, zero_division=0))

        # dev + test (their groups are already built)
        one_split(self.X_dev,  self.y_dev,  self.group_dev,  "DEV")
        one_split(self.X_test, self.y_test, self.group_test, "TEST")

        # external set if given
        if ext_query_tsv and ext_qrel_tsv:
            X_ext, y_ext, g_ext = self.load_external_split(
                ext_query_tsv, ext_qrel_tsv
            )
            one_split(X_ext, y_ext, g_ext, "EXTERNAL")

    def evaluate_external(self,
                      query_tsv: str,
                      qrel_tsv: str):
        """
        Evaluate on a completely new set of queries/qrels.
        """
        X_ext, y_ext, group_ext = self.load_external_split(query_tsv, qrel_tsv)

        # ---- predictions ----
        preds = self.model.predict(X_ext)

        # ---- metrics ----
        if self.mode == 'classification':
            bin_pred = (preds >= 0.5).astype(int)
            print(classification_report(y_ext, bin_pred, digits=4, zero_division=0))
        else:
            # ranking – compute NDCG@10 per query
            # break into query blocks
            scores, labels = [], []
            idx = 0
            for g in group_ext:
                scores.append(preds[idx: idx + g])
                labels.append(y_ext[idx: idx + g])
                idx += g
            ndcgs = [
                ndcg_score(l.reshape(1, -1), s.reshape(1, -1), k=10)
                for s, l in zip(scores, labels)
            ]
            print(f"Mean NDCG@10 on external set: {np.mean(ndcgs):.4f}")

    def run(self, **train_kwargs):
        """Wrapper: train then evaluate and save."""
        self.train(**train_kwargs)
        self.evaluate()
        fname = f"xgb_{self.mode}.json"
        self.model.save_model(fname)
        print(f"Model saved to {fname}")


# --- Main Execution Block ---
if __name__ == "__main__":
    base = os.path.join(STORAGE_DIR, "legal_ir", "data")
    train_tsv = os.path.join(base, "datasets", "cross_encoder", "bce_6x_inpars_train.tsv")
    dev_tsv   = os.path.join(base, "datasets", "cross_encoder", "bce_6x_inpars_dev.tsv")
    test_tsv  = os.path.join(base, "datasets", "cross_encoder", "bce_6x_inpars_test.tsv")

    # # Classification example
    # clf_trainer = XGBoostPairwiseTrainer(
    #     train_tsv, dev_tsv, test_tsv, mode='classification'
    # )
    # clf_trainer.run(n_estimators=500, learning_rate=0.1)

    # Ranking example
    rank_trainer = XGBoostPairwiseTrainer(
        train_tsv, dev_tsv, test_tsv, mode='ranking'
    )
    rank_trainer.run(n_estimators=500, learning_rate=0.05)

    # rank_trainer.evaluate_external(
    #     query_tsv=os.path.join(base, "corpus", "queries_57.csv"),
    #     qrel_tsv =os.path.join(base, "annotations", "qrels_py.tsv")
    # )

    rank_trainer.evaluate_all(
        ext_query_tsv=os.path.join(base, "corpus", "queries_57.csv"),
        ext_qrel_tsv =os.path.join(base, "annotations", "qrels_py.tsv"),
        k=10
    )

