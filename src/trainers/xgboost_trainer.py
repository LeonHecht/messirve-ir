# --- Imports ---
# Keep necessary imports
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import sys
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
# Add xgboost import
import xgboost as xgb
import torch
from scipy.sparse import hstack

# Remove PyTorch imports if no longer needed elsewhere
# import torch
# import torch.nn as nn
# import torch.optim as optim

# --- Path Configuration (keep as is) ---
def configure_python_path():
    """ Add the project root directory to sys.path. """
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

configure_python_path()

# Assume these imports work
from src.utils.retrieval_utils import get_legal_dataset, get_legal_queries
from config.config import STORAGE_DIR
# Remove NN model import
# from src.models.dual_enc_nn import DualEncoderNN

# --- New XGBoost Trainer Class ---
class XGBoostPairwiseTrainer:
    def __init__(self, train_tsv: str, dev_tsv: str, test_tsv: str):
        """
        Trainer for an XGBoost model using sparse TF-IDF features.
        """
        # 1) Load queries & documents
        base_dir = os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus")
        self.qids, self.queries = get_legal_queries(
            os.path.join(base_dir, "inpars_mistral-small-2501_queries.tsv")
        )
        self.dids, self.docs = get_legal_dataset(
            os.path.join(base_dir, "corpus_py.csv")
        )

        # 2) Build ID→index maps for sparse slicing
        self._qid2idx = {qid: idx for idx, qid in enumerate(self.qids)}
        self._did2idx = {did: idx for idx, did in enumerate(self.dids)}

        # 3) Load train/dev/test splits
        self.train_qids, self.train_dids, self.y_train = self._load_split(train_tsv)
        self.dev_qids,   self.dev_dids,   self.y_dev   = self._load_split(dev_tsv)
        self.test_qids,  self.test_dids,  self.y_test  = self._load_split(test_tsv)

        n_pos = int(np.sum(self.y_train))
        n_neg = len(self.y_train) - n_pos
        if n_pos > 0:
            self.scale_pos_weight = n_neg / n_pos
        else:
            self.scale_pos_weight = 1.0
        print(f"Scale pos weight set to {self.scale_pos_weight:.4f}")

        # 4) Fit TF-IDF on all text
        self.vectorizer = TfidfVectorizer(max_features=10000)
        self.vectorizer.fit(self.queries + self.docs)

        # 5) Prepare sparse XGBoost data
        self._prepare_xgboost_data()

    def _prepare_xgboost_data(self):
        """
        Builds sparse TF-IDF features and combines them 
        into CSR matrices (X_train, X_dev, X_test).
        """
        # 1) Transform all queries/docs into sparse matrices
        Q_mat = self.vectorizer.transform(self.queries)  # CSR [n_q, F]
        D_mat = self.vectorizer.transform(self.docs)     # CSR [n_d, F]

        # 2) Helper to slice & hstack
        def make_sparse_split(qids, dids):
            q_idxs = [self._qid2idx[q] for q in qids]
            d_idxs = [self._did2idx[d] for d in dids]
            Xq = Q_mat[q_idxs]
            Xd = D_mat[d_idxs]
            # horizontal stack: [Xq | Xd]
            return hstack([Xq, Xd], format='csr')

        # 3) Build each split (sparse!)
        self.X_train = make_sparse_split(self.train_qids, self.train_dids)
        self.X_dev   = make_sparse_split(self.dev_qids,   self.dev_dids)
        self.X_test  = make_sparse_split(self.test_qids,  self.test_dids)

    def _load_split(self, path: str):
        """ Reads TSV, returns lists qids, dids, and np.array labels. """
        print(f"Loading split from: {path}")
        df = pd.read_csv(path, sep='\t', dtype={'qid': str, 'doc_id': str, 'label': int})
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        # Ensure labels are integer type for XGBoost if needed, float32 is usually fine too
        labels = df['label'].astype(np.int32).to_numpy()
        print(f"Loaded {len(df)} examples. Label distribution: {np.bincount(labels)}")
        return df['qid'].tolist(), df['doc_id'].tolist(), labels

    def train(self,
              # XGBoost specific hyperparameters
              n_estimators=1000, # Max boosting rounds
              learning_rate=0.1,
              max_depth=6,
              subsample=0.8,
              colsample_bytree=0.8,
              gamma=0,
              early_stopping_rounds=50, # Stop if eval metric doesn't improve for 50 rounds
              use_gpu=False # Set to True to attempt GPU usage
             ):
        """
        Trains the XGBoost Classifier model.
        """
        print("\n--- Starting XGBoost Training ---")
        print(f"Parameters: n_estimators={n_estimators}, lr={learning_rate}, max_depth={max_depth}, early_stopping={early_stopping_rounds}")
        print(f"Using scale_pos_weight: {self.scale_pos_weight:.4f}")

        # Configure GPU usage if requested
        tree_method_param = 'hist' # Default efficient CPU method
        device_param = 'cpu'
        if use_gpu:
             print("Attempting to use GPU...")
             # Check if XGBoost was compiled with GPU support and if a GPU is visible
             # Basic check, might need more robust check depending on environment
             try:
                 # A simple check: try instantiating with GPU params
                 _ = xgb.XGBClassifier(tree_method='hist', device='cuda')
                 print("GPU seems available for XGBoost.")
                 tree_method_param = 'hist'
                 device_param = 'cuda'
             except Exception as e:
                 print(f"Could not enable GPU for XGBoost ({e}), falling back to CPU.")

        # Instantiate the XGBoost model
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',       # Binary classification objective
            eval_metric='logloss',             # Evaluation metric for early stopping (can also use 'auc', 'error', etc.)
            scale_pos_weight=self.scale_pos_weight, # Handle class imbalance
            n_estimators=n_estimators,         # Max number of boosting rounds
            learning_rate=learning_rate,       # Step size shrinkage
            max_depth=max_depth,               # Max depth of a tree
            subsample=subsample,               # Fraction of samples used per tree
            colsample_bytree=colsample_bytree, # Fraction of features used per tree
            gamma=gamma,                       # Minimum loss reduction required for split
            random_state=42,                   # For reproducibility
            tree_method=tree_method_param,     # Use 'hist' for efficiency, potentially 'gpu_hist' if available
            device=device_param,               # Specify 'cuda' if using GPU
            early_stopping_rounds=early_stopping_rounds,
            n_jobs=-1                          # Use all available CPU cores for CPU training
        )

        # Collect eval metrics
        evals_result = {}
        self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_train, 'train'),
                    (self.X_dev,   'dev')],
            eval_metric='logloss',
            early_stopping_rounds=early_stopping_rounds,
            verbose=10,
            evals_result=evals_result
        )

        # Compare losses for overfitting signal
        train_losses = evals_result['train']['logloss']
        dev_losses   = evals_result['dev']['logloss']

        best_iter = self.model.best_iteration or len(train_losses)
        print(f"\nBest iteration: {best_iter}")
        print(f"Train loss at best iter: {train_losses[best_iter-1]:.4f}")
        print(f"Dev   loss at best iter: {dev_losses[best_iter-1]:.4f}")

        if dev_losses[best_iter-1] < train_losses[best_iter-1]:
            print("✅ No sign of overfitting (dev loss < train loss).")
        else:
            print("⚠️ Possible overfitting (dev loss ≥ train loss).")

        # (Optionally) return evals_result for plotting
        return evals_result


    def evaluate(self):
        """
        Evaluates the trained XGBoost model on the dev and test sets.
        """
        print("\n--- Evaluating XGBoost Model ---")
        if not hasattr(self, 'model'):
            print("Error: Model has not been trained yet. Call train() first.")
            return

        def _eval(X_data, y_data, split_name):
            print(f"\n--- Evaluation on {split_name} split ---")
            if X_data is None or y_data is None:
                 print(f"Data for {split_name} split not available.")
                 return

            # Get predictions (outputs 0 or 1 directly for binary:logistic)
            preds = self.model.predict(X_data)

            # Calculate and print metrics
            print(classification_report(y_data, preds, digits=4, zero_division=0))

        # Evaluate on Development set
        _eval(self.X_dev, self.y_dev, "Dev")

        # Evaluate on Test set
        _eval(self.X_test, self.y_test, "Test")

    def run(self, **train_kwargs):
        """
        Runs the training and evaluation process. Accepts XGBoost training args.
        """
        self.train(**train_kwargs) # Pass kwargs like n_estimators, learning_rate etc.
        self.evaluate()

        # Save the model (optional)
        model_save_path = "xgboost_pairwise_model.json" # XGBoost models often saved in json/ubj format
        self.model.save_model(model_save_path)
        print(f"XGBoost model saved as {model_save_path}")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Ensure STORAGE_DIR is defined or imported correctly
    # from config.config import STORAGE_DIR

    base_dir = os.path.join(STORAGE_DIR, "legal_ir", "data")
    train_ds_path = os.path.join(base_dir, "datasets", "cross_encoder", "bce_6x_inpars_train.tsv")
    dev_ds_path = os.path.join(base_dir, "datasets", "cross_encoder", "bce_6x_inpars_dev.tsv")
    test_ds_path = os.path.join(base_dir, "datasets", "cross_encoder", "bce_6x_inpars_test.tsv")

    for path in [train_ds_path, dev_ds_path, test_ds_path]:
         if not os.path.exists(path):
              print(f"ERROR: Data file not found at {path}")
              sys.exit(1)

    # Instantiate the XGBoost trainer
    trainer = XGBoostPairwiseTrainer(
        train_tsv=train_ds_path,
        dev_tsv=dev_ds_path,
        test_tsv=test_ds_path
    )

    # Run training and evaluation
    # Pass XGBoost hyperparameters here
    trainer.run(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=6,
        early_stopping_rounds=50,
        use_gpu=torch.cuda.is_available() # Example: try using GPU if PyTorch detects one
        # add other xgb params as needed: subsample, colsample_bytree etc.
    )

    print("\nXGBoost training and evaluation complete.")