from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
# make gpu 0 visible
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import torch
import torch.nn as nn
import torch.optim as optim
# Import f1_score specifically for validation reporting
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm


def configure_python_path():
    """
    Add the project root directory to sys.path.
    """
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Apply the path tweak before any project imports
configure_python_path()

# Assume these imports work after path configuration
from src.utils.retrieval_utils import get_legal_dataset, get_legal_queries
from config.config import STORAGE_DIR
# Assuming DualEncoderNN is defined here or imported correctly
# Make sure DualEncoderNN's forward method outputs LOGITS (no final sigmoid)
from src.models.dual_enc_nn import DualEncoderNN


class DualEncNNTrainer:
    def __init__(self,
                 train_tsv: str,
                 dev_tsv: str,
                 test_tsv: str):
        """
        Trainer for a Dual-Encoder MLP using pre-split TSVs.
        """
        # --- Load original queries and documents FIRST ---
        base_dir = os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus")
        self.qids, self.queries = get_legal_queries(
            os.path.join(base_dir, "inpars_mistral-small-2501_queries.tsv")
        )
        self.dids, self.docs = get_legal_dataset(
            os.path.join(base_dir, "corpus_py.csv")
        )
        print(f"Loaded {len(self.queries)} queries and {len(self.docs)} documents.")

        # --- Load splits ---
        print("Loading training data...")
        self.train_qids, self.train_dids, self.train_labels = (
            self._load_split(train_tsv)
        )
        print("Loading development data...")
        self.dev_qids, self.dev_dids, self.dev_labels = (
            self._load_split(dev_tsv)
        )
        print("Loading test data...")
        self.test_qids, self.test_dids, self.test_labels = (
            self._load_split(test_tsv)
        )

        # Calculate pos_weight
        n_samples = len(self.train_labels)
        n_class1 = np.sum(self.train_labels)
        n_class0 = n_samples - n_class1
        if n_class1 > 0 and n_class0 > 0:
            self.pos_weight_value = n_class0 / n_class1 * 0.16666
            print(f"Dataset balance: {n_class0} (0) / {n_class1} (1)")
            print(f"Calculated pos_weight for BCELoss: {self.pos_weight_value:.4f}")
        else:
            print("Warning: Training data contains only one class or is empty. Using default pos_weight=1.")
            self.pos_weight_value = 1.0
        self.pos_weight = torch.tensor([self.pos_weight_value], dtype=torch.float32)

        # --- Initialize and Fit Vectorizer BEFORE building feature matrices ---
        self.vectorizer = TfidfVectorizer(max_features=10000)
        print("Fitting TF-IDF on ALL queries and documents...")
        # Fit on the original full lists of queries and documents
        self.vectorizer.fit(self.queries + self.docs)
        print("TF-IDF fitting complete.")
        # --- End Vectorizer Fitting ---

        # --- Build feature matrices (THIS CREATES _qid2idx / _did2idx) ---
        print("Building feature matrices...")
        self._build_feature_matrices() # Now safe to call
        # --- End Build Features ---

        # Prepare dev tensors
        print("Preparing dev tensors...")
        self.Xq_dev_t = torch.from_numpy(self.Xq_dev)
        self.Xd_dev_t = torch.from_numpy(self.Xd_dev)
        self.y_dev_t = torch.from_numpy(self.dev_labels)

        print("Initialization complete.")


    def _load_split(self, path: str):
        """ Reads TSV, returns lists qids, dids, and np.array labels. """
        print(f"Loading split from: {path}")
        df = pd.read_csv(path, sep='\t', dtype={'qid': str, 'doc_id': str, 'label': int})
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        labels = df['label'].astype(np.float32).to_numpy()
        print(f"Loaded {len(df)} examples. Label distribution: {np.bincount(labels.astype(int))}")
        return df['qid'].tolist(), df['doc_id'].tolist(), labels

    def _build_feature_matrices(self):
        """ Builds TF-IDF features using the fitted vectorizer. """
        print("Transforming queries with fitted TF-IDF...")
        # Use the already loaded self.queries and self.docs lists
        # Ensure IDs used here match the ones loaded in __init__
        query_texts = self.queries # List of all query texts
        doc_texts = self.docs     # List of all document texts

        Q_mat = self.vectorizer.transform(query_texts).toarray().astype(np.float32)
        if not np.any(Q_mat):
            print("Warning: Q_mat (Query TF-IDF matrix) is all zeros after transform.")
        else:
            print("Q_mat TF-IDF transformation successful.")

        print("Transforming documents with fitted TF-IDF...")
        D_mat = self.vectorizer.transform(doc_texts).toarray().astype(np.float32)
        if not np.any(D_mat):
             print("Warning: D_mat (Document TF-IDF matrix) is all zeros after transform.")
        else:
             print("D_mat TF-IDF transformation successful.")


        # Rebuild maps from ID to row index based on original full lists
        self._qid2idx = {qid: i for i, qid in enumerate(self.qids)}
        self._did2idx = {did: i for i, did in enumerate(self.dids)}

        def stack_pairwise(qids, dids, mat_q, mat_d, qid2idx_map, did2idx_map):
            missing_qids = [q for q in qids if q not in qid2idx_map]
            missing_dids = [d for d in dids if d not in did2idx_map]
            if missing_qids: raise KeyError(f"Missing query IDs in vectorizer map: {missing_qids[:5]}...")
            if missing_dids: raise KeyError(f"Missing document IDs in vectorizer map: {missing_dids[:5]}...")
            q_idxs = [qid2idx_map[q] for q in qids]
            d_idxs = [did2idx_map[d] for d in dids]
            return mat_q[q_idxs], mat_d[d_idxs]

        print("Stacking train features...")
        self.Xq_train, self.Xd_train = stack_pairwise(self.train_qids, self.train_dids, Q_mat, D_mat, self._qid2idx, self._did2idx)
        print("Stacking dev features...")
        self.Xq_dev, self.Xd_dev = stack_pairwise(self.dev_qids, self.dev_dids, Q_mat, D_mat, self._qid2idx, self._did2idx)
        print("Stacking test features...")
        self.Xq_test, self.Xd_test = stack_pairwise(self.test_qids, self.test_dids, Q_mat, D_mat, self._qid2idx, self._did2idx)


    # --- New Method: _validate_epoch ---
    def _validate_epoch(self, device, batch_size=512):
        """
        Evaluates the model on the development set.

        Parameters
        ----------
        device : torch.device
            The device to run evaluation on.
        batch_size : int
            Batch size for evaluation.

        Returns
        -------
        tuple[float, float]
            Average validation loss and macro F1 score.
        """
        self.model.eval()  # Set model to evaluation mode
        total_val_loss = 0.0
        all_preds = []
        all_labels = []

        # Use the pre-converted tensors
        q_dev_t = self.Xq_dev_t
        d_dev_t = self.Xd_dev_t
        y_dev_t = self.y_dev_t # This is already a tensor

        with torch.no_grad():
            for i in range(0, q_dev_t.size(0), batch_size):
                batch_q = q_dev_t[i:i+batch_size].to(device)
                batch_d = d_dev_t[i:i+batch_size].to(device)
                batch_y = y_dev_t[i:i+batch_size].to(device) # Ground truth labels

                logits = self.model(batch_q, batch_d)
                # Calculate loss using the same criterion instance
                loss = self.criterion(logits.squeeze(), batch_y)
                total_val_loss += loss.item() * batch_q.size(0) # Accumulate weighted loss

                # Get predictions
                probs = torch.sigmoid(logits)
                preds = (probs.cpu().numpy() >= 0.5).astype(int)
                if preds.ndim > 1 and preds.shape[1] == 1:
                   preds = preds.squeeze(1)

                all_preds.extend(preds.tolist())
                all_labels.extend(batch_y.cpu().numpy().tolist()) # Store labels from batch

        avg_val_loss = total_val_loss / len(all_labels)
        # Calculate Macro F1-score (good for imbalance)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        self.model.train() # Set model back to training mode
        return avg_val_loss, val_f1
    # --- End New Method ---


    def train(self, epochs: int = 5, lr: float = 1e-3, batch_size: int = 256):
        """
        Train the Dual-Encoder MLP with validation after each epoch.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        self.model = DualEncoderNN(
            input_dim=self.Xq_train.shape[1],
            hidden_dim=128
        ).to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # --- Modified: Store criterion as instance variable ---
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(device))
        # --- End Modified ---

        q_train = torch.from_numpy(self.Xq_train)
        d_train = torch.from_numpy(self.Xd_train)
        y_train = torch.from_numpy(self.train_labels)

        train_dataset = torch.utils.data.TensorDataset(q_train, d_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        print(f"Starting training for {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            self.model.train() # Ensure model is in training mode
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Training", leave=False)
            for batch_q, batch_d, batch_y in pbar:
                batch_q = batch_q.to(device)
                batch_d = batch_d.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                logits = self.model(batch_q, batch_d)
                loss = self.criterion(logits.squeeze(), batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'train_loss': loss.item()})

            avg_epoch_loss = epoch_loss / num_batches

            # --- New: Validation Step ---
            val_loss, val_f1 = self._validate_epoch(device, batch_size=batch_size * 2) # Use larger batch for validation
            # --- End New ---

            # --- Modified Print Statement ---
            print(f'Epoch {epoch} — Train Loss: {avg_epoch_loss} | Val Loss: {val_loss} | Val Macro F1: {val_f1}')
            # --- End Modified ---

            # (Optional: Add logic here for early stopping or saving best model based on val_loss or val_f1)

    def evaluate(self):
        """
        Final evaluation on dev and test splits after training.
        Uses the _eval helper function.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval() # Ensure model is in eval mode for final evaluation

        # --- Define _eval helper inside or ensure it's accessible ---
        def _eval(q_mat, d_mat, labels, split_name):
            print(f"\n--- Final Evaluation on {split_name} split ---")
            q_t = torch.from_numpy(q_mat).to(device)
            d_t = torch.from_numpy(d_mat).to(device)

            all_preds = []
            batch_size = 512
            with torch.no_grad():
                for i in range(0, q_t.size(0), batch_size):
                    batch_q = q_t[i:i+batch_size]
                    batch_d = d_t[i:i+batch_size]
                    logits = self.model(batch_q, batch_d)
                    probs = torch.sigmoid(logits)
                    batch_preds = (probs.cpu().numpy() >= 0.5).astype(int)
                    if batch_preds.ndim > 1 and batch_preds.shape[1] == 1:
                       batch_preds = batch_preds.squeeze(1)
                    all_preds.extend(batch_preds.tolist())

            all_preds = np.array(all_preds)
            # Ensure labels are 1D numpy array
            if isinstance(labels, torch.Tensor):
                 labels = labels.cpu().numpy() # Convert if still tensor
            if labels.ndim > 1 and labels.shape[1] == 1:
                labels = labels.squeeze(1)

            if len(all_preds) != len(labels):
                 print(f"Warning: Mismatch length preds ({len(all_preds)}) vs labels ({len(labels)}) for {split_name}.")
                 min_len = min(len(all_preds), len(labels))
                 all_preds = all_preds[:min_len]
                 labels = labels[:min_len]

            print(classification_report(labels, all_preds, digits=4, zero_division=0))
        # --- End _eval definition ---

        # Use the original numpy arrays for final evaluation
        _eval(self.Xq_dev, self.Xd_dev, self.dev_labels, "Dev")
        _eval(self.Xq_test, self.Xd_test, self.test_labels, "Test")


    def run(self, epochs: int = 5, lr: float = 1e-3, batch_size: int = 256):
        """
        Run the training (with per-epoch validation) and final evaluation process.
        """
        self.train(epochs=epochs, lr=lr, batch_size=batch_size)
        self.evaluate() # Perform final evaluation on dev and test

        model_save_path = "dual_enc_nn_model_weighted.pth"
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model saved as {model_save_path}")


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

    trainer = DualEncNNTrainer(
        train_tsv=train_ds_path,
        dev_tsv=dev_ds_path,
        test_tsv=test_ds_path
    )
    # Consider adjusting hyperparameters based on validation performance
    trainer.run(epochs=15, lr=1e-4, batch_size=256)
    print("Training and evaluation complete.")