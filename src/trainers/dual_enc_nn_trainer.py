from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


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

# Apply the path tweak before any project imports
configure_python_path()

from src.utils.retrieval_utils import get_legal_dataset, get_legal_queries
from config.config import STORAGE_DIR


class DualEncNNTrainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset):
        self.model = model

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        base_dir = os.path.join(STORAGE_DIR, "legal_ir", "data")

        self.qids, self.queries = get_legal_queries(os.path.join(base_dir, "corpus", "queries_57.csv"))
        self.dids, self.docs = get_legal_dataset(os.path.join(base_dir, "corpus", "corpus_py.csv"))

    def preprocess_data(self):            
        # queries: list of query texts
        # docs: list of document texts

        vectorizer = TfidfVectorizer(max_features=10000)
        vectorizer.fit(queries + docs)  # or just docs if large enough

        X_query = vectorizer.transform(queries).toarray()  # shape: [num_examples, num_features]
        X_doc = vectorizer.transform(docs).toarray()       # shape: [num_examples, num_features]


    
    def train(self):
        # Implement training logic here
        pass

    def evaluate(self):
        # Implement evaluation logic here
        pass

