"""
This examples trains a CrossEncoder for the NLI task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it learns to predict the labels: "irrelevant": 0, "relevant": 1.

It does NOT produce a sentence embedding and does NOT work for individual sentences.

Usage:
python training_nli.py
"""

import csv
import gzip
import logging
import math
import os
from datetime import datetime
import pickle

import sys
print("Executable", sys.executable)

STORAGE_DIR = os.getenv("STORAGE_DIR")
print(f"STORAGE_DIR: {STORAGE_DIR}")    # STORAGE_DIR: /media/discoexterno/leon/messirve-ir

from tqdm import tqdm

from datasets import load_from_disk

from torch.utils.data import DataLoader

from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEF1Evaluator, CESoftmaxAccuracyEvaluator
from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.readers import InputExample


from sentence_transformers.cross_encoder.evaluation import CEF1Evaluator

class CustomCEF1Evaluator(CEF1Evaluator):
    """
    Custom evaluator that wraps CEF1Evaluator to return a single F1 score.
    """
    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        results = super().__call__(model, output_path=output_path, epoch=epoch, steps=steps)
        
        print("Results:", results)
        print("Type of results:", type(results))
        # Adjust the key based on the actual dictionary output.
        # Your log prints "Binary F1 score     : 0.00", so the key might be 'binary_f1'
        # return results.get("f1", results.get("binary_f1", 0))
        if type(results) == dict:
            return results.get("f1", 0)
        elif type(results) == float:
            return results
        

from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator

class CustomCESoftmaxAccuracyEvaluator(CESoftmaxAccuracyEvaluator):
    """
    Custom evaluator that wraps CESoftmaxAccuracyEvaluator to return a single accuracy score.
    """
    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        results = super().__call__(model, output_path=output_path, epoch=epoch, steps=steps)
        # Print for debugging (you can remove these prints once it's working)
        print("Custom Accuracy Evaluator Results:", results)
        print("Type of Accuracy Results:", type(results))
        if isinstance(results, dict):
            # Adjust the key name if needed; usually it's "accuracy"
            return results.get("accuracy", 0)
        elif isinstance(results, (float, int)):
            return results
        # Fallback: if something unexpected, return 0
        return 0


from sentence_transformers.evaluation import SequentialEvaluator

class CustomSequentialEvaluator(SequentialEvaluator):
    """
    Custom sequential evaluator that aggregates multiple evaluator scores into a single float.
    
    Parameters
    ----------
    evaluators : list
        List of evaluator instances that are callable and return a numeric score.
    """
    
    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        """
        Evaluate the model with each evaluator, aggregate the results, and return a single score.
        
        Parameters
        ----------
        model : CrossEncoder
            The model to evaluate.
        output_path : str, optional
            Directory where evaluation results can be saved.
        epoch : int, optional
            The current epoch.
        steps : int, optional
            The current step.
        
        Returns
        -------
        float
            The aggregated evaluation score.
        """
        scores = []
        for evaluator in self.evaluators:
            score = evaluator(model, output_path=output_path, epoch=epoch, steps=steps)
            try:
                # Force conversion to float if possible.
                score = float(score)
            except Exception:
                score = 0.0
            scores.append(score)
        # You can change the aggregation logic here.
        # For example, take the average of all evaluator scores:
        return sum(scores) / len(scores) if scores else 0.0


def convert_ds(ds, num_negatives):
    """
    Convert a HuggingFace Dataset into a list of InputExample objects.

    Each row is expected to have an "anchor", a "positive", and negative columns
    named "negative_1", ..., "negative_N" where N equals num_negatives.

    Parameters
    ----------
    ds : datasets.Dataset
        HuggingFace Dataset with columns "anchor", "positive", and negative columns.
    num_negatives : int
        Number of negative examples per sample.

    Returns
    -------
    list
        List of InputExample objects. Each row produces one positive sample and
        num_negatives negative samples.
    """
    # Convert to a dict-of-lists to avoid repeated expensive indexing.
    data = ds.to_dict()
    anchors = data["anchor"]
    positives = data["positive"]
    negatives_lists = [data[f"negative_{j}"] for j in range(1, num_negatives + 1)]

    samples = []
    # Zip through the columns
    for anchor, positive, *negatives in tqdm(zip(anchors, positives, *negatives_lists), total=len(anchors), desc="Converting ds"):
        samples.append(InputExample(texts=[anchor, positive], label=1))
        for neg in negatives:
            samples.append(InputExample(texts=[anchor, neg], label=0))
    return samples


def prepare_dataset(ds_path):
    train_ds_path = os.path.join(ds_path, "messirve_train_ar_hard_negatives_sbert")
    test_ds_path = os.path.join(ds_path, "messirve_test_ar_hard_negatives_sbert")

    train_ds = load_from_disk(train_ds_path)
    test_ds = load_from_disk(test_ds_path)

    num_negatives = 5

    train_samples = convert_ds(train_ds, num_negatives)
    test_samples = convert_ds(test_ds, num_negatives)

    return train_samples, test_samples
    

def main():
    #### Just some code to print debug information to stdout
    logging.basicConfig(
        format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
    )
    logger = logging.getLogger(__name__)
    #### /print debug information to stdout

    ds_path = os.path.join(STORAGE_DIR, "datasets")

    train_samples_path = os.path.join(STORAGE_DIR, "datasets", "train_samples_cross_enc.pkl")
    test_samples_path = os.path.join(STORAGE_DIR, "datasets", "test_samples_cross_enc.pkl")
    
    if not os.path.exists(train_samples_path):
        logger.info("Preparing datasets...")
        train_samples, test_samples = prepare_dataset(ds_path)
        logger.info("Done preparing datasets.")

        logger.info("Saving samples to disk...")
        # save samples to disk
        with open(train_samples_path, "wb") as f:
            pickle.dump(train_samples, f)
        
        with open(test_samples_path, "wb") as f:
            pickle.dump(test_samples, f)
        logger.info("Done saving samples to disk.")
    else:
        logger.info("Reading samples from disk...")
        # read samples from disk
        with open(train_samples_path, "rb") as f:
            train_samples = pickle.load(f)
        
        with open(test_samples_path, "rb") as f:
            test_samples = pickle.load(f)
        logger.info("Done reading samples from disk.")

    train_batch_size = 16
    num_epochs = 1
    model_save_path = os.path.join(STORAGE_DIR, "cross_enc_experiment_logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # Define our CrossEncoder model. We use distilroberta-base as basis and setup it up to predict 2 labels (0 and 1)
    model = CrossEncoder("distilbert/distilbert-base-multilingual-cased", num_labels=2)

    # We wrap train_samples, which is a list of InputExample, in a pytorch DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

    # During training, we use CESoftmaxAccuracyEvaluator and CEF1Evaluator to measure the performance on the dev set
    accuracy_evaluator = CustomCESoftmaxAccuracyEvaluator.from_input_examples(test_samples, name="messIRve-dev")
    f1_evaluator = CustomCEF1Evaluator.from_input_examples(test_samples, name="messIRve-dev")
    evaluator = CustomSequentialEvaluator([accuracy_evaluator, f1_evaluator])

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    logger.info(f"Warmup-steps: {warmup_steps}")

    # Train the model
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=100,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
    )


if __name__ == "__main__":
    main()