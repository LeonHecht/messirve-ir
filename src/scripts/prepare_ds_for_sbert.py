"""
This script adapts the messirve_{split}_{country}_hard_negatives dataset
for training a Sentence Transformer model.
Output: messirve_{split}_{country}_hard_negatives1_sbert
Required input: messirve_{split}_{country}_hard_negatives

This is Script 3) of the pipeline
1) mine_hard_negatives.py
2) preprocess_dataset.py
3) prepare_ds_for_sbert.py (if later training with Sentence Transformers)
"""
from datasets import load_from_disk, load_dataset
import os

STORAGE_DIR = os.getenv("STORAGE_DIR", "/media/discoexterno/leon/messirve-ir/data")
# STORAGE_DIR = os.getenv("STORAGE_DIR", "/tmpu/helga_g/leonh_a/messirve-ir/data")


# Expand the 'hard_negatives' column
def expand_hard_negatives(example):
    # Ensure a fixed length of 5
    num_negs = 1
    # assert len(example['hard_negatives']) == num_negs, f"Hard negatives have to be exactly {num_negs}"
    negatives = example['hard_negatives']
    
    if num_negs > 1:
        # Create new keys for each negative
        return {f"negative_{i+1}": negatives[i] for i in range(num_negs)}
    else:
        return {"negative": negatives[0]}

def prepare(split):
    assert split in ["train", "test"], "Split should be either 'train' or 'test'"
    
    country = "ar"
    ds = load_from_disk(os.path.join(STORAGE_DIR, f"messirve_{split}_{country}_hard_negatives"))

    # Remove unwanted columns
    ds = ds.map(lambda x: x, remove_columns=['id',
                                             'docid',
                                             'query_date',
                                             'answer_date',
                                             'match_score',
                                             'expanded_search',
                                             'answer_type'])

    # Apply the transformation
    ds = ds.map(expand_hard_negatives)

    # Remove the original 'hard_negatives' column
    ds = ds.map(lambda x: x, remove_columns=["hard_negatives"])

    # Rename query column to 'anchor'
    ds = ds.rename_column("query", "anchor")

    # Rename docid_text column to 'positive'
    ds = ds.rename_column("docid_text", "positive")

    # df = ds.to_pandas()   # for visualizing

    # Save the dataset
    ds.save_to_disk(os.path.join(STORAGE_DIR, f"messirve_{split}_{country}_hard_negatives1_sbert"))


if __name__ == "__main__":
    prepare("test")