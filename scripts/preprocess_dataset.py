from datasets import load_from_disk
import pickle


def add_hard_negatives(example, docid_to_text, hard_negatives):
    # Example logic to add hard negatives:
    # You can retrieve hard negatives from some precomputed data or logic
    query_id = example["id"]
    hard_neg_docids = hard_negatives[str(query_id)]
    # Get hard negative texts using the precomputed dictionary
    hard_neg_doc_texts = [docid_to_text[docid] for docid in hard_neg_docids if docid in docid_to_text]
    
    # Return the updated example
    example["hard_negatives"] = hard_neg_doc_texts
    return example


def preprocess_dataset(train_ds, hard_negatives):
    # Precompute a mapping from docid to docid_text
    docid_to_text = {item["docid"]: item["docid_text"] for item in train_ds}

    # Map the function over the dataset
    train_ds = train_ds.map(
        lambda example: add_hard_negatives(example, docid_to_text, hard_negatives),
        batched=False  # If hard_negatives are added per example, keep this False
    )
    return train_ds


def limit_hard_negatives(example):
    # Limit hard negatives to 5 per query
    example["hard_negatives"] = example["hard_negatives"][:5]
    return example


def main(split):
    """ 
    Preprocess the dataset by adding hard negatives to the examples
    The output can be used in prepare_ds_for_sbert.py script.
    """
    assert split in ["train", "test"], "Split should be either 'train' or 'test'"
    
    country = "ar"
    print(f"Training on {country} dataset...")
    ds = load_from_disk(f"messirve_{country}")
    ds = ds[split]
    print("Dataset loaded")

    # read hard negatives from disk
    print("Loading hard negatives from pickle file...", end="")
    with open(f"hard_negatives_{split}_bge_{country}.pkl", "rb") as f:
        hard_negatives = pickle.load(f)
        print("Done")

    ds = preprocess_dataset(ds, hard_negatives)
    ds = ds.map(limit_hard_negatives)

    # save ds to disk
    ds.save_to_disk(f"messirve_{split}_{country}_hard_negatives")


if __name__ == "__main__":
    main("test")