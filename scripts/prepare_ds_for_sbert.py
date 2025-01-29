from datasets import load_from_disk, load_dataset


# Expand the 'hard_negatives' column
def expand_hard_negatives(example):
    # Ensure a fixed length of 5
    num_negs = 5
    assert len(example['hard_negatives']) == num_negs, f"Hard negatives have to be exactly {num_negs}"
    negatives = example['hard_negatives']
    
    # Create new keys for each negative
    return {f"negative_{i+1}": negatives[i] for i in range(num_negs)}


def prepare(split):
    assert split in ["train", "test"], "Split should be either 'train' or 'test'"
    
    country = "ar"
    ds = load_from_disk(f"messirve_{split}_{country}_hard_negatives")

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

    # Save the dataset
    ds.save_to_disk(f"messirve_{split}_{country}_hard_negatives_sbert")


if __name__ == "__main__":
    prepare("test")