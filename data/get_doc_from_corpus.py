import sys
sys.path.append("home/leon/tesis/messirve-ir")
from config.config import STORAGE_DIR
from pathlib import Path
import pandas as pd
from src.utils.retrieval_utils import get_legal_dataset


def get_doc_from_corpus(docid, corpus_path):
    """
    Get the document text from the corpus given its docid.
    """
    # Load the corpus
    if corpus_path.suffix == ".csv":
        corpus = pd.read_csv(corpus_path)

        # conver Codigo column to string
        corpus["Codigo"] = corpus["Codigo"].astype(str)
        
        # Find the document with the given docid
        doc = corpus[corpus["Codigo"] == str(docid)]
        doc_text = doc.iloc[0]["text"]
    elif corpus_path.suffix == ".jsonl":
        dids, docs = get_legal_dataset(str(corpus_path))
        # Find the document with the given docid
        doc_dict = dict(zip(dids, docs))
        doc_text = doc_dict.get(docid, None)
    else:
        raise ValueError("Unsupported file format. Only .csv and .jsonl are supported.")
    
    return doc_text


def main():
    # Define the path to the corpus
    corpus_path = Path(STORAGE_DIR) / "legal_ir" / "data" / "corpus" / "corpus_raw_google_ocr.csv"
    # corpus_path = Path(STORAGE_DIR) / "legal_ir" / "data" / "corpus" / "corpus_Gpt4o-mini_cleaned.jsonl"
    # corpus_path = Path(STORAGE_DIR) / "legal_ir" / "data" / "corpus" / "corpus_chunk_headers.jsonl"
    # corpus_path = Path(STORAGE_DIR) / "legal_ir" / "data" / "corpus" / "corpus_mistral_summaries.jsonl"
    
    # Check if the corpus file exists
    if not corpus_path.exists():
        print(f"Corpus file not found: {corpus_path}")
        return
    
    # Example docid to search for
    docid = "96748"
    
    # Get the document text
    doc_text = get_doc_from_corpus(docid, corpus_path)
    
    if doc_text:
        print(f"Document text for docid {docid}:\n{doc_text}")
    else:
        print(f"No document found for docid {docid}")

if __name__ == "__main__":
    main()