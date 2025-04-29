"""
This script is used to retrieve documents for annotation using different embeddings models and BM25.
It retrieves the top_k most similar documents for each query and saves the query-top_k document pairs to a CSV file.
The embeddings models used are Jinja and BAAI embeddings models.
"""

import pandas as pd
import numpy as np
import os
# make only GPU1 visible
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
print(sys.executable)


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

# Globals
MAX_QUERY_LEN = 128
MAX_DOC_LEN = 4096


def get_docs(path):
    """ Get all the documents of the corpus and their doc_id (Codigo).
        Result is a dictionary with document_id as key and text as value.
        Example: {'492385': 'text1', '684933': 'text2', ...}

    Args:
        path (str): Path to the CSV file with the corpus.

    Returns:
        dict: Dictionary with document_id as key and text as value.
    """
    df = pd.read_csv(path)
    doc_ids = df['Codigo'].tolist()
    texts = df['text'].tolist()

    docs = {}
    for i in range(len(doc_ids)):
        docs[doc_ids[i]] = texts[i]

    return docs


def retrieve(query_ids, texts, doc_ids, embeddings_queries, embeddings_docs, top_k=10, sim_type='dot'):
    """
    Given a list of queries and documents, and their embeddings, compute the similarity between queries and documents
    and return the top_k most similar documents for each query.
    Dot-product and cosine similarity are supported.

    Args:
        queries (list): List of queries.
        texts (list): List of documents.
        doc_ids (list): List of document IDs (Codigos).
        embeddings_queries (np.ndarray): Embeddings of queries.
        embeddings_docs (np.ndarray): Embeddings of documents.
        top_k (int): Number of most similar documents to retrieve.
        sim_type (str): Type of similarity to use. Options: 'dot', 'cosine'.
    
    Returns:
        dict: Dictionary with query as key and a list of tuples of (similarity, document text, doc_id) as value.
    """
    import torch.nn.functional as F
    
    if sim_type == 'dot':
        similarity = embeddings_queries @ embeddings_docs.T
    elif sim_type == 'cosine':
        # Only use angle between embeddings
        embeddings_queries = F.normalize(embeddings_queries, p=2, dim=1)
        embeddings_docs = F.normalize(embeddings_docs, p=2, dim=1)
        similarity = (embeddings_queries @ embeddings_docs.T) * 100
    else:
        raise ValueError(f"Invalid similarity type: {sim_type}")
    # print("similarity", similarity)   # [[0.6265, 0.3477], [0.3499, 0.678 ]]
    
    similarity_dict = {}
    for i in range(len(query_ids)):
        # Sort similarity scores
        similarity_dict[query_ids[i]] = sorted(enumerate(similarity[i]), key=lambda x: x[1], reverse=True)
        # Cut similarity scores to top_k results
        similarity_dict[query_ids[i]] = similarity_dict[query_ids[i]][:top_k]
        # Fill similarity_dict with the documents (for each query)
        # similarity_dict will be a dict with query as key and a list of
        # tuples of (similarity, document text, doc_id) as value
        for j, (doc_i, doc_sim) in enumerate(similarity_dict[query_ids[i]]):
            similarity_dict[query_ids[i]][j] += tuple([doc_ids[doc_i]])

    return similarity_dict


def embed_jinja(model, docs, doc_ids, queries, query_ids, top_k=10):
    """
    Embed the queries and documents using the Jinja embeddings model and compute the similarity between queries and documents.
    Calls the retrieve function.

    Args:
        model: Jinja embeddings model.
        docs (dict): Dictionary with document_id as key and text as value.
        queries (list): List of queries.
        top_k (int): Number of most similar documents to retrieve.
    
    Returns:
        dict: Dictionary with query as key and a list of tuples of (similarity, document text, doc_id) as value.
    """
    # When calling the `encode` function, you can choose a `task` based on the use case:
    # 'retrieval.query', 'retrieval.passage', 'separation', 'classification', 'text-matching'
    # Alternatively, you can choose not to pass a `task`, and no specific LoRA adapter will be used.
    embeddings_queries = model.encode(queries, task="retrieval.query", max_length=MAX_QUERY_LEN)
    
    embeddings_docs = model.encode(docs, task="retrieval.passage", max_length=MAX_DOC_LEN)
    
    # Compute similarities
    similarity_dict = retrieve(query_ids, docs, doc_ids, embeddings_queries, embeddings_docs, top_k)
    return similarity_dict


def embed_bge(model, docs, doc_ids, queries, query_ids, top_k=10):
    """
    Embed the queries and documents using the BAAI embeddings models and compute the similarity between queries and documents.
    Calls the retrieve function.

    Args:
        model: BAAI embeddings model.
        docs (dict): Dictionary with document_id as key and text as value.
        queries (list): List of queries.
        top_k (int): Number of most similar documents to retrieve.

    Returns:   
        dict: Dictionary with query as key and a list of tuples of (similarity, document text, doc_id) as value.
    """
    embeddings_queries = model.encode(queries, batch_size=12, max_length=MAX_QUERY_LEN)['dense_vecs']
    embeddings_docs = model.encode(docs, batch_size=12, max_length=MAX_DOC_LEN)['dense_vecs']    # takes about 7min

    # Compute similarities
    similarity_dict = retrieve(query_ids, docs, doc_ids, embeddings_queries, embeddings_docs, top_k)
    return similarity_dict


def embed_bm25(docs, queries, doc_ids, query_ids, top_k=10):
    """
    Embed the queries and documents using the BM25 model and compute the similarity between queries and documents.

    Args:
        docs (dict): Dictionary with document_id as key and text as value.
        queries (list): List of queries.
        doc_ids (list): List of document IDs (Codigos).
        top_k (int): Number of most similar documents to retrieve.

    Returns:
        dict: Dictionary with query as key and a list of tuples of (similarity, document text, doc_id) as value.
    """
    from rank_bm25 import BM25Okapi
    import numpy as np

    # Simple space-based tokenization
    tokenized_corpus = [text.lower().split() for text in docs]

    # Create BM25 model
    bm25 = BM25Okapi(tokenized_corpus)

    # Queries
    tokenized_queries = [query.lower().split() for query in queries]

    # key: query, value: (similarity, text, doc_id)
    similarity_dict = {}
    for tokenized_query, qid in zip(tokenized_queries, query_ids):
        similarity_dict[qid] = []
        
        # Calcular las puntuaciones BM25 para la consulta en cada documento
        scores = bm25.get_scores(tokenized_query)

        # Ordenar los documentos por relevancia
        sorted_docs = np.argsort(scores)[::-1]  # Orden descendente
        for idx in sorted_docs[:top_k]:
            value = (idx, scores[idx], doc_ids[idx])
            similarity_dict[qid].append(value)
    return similarity_dict


def get_jinja_model():
    """ Load Jinja embeddings model."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
    return model


def get_bge_m3_model(checkpoint):
    """ Load BAAI embeddings model."""
    from FlagEmbedding import BGEM3FlagModel

    # model = BGEM3FlagModel(checkpoint, use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    model = BGEM3FlagModel(checkpoint) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    return model


def main():
    """
    Output is a df for each model (CSV file) with the following columns:
    - Query
    - doc_sim
    - Codigo
    - doc_text

    This step has to be followed by the pooling.py script to create a single
    CSV file with the top_k retrieved docs for each query.

    Example output:
    Query,doc_sim,Codigo,doc_text
    What is AI?,0.6265,492385,Artificial Intelligence (AI) is a branch of computer science [...] tasks that typically require human intelligence.

    For all the Queries in the queries_57.csv file,
    the top 30 most similar documents are retrieved using the BM25, BAAI and Jinja models.
    """
    from tqdm import tqdm
    import torch

    torch.cuda.empty_cache()

    # I want 30 annotated documents for each query
    top_k = 40

    doc_ids, docs = get_legal_dataset(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus_py.csv"))
    # assert len of texts is 5000
    assert len(docs) == 5000

    # Load queries
    queries_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "consultas_sinteticas_380.tsv")
    qids, queries = get_legal_queries(queries_path, header=0)
    assert len(queries) == 380

    models = [
        # "bm25",
        # "bge-m3",
        "Jinja",
    ]

    device = torch.device("cuda")

    for model_name in tqdm(models):

        print(f"Retrieving documents using {model_name} model...")
        
        if model_name == "bm25":
            # doc_ids = list(docs.keys())
            similarity_dict = embed_bm25(docs, queries, doc_ids, qids, top_k)
        elif model_name == "bge-m3":
            checkpoint = 'BAAI/bge-m3'
            # checkpoint = 'BAAI/bge-m3-unsupervised'
            # checkpoint = 'BAAI/bge-m3-retromae'
            model = get_bge_m3_model(checkpoint)
            similarity_dict = embed_bge(model, docs, doc_ids, queries, qids, top_k)

        elif model_name == "Jinja":
            model = get_jinja_model()
            model.to(device)
            similarity_dict = embed_jinja(model, docs, doc_ids, queries, qids, top_k)
        
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        if len(similarity_dict) == 0:
            raise ValueError("No documents retrieved.")
        # Prepare a list of rows
        rows = []
        for qid, doc_item in similarity_dict.items():
            for _, doc_sim, doc_id in doc_item:
                rows.append({"query_id": qid, "doc_sim": doc_sim, "Codigo": doc_id})

        # Create DataFrame
        df = pd.DataFrame(rows)

        path = os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", f"retrieved_docs_{model_name}.csv")
        df.to_csv(path, index=False)

        print(f"Retrieved documents saved to {path}")
    
        torch.cuda.empty_cache()



if __name__ == '__main__':
    main()