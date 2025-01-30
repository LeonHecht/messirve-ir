import os
import numpy as np
from tqdm import tqdm
import torch

# Globals
MAX_QUERY_LEN = 256
MAX_DOC_LEN = 2048


def retrieve(query_ids, doc_ids, similarity):
    print("Creating run...", end="")
    run = {}
    for i in range(len(query_ids)):
        query_sim = similarity[i]
        
        run[str(query_ids[i])] = {str(doc_ids[j]): float(query_sim[j]) for j in range(len(doc_ids))}
    print("Done.")
    return run


def compute_similarity(query_ids, doc_ids, embeddings_queries, embeddings_docs, sim_type='dot'):
    """
    Given a list of queries and documents, and their embeddings, compute the similarity between queries and documents
    and return the top_k most similar documents for each query.
    Dot-product and cosine similarity are supported.

    Args:
        queries (list): List of queries.
        docs (list): List of documents.
        query_ids (list): List of query IDs.
        doc_ids (list): List of document IDs (Codigos).
        embeddings_queries (np.ndarray): Embeddings of queries.
        embeddings_docs (np.ndarray): Embeddings of documents.
        top_k (int): Number of most similar documents to retrieve.
        sim_type (str): Type of similarity to use. Options: 'dot', 'cosine'.
    
    Returns:
        dict: Dictionary with query_id as key and a dict of {doc_id: similarity, ...} as value
        (run format for pytrec_eval library).
    """
    import torch.nn.functional as F
    
    print("Computing similarity...", end="")
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
    print("Done.")

    return retrieve(query_ids, doc_ids, similarity)


def embed_jinja(model, docs, queries, doc_ids, query_ids):
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
    path = 'corpus/embeddings_corpus_jinja.npy'

    if not os.path.exists(path):
        embeddings_docs = model.encode(docs, task="retrieval.passage", max_length=MAX_DOC_LEN)
        # save embeddings
        np.save(path, embeddings_docs)
    else:
        # Load embeddings
        embeddings_docs = np.load(path)

    # Compute similarities
    run = compute_similarity(query_ids, doc_ids, embeddings_queries, embeddings_docs)
    return run


def embed_bge(model, docs, queries, doc_ids, query_ids):
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
    embeddings_queries = model.encode(queries, batch_size=8, max_length=MAX_QUERY_LEN)['dense_vecs']
    # Embed entire corpus if file does not exist
    path = 'corpus/embeddings_train_corpus_bge-m3.npy'
    if not os.path.exists(path):
        print("Embedding docs...", end="")
        embeddings_docs = model.encode(docs, batch_size=8, max_length=MAX_DOC_LEN)['dense_vecs']    # takes about 7min
        print("Done.")
        # save embeddings
        print("Saving embeddings...", end="")
        np.save(path, embeddings_docs)
        print("Done.")
    else:
        # Load embeddings
        embeddings_docs = np.load(path)

    # Compute similarities
    run = compute_similarity(query_ids, doc_ids, embeddings_queries, embeddings_docs)
    return run


def embed_mamba(model, tokenizer, docs, queries, doc_ids, query_ids):
    from torch.utils.data import DataLoader, TensorDataset
    batch_size = 8
    device = next(model.parameters()).device  # Automatically detect model's device
    print("Device: ", device)

    inputs_docs = tokenizer(
        docs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_DOC_LEN
    )
    inputs_queries = tokenizer(
        queries,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_QUERY_LEN
    )
    print("Docs and queries tokenized.")
    inputs_docs = {key: tensor.to(device) for key, tensor in inputs_docs.items()}  # Move to device
    inputs_queries = {key: tensor.to(device) for key, tensor in inputs_queries.items()}  # Move to device

    # eos_token_id = tokenizer.eos_token_id
    # eos_tokens = torch.full((inputs_docs["input_ids"].size(0), 1), eos_token_id, dtype=torch.long).to(device)
    # attention_tokens = torch.ones((inputs_docs["attention_mask"].size(0), 1), dtype=torch.long).to(device)

    # inputs_docs["input_ids"] = torch.cat([inputs_docs["input_ids"], eos_tokens], dim=1)
    # inputs_docs["attention_mask"] = torch.cat([inputs_docs["attention_mask"], attention_tokens], dim=1)    

    # eos_token_id = tokenizer.eos_token_id
    # eos_tokens = torch.full((inputs_queries["input_ids"].size(0), 1), eos_token_id, dtype=torch.long).to(device)
    # attention_tokens = torch.ones((inputs_queries["attention_mask"].size(0), 1), dtype=torch.long).to(device)

    # inputs_queries["input_ids"] = torch.cat([inputs_queries["input_ids"], eos_tokens], dim=1)
    # inputs_queries["attention_mask"] = torch.cat([inputs_queries["attention_mask"], attention_tokens], dim=1)
    
    # Create DataLoaders for batching
    doc_dataset = TensorDataset(inputs_docs["input_ids"], inputs_docs["attention_mask"])
    query_dataset = TensorDataset(inputs_queries["input_ids"], inputs_queries["attention_mask"])
    doc_loader = DataLoader(doc_dataset, batch_size=batch_size)
    query_loader = DataLoader(query_dataset, batch_size=batch_size)

    print("Embedding docs and queries...")
    embeddings_docs = []
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(doc_loader):
            batch = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}
            output = model(**batch)
            embeddings_docs.append(output.logits.mean(dim=1))  # Mean pooling
    embeddings_docs = torch.cat(embeddings_docs, dim=0)  # Combine batches

    embeddings_queries = []
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(query_loader):
            batch = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}
            output = model(**batch)
            embeddings_queries.append(output.logits.mean(dim=1))  # Mean pooling
    embeddings_queries = torch.cat(embeddings_queries, dim=0)  # Combine batches
    print("Embeddings done.")

    run = compute_similarity(query_ids, doc_ids, embeddings_queries, embeddings_docs)
    return run


def embed_s_transformers(model, docs, queries, doc_ids, query_ids):
    embeddings_queries = model.encode(queries)
    embeddings_docs = model.encode(docs)
    similarity = model.similarity(embeddings_queries, embeddings_docs)
    return retrieve(query_ids, doc_ids, similarity)


def retrieve_bm25(docs, queries, doc_ids, query_ids):
    """
    Embed the queries and documents using the BM25 model and compute the similarity between queries and documents.

    Args:
        docs (dict): Dictionary with document_id as key and text as value.
        queries (list): List of queries.
        doc_ids (list): List of document IDs.
        top_k (int): Number of most similar documents to retrieve.

    Returns:
        dict: Dictionary with (query_id, doc_id) as key and a list of tuples of score as value.
    """
    from rank_bm25 import BM25Okapi

    # Simple space-based tokenization
    tokenized_corpus = [doc.lower().split() for doc in docs]

    # Create BM25 model
    bm25 = BM25Okapi(tokenized_corpus)

    # Queries
    tokenized_queries = [query.lower().split() for query in queries]

    # key: query, value: (similarity, text, doc_id)
    run = {}
    for tokenized_query, query_id in tqdm(zip(tokenized_queries, query_ids), total=len(tokenized_queries)):
        # Calcular las puntuaciones BM25 para la consulta en cada documento
        scores = bm25.get_scores(tokenized_query)

        run_query = {}
        for doc_id, score in zip(doc_ids, scores):
            run_query[str(doc_id)] = score
        run[str(query_id)] = run_query
    return run
