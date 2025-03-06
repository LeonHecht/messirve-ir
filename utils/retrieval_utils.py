import os
import numpy as np
from tqdm import tqdm
import torch
import pytrec_eval
from .train_utils import tokenize_with_manual_eos, get_eos_embeddings
from config import MAX_QUERY_LEN, MAX_DOC_LEN
import torch.nn.functional as F
# import faiss
import pandas as pd


def build_faiss_index(embeddings, use_cosine=False):
    """
    Build a FAISS index from document embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Array of document embeddings with shape (num_docs, emb_dim).
    use_cosine : bool, optional
        If True, normalize embeddings for cosine similarity (default is False).

    Returns
    -------
    index : faiss.Index
        A FAISS index built from the provided embeddings.
    """
    if use_cosine:
        faiss.normalize_L2(embeddings)
    emb_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(emb_dim)  # Using inner product for dot-product similarity.
    index.add(embeddings)
    return index


def search_faiss_index(index, query_embeddings, top_k, use_cosine=False):
    """
    Search the FAISS index to retrieve the top-k nearest neighbors for each query.

    Parameters
    ----------
    index : faiss.Index
        The FAISS index built from document embeddings.
    query_embeddings : np.ndarray
        Array of query embeddings with shape (num_queries, emb_dim).
    top_k : int
        Number of nearest neighbors to retrieve per query.
    use_cosine : bool, optional
        If True, normalize query embeddings for cosine similarity (default is False).

    Returns
    -------
    distances : np.ndarray
        Array of shape (num_queries, top_k) containing similarity scores.
    indices : np.ndarray
        Array of shape (num_queries, top_k) containing indices of the nearest documents.
    """
    if use_cosine:
        faiss.normalize_L2(query_embeddings)
    distances, indices = index.search(query_embeddings, top_k)
    return distances, indices


def batch_encode_jina(model, data, batch_size, task, max_length):
    """
    Encode data in batches using the provided model.

    Parameters
    ----------
    model : object
        The Jinja embeddings model.
    data : list or dict
        Data to be encoded. If a dict, the values will be used.
    batch_size : int
        Number of items to encode per batch.
    task : str
        Task identifier for the encoding (e.g., 'retrieval.query' or 'retrieval.passage').
    max_length : int
        Maximum token length for the encoding.

    Returns
    -------
    list
        A list of embeddings corresponding to the input data.
    """
    # If data is a dict, convert to list of values.
    if isinstance(data, dict):
        data = list(data.values())

    embeddings = []
    for i in tqdm(range(0, len(data), batch_size), desc="Batch encoding data"):
        batch = data[i : i + batch_size]
        batch_embeddings = model.encode(batch, task=task, max_length=max_length)
        embeddings.extend(batch_embeddings)
    return embeddings


def retrieve(query_ids, doc_ids, similarity):
    run = {}
    for i in tqdm(range(len(query_ids)), desc="Creating run"):
        query_sim = similarity[i]
        
        run[str(query_ids[i])] = {str(doc_ids[j]): float(query_sim[j]) for j in range(len(doc_ids))}
    print("Run created.")
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
    print("Computing similarity...", end="")
    if sim_type == 'dot':
        similarities = []
        batch_size = 32
        for query_batch in torch.split(embeddings_queries, batch_size, dim=0):
            sim_batch = query_batch @ embeddings_docs.T
            similarities.append(sim_batch)
        similarity = torch.cat(similarities, dim=0)
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


def compute_similarity_streaming(doc_embeddings_iterator, embeddings_queries,
                                 top_k=10, sim_type='dot'):
    """
    Compute similarity between queries and documents in a streaming fashion without
    loading all document embeddings into memory. Returns the top_k document IDs and
    their similarity scores for each query.

    Parameters
    ----------
    doc_embeddings_iterator : iterator
        Iterator that yields tuples of (doc_embeddings, doc_ids) for each batch.
    embeddings_queries : torch.Tensor
        Tensor of query embeddings with shape (num_queries, embedding_dim).
    top_k : int, optional
        Number of top similar documents to retrieve for each query (default is 10).
    sim_type : str, optional
        Similarity type to use: 'dot' for dot product or 'cosine' for cosine similarity
        (default is 'dot').

    Returns
    -------
    tuple
        A tuple (top_doc_ids, top_scores) where:
            top_doc_ids : list of lists
                Each sublist contains the top_k document IDs for the corresponding query.
            top_scores : torch.Tensor
                Tensor of shape (num_queries, top_k) with the similarity scores.
    """
    num_queries = embeddings_queries.size(0)
    device = embeddings_queries.device

    # Initialize top scores and corresponding doc IDs for each query
    top_scores = torch.full((num_queries, top_k), float('-inf'), device=device)
    top_doc_ids = [[None] * top_k for _ in range(num_queries)]

    if sim_type == 'cosine':
        embeddings_queries = F.normalize(embeddings_queries, p=2, dim=1)

    for doc_batch_embeddings, doc_batch_ids in doc_embeddings_iterator:
        doc_batch_embeddings = doc_batch_embeddings.to(device)
        if sim_type == 'cosine':
            doc_batch_embeddings = F.normalize(doc_batch_embeddings, p=2, dim=1)

        # Compute similarity between all queries and the current doc batch.
        sim_batch = embeddings_queries @ doc_batch_embeddings.T  # (num_queries, batch_size)

        # Concatenate current top_scores with new batch scores along dimension=1.
        combined_scores = torch.cat([top_scores, sim_batch], dim=1)
        new_topk_scores, indices = combined_scores.topk(k=top_k, dim=1)

        # Update top_doc_ids based on the new indices.
        new_top_doc_ids = []
        for i in range(num_queries):
            # Build a combined list of previous doc IDs and current batch doc IDs.
            combined_ids = top_doc_ids[i] + doc_batch_ids
            # Select the doc IDs corresponding to the topk indices.
            new_ids = [combined_ids[idx] for idx in indices[i].tolist()]
            new_top_doc_ids.append(new_ids)
        top_scores = new_topk_scores
        top_doc_ids = new_top_doc_ids

    return top_doc_ids, top_scores


def embed_jina_faiss(model, docs, queries, doc_ids, query_ids, top_k=1000, sim_type='dot'):
    """
    Embed queries and documents using the Jinja embeddings model, build a FAISS index
    for the document embeddings in a streaming fashion, and retrieve the top-k most similar
    documents for each query.

    Parameters
    ----------
    model : object
        The Jinja embeddings model.
    docs : list
        List of document texts.
    queries : list
        List of query texts.
    doc_ids : list
        List of document IDs corresponding to docs.
    query_ids : list
        List of query IDs.
    top_k : int, optional
        Number of top similar documents to retrieve for each query (default is 10).
    sim_type : str, optional
        Similarity type: 'dot' for dot product or 'cosine' for cosine similarity (default is 'dot').

    Returns
    -------
    dict
        Run dictionary mapping each query ID to a dictionary of document IDs and similarity scores.
    """
    # 1) Encode Queries (Fits in Memory)
    print("Encoding queries...", end="")
    embeddings_queries = batch_encode_jina(
        model, queries, batch_size=128, task="retrieval.query", max_length=MAX_QUERY_LEN
    )
    embeddings_queries = torch.tensor(np.array(embeddings_queries), dtype=torch.float32)
    print("Done.")

    # 2) Create a CPU index as a placeholder, then transfer to GPU
    print("Initializing GPU index...", end="")
    # We'll use inner product (dot product) as the metric.
    # If you want L2 distance or something else, choose a different index type.
    emb_dim = None  # We'll figure this out after we get the first batch of doc embeddings
    cpu_index = None
    gpu_res = faiss.StandardGpuResources()  # Manages GPU resources
    gpu_index = None
    print("Done.")

    # 3) Stream Document Embeddings and Add Them to the GPU Index
    print("Building FAISS index from document embeddings (streaming)...", end="")
    doc_iter = batch_encode_jina_stream(
        model, data=docs, ids=doc_ids, batch_size=128,
        task="retrieval.passage", max_length=MAX_DOC_LEN
    )

    all_doc_ids = []
    first_batch = True

    for batch_embeddings, batch_ids in doc_iter:
        # Convert to NumPy
        batch_embeddings_np = batch_embeddings.cpu().numpy().astype(np.float32)

        # If using cosine similarity, normalize each batch
        if sim_type == 'cosine':
            faiss.normalize_L2(batch_embeddings_np)

        # Initialize the index on the very first batch
        if first_batch:
            print("Processing first batch...")
            emb_dim = batch_embeddings_np.shape[1]
            # Create a CPU index for dot product
            cpu_index = faiss.IndexFlatIP(emb_dim)
            # Transfer CPU index to GPU
            print("About to transfer index to GPU...")
            gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
            print("Done.")
            first_batch = False

        print(f"Adding batch of {batch_embeddings_np.shape[0]} documents to the index...")
        # Add the batch to the GPU index
        gpu_index.add(batch_embeddings_np)
        print("Done")

        # Keep track of the doc IDs in the order they're added
        all_doc_ids.extend(batch_ids)
    print("Done.")

    # 4) Search the GPU Index with Query Embeddings
    print("Searching FAISS index for similar documents...", end="")
    query_embeddings_np = embeddings_queries.cpu().numpy().astype(np.float32)

    if sim_type == 'cosine':
        faiss.normalize_L2(query_embeddings_np)

    distances, faiss_indices = gpu_index.search(query_embeddings_np, top_k)
    print("Done.")

    # 5) Build Run Dictionary: query_id -> {doc_id: similarity}
    run = {}
    for i, qid in enumerate(query_ids):
        # For each query, we have top_k FAISS indices
        # Map each index back to the original doc_id
        run[str(qid)] = {}
        for rank in range(top_k):
            doc_idx = faiss_indices[i, rank]
            if doc_idx < 0 or doc_idx >= len(all_doc_ids):
                continue  # This can happen if FAISS returns -1 for empty index, etc.
            doc_id = all_doc_ids[doc_idx]
            score = distances[i, rank]
            run[str(qid)][str(doc_id)] = float(score)

    return run


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
    print("Encoding queries...", end="")
    embeddings_queries = batch_encode_jina(
        model, queries, batch_size=128, task="retrieval.query", max_length=MAX_QUERY_LEN
    )
    print("Done.")

    print("Encoding docs...", end="")
    embeddings_docs = batch_encode_jina(
        model, docs, batch_size=128, task="retrieval.passage", max_length=MAX_DOC_LEN
    )
    print("Done.")

    embeddings_queries = torch.tensor(np.array(embeddings_queries), dtype=torch.float32)
    embeddings_docs = torch.tensor(np.array(embeddings_docs), dtype=torch.float32)

    # Compute similarities
    run = compute_similarity(query_ids, doc_ids, embeddings_queries, embeddings_docs)
    return run


def batch_encode_jina_stream(model, data, ids, batch_size, task, max_length):
    """
    Generator that yields embeddings in batches.

    Parameters
    ----------
    model : object
        The Jinja embeddings model.
    data : list
        Data to be encoded.
    batch_size : int
        Number of items to encode per batch.
    task : str
        Task identifier for the encoding.
    max_length : int
        Maximum token length for the encoding.
    ids : list, optional
        List of IDs corresponding to the data. If provided, these will be used
        instead of auto-generated indices.

    Yields
    ------
    tuple of (torch.Tensor, list)
        A tuple containing a tensor of embeddings for the current batch and the
        corresponding IDs.
    """
    for i in tqdm(range(0, len(data), batch_size), desc="Batch encoding data", total=len(data)//batch_size):
        batch = data[i: i + batch_size]
        print(f"Processing batch {i//batch_size}: {len(batch)} documents")
        batch_embeddings = model.encode(batch, task=task, max_length=max_length)
        print(f"Finished batch {i//batch_size}")
        batch_embeddings = torch.tensor(np.array(batch_embeddings), dtype=torch.float32)
        print(f"Converted to tensor")
        batch_ids = ids[i: i + batch_size]
        yield batch_embeddings, batch_ids


def embed_jinja_stream(model, docs, queries, doc_ids, query_ids, top_k=1000, sim_type='dot'):
    """
    Embed queries and documents using the Jinja embeddings model and compute similarities
    in a streaming fashion to avoid loading all document embeddings into memory.

    Parameters
    ----------
    model : object
        The Jinja embeddings model.
    docs : dict
        Dictionary mapping document IDs to document text.
    queries : list
        List of query texts.
    doc_ids : list
        List of document IDs (should match keys of docs).
    query_ids : list
        List of query IDs.
    top_k : int, optional
        Number of top similar documents to retrieve for each query (default is 10).
    sim_type : str, optional
        Type of similarity to compute: 'dot' or 'cosine' (default is 'dot').

    Returns
    -------
    dict
        Run dictionary mapping each query ID to a dictionary of document IDs and similarity scores.
    """
    print("Encoding queries...", end="")
    # Encode all queries (assumed to be small enough to fit into memory)
    embeddings_queries = batch_encode_jina(model, queries, batch_size=128,
                                            task="retrieval.query", max_length=MAX_QUERY_LEN)
    embeddings_queries = torch.tensor(np.array(embeddings_queries), dtype=torch.float32)
    print("Done.")

    print("Streaming document embeddings...", end="")
    # Create a streaming generator for document embeddings.
    doc_embeddings_iterator = batch_encode_jina_stream(model, docs, doc_ids, batch_size=128,
                                                       task="retrieval.passage", max_length=MAX_DOC_LEN)
    print("Done.")

    print("Computing similarity in streaming mode...", end="")
    top_doc_ids, top_scores = compute_similarity_streaming(doc_embeddings_iterator,
                                                             embeddings_queries, top_k=top_k, sim_type=sim_type)
    print("Done.")

    run = retrieve(query_ids, top_doc_ids, top_scores)
    return run


def embed_bge(model, docs, queries, doc_ids, query_ids, reuse_run):
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
    if not os.path.exists(path) or not reuse_run:
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
    run = compute_similarity(query_ids, doc_ids, torch.tensor(embeddings_queries, dtype=torch.float32), torch.tensor(embeddings_docs, dtype=torch.float32))
    return run


def embed_qwen(model, tokenizer, docs, queries, doc_ids, query_ids):
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
    from torch.utils.data import DataLoader, TensorDataset
    batch_size = 8
    device = next(model.parameters()).device  # Automatically detect model's device
    print("Device: ", device)

    inputs_docs = tokenize_with_manual_eos(tokenizer, docs, max_length=MAX_DOC_LEN)
    inputs_queries = tokenize_with_manual_eos(tokenizer, queries, max_length=MAX_QUERY_LEN)

    print("Docs and queries tokenized.")

    # Create DataLoaders for batching
    doc_input_ids = torch.tensor(inputs_docs["input_ids"], dtype=torch.long)
    doc_attention_mask = torch.tensor(inputs_docs["attention_mask"], dtype=torch.long)
    doc_dataset = TensorDataset(doc_input_ids, doc_attention_mask)
    
    query_input_ids = torch.tensor(inputs_queries["input_ids"], dtype=torch.long)
    query_attention_mask = torch.tensor(inputs_queries["attention_mask"], dtype=torch.long)
    query_dataset = TensorDataset(query_input_ids, query_attention_mask)

    doc_loader = DataLoader(doc_dataset, batch_size=batch_size, pin_memory=True)
    query_loader = DataLoader(query_dataset, batch_size=batch_size, pin_memory=True)

    print("Embedding docs and queries...", end="")
    embeddings_queries = []
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(query_loader, desc="Embedding queries"):
            query_embeds = get_eos_embeddings(model, input_ids.to(device), attention_mask.to(device), tokenizer)
            embeddings_queries.append(query_embeds)
    embeddings_queries = torch.cat(embeddings_queries, dim=0)  # Combine batches
    print("Done.")

    embeddings_docs = []
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(doc_loader, desc="Embedding docs"):
            doc_embeds = get_eos_embeddings(model, input_ids.to(device), attention_mask.to(device), tokenizer)
            embeddings_docs.append(doc_embeds)
    embeddings_docs = torch.cat(embeddings_docs, dim=0)  # Combine batches

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

    print("Tokenizing corpus...", end="")
    # Simple space-based tokenization
    tokenized_corpus = [doc.lower().split() for doc in tqdm(docs)]
    print("Done.")

    print("Creating BM25 model...", end="")
    # Create BM25 model
    bm25 = BM25Okapi(tokenized_corpus)
    print("Done.")

    print("Tokenizing queries...", end="")
    # Queries
    tokenized_queries = [query.lower().split() for query in tqdm(queries)]
    print("Done.")

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


def merge_reranked_into_full_run(full_run, reranked_run):
    """
    full_run: dict mapping query_id -> {doc_id: original_bi_encoder_score, ...}
    reranked_run: dict mapping query_id -> {doc_id: cross_encoder_score, ...} for top-k documents
    """
    for query_id, reranked_docs in reranked_run.items():
        # Set non re-ranked documents to 0, then update the ones you reranked
        updated_run = {doc_id: 0 for doc_id in full_run[query_id]}
        for doc_id, new_score in reranked_docs.items():
            updated_run[doc_id] = new_score
        # Optionally, re-sort the dictionary based on scores in descending order:
        full_run[query_id] = dict(sorted(updated_run.items(), key=lambda x: x[1], reverse=True))
    return full_run


def rerank_cross_encoder(model, tokenizer, run, top_k, queries, query_ids, docs, doc_ids, max_length, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # set the model to evaluation mode

    doc_dict = {doc_id: doc for doc_id, doc in zip(doc_ids, docs)}
    query_dict = {str(query_id): query for query_id, query in zip(query_ids, queries)}

    reranked_run = {}

    # cut the run to top_k
    for query_id in run:
        reranked_run[query_id] = dict(sorted(run[query_id].items(), key=lambda x: x[1], reverse=True)[:top_k])

    # Loop over each query
    for query_id, doc_score_dict in tqdm(reranked_run.items(), total=len(reranked_run)):
        curr_doc_ids = list(doc_score_dict.keys())
        similarity_scores = []

        # Process the docs in batches for the current query
        for start_idx in range(0, len(curr_doc_ids), batch_size):
            end_idx = min(start_idx + batch_size, len(curr_doc_ids))
            batch_doc_ids = curr_doc_ids[start_idx:end_idx]
            batch_docs = [doc_dict[doc_id] for doc_id in batch_doc_ids]
            # Tokenize the current batch; the same query is paired with each document in the batch
            encodings = tokenizer([query_dict[query_id]] * len(batch_docs), batch_docs,
                                  truncation=True, padding=True,
                                  max_length=max_length, return_tensors="pt")
            # Move tensors to the appropriate device
            for key in encodings:
                encodings[key] = encodings[key].to(device)

            # Forward pass in inference mode
            with torch.no_grad():
                logits = model(**encodings).logits

            # Compute similarity scores (assuming index 1 is the "relevant" class)
            batch_similarity = torch.nn.functional.softmax(logits, dim=1)[:, 1].cpu().numpy()
            similarity_scores.extend(batch_similarity)

        # Map each document to its similarity score for this query
        reranked_run[query_id] = {curr_doc_ids[j]: float(similarity_scores[j]) for j in range(len(curr_doc_ids))}

    return merge_reranked_into_full_run(run, reranked_run)


def get_eval_metrics(run, qrels_dev_df, all_docids, metrics):
    # Evaluate BM25
    # qrels = {query_id: {doc_id: relevance, ...},
    #          query_id: {doc_id: relevance, ...}, ...},
    # convert qrels_dev_df to qrels dict
    qrels = {}
    for _, row in qrels_dev_df.iterrows():
        # row headers "query_id", "iteration", "doc_id", "relevance"
        query_id = str(row["query_id"])
        doc_id = str(row["doc_id"])
        relevance = int(row["relevance"])

        if query_id not in qrels:
            qrels[query_id] = {}
        qrels[query_id][doc_id] = relevance
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    results = evaluator.evaluate(run)
    print("Evaluation done.")

    result_values = list(results.values())
    metric_names = list(result_values[0].keys())      # because some result names change e.g. from ndcg_cut.10 to ndcg_cut_10
    metric_sums = {metric_name: 0 for metric_name in metric_names}
    for metrics_ in results.values():
        for metric in metric_names:
            metric_sums[metric] += metrics_[metric]
    
    # Average metrics over all queries
    # assert len(results) == len(query_ids)
    avg_metrics = {metric_name: metric_sums[metric_name]/len(results) for metric_name in metric_names}
    
    print("\nResults:")
    for metric_name, metric_value in avg_metrics.items():
        print(f"Average {metric_name}: {metric_value}")

    print("\n")
    
    return avg_metrics


def create_results_file(run):
    """
    run has the following stucture:
    run[str(query_ids[i])] = {str(doc_ids[j]): float(query_sim[j]) for j in range(len(doc_ids))}
    {query_id: {doc_id: similarity, ...}, ...}
    """
    # sort run dict by similarity
    for query_id in run:
        run[query_id] = dict(sorted(run[query_id].items(), key=lambda x: x[1], reverse=True)[:10])
    
    with open("results.txt", "w") as f:
        for query_id, doc_dict in run.items():
            for i, (doc_id, similarity) in enumerate(doc_dict.items()):
                f.write(f"{query_id}\t{doc_id}\t{i+1}\n")


def get_legal_dataset(path):
    # Load the dataset
    df = pd.read_csv(path, usecols=["Codigo", "text"])
    # convert Codigo column to list
    df["Codigo"] = df["Codigo"].astype(int)
    return df["Codigo"].tolist(), df["text"].tolist()


def get_legal_queries(path):
    # Load the queries
    df = pd.read_csv(path, usecols=["topic_id", "Query"])
    # convert topic_id column to list
    df["topic_id"] = df["topic_id"].astype(int)
    return df["topic_id"].tolist(), df["Query"].tolist()