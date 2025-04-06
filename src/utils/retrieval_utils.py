import os
import numpy as np
from tqdm import tqdm
import torch
import pytrec_eval
from .train_utils import tokenize_with_manual_eos, get_eos_embeddings
from config.config import MAX_QUERY_LEN, MAX_DOC_LEN
import torch.nn.functional as F
try:
    import faiss
except ImportError:
    pass
import pandas as pd
import pickle


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
    Encode data in batches using the provided Jina model.
    Returns a list of all embeddings (on CPU).
    """
    from tqdm import trange

    embeddings = []
    for start_idx in trange(0, len(data), batch_size, desc="Batch encoding data"):
        batch = data[start_idx : start_idx + batch_size]
        batch_embeddings = model.encode(batch, task=task, max_length=max_length)
        batch_embeddings = np.array(batch_embeddings, dtype=np.float32)
        embeddings.extend(batch_embeddings)
    return embeddings


def retrieve(query_ids, doc_ids, similarity):
    run = {}
    for i in tqdm(range(len(query_ids)), desc="Creating run"):
        query_sim = similarity[i]
        
        run[str(query_ids[i])] = {str(doc_ids[j]): float(query_sim[j]) for j in range(len(doc_ids))}
    print("Run created.")
    return run


def compute_similarity(embeddings_queries, embeddings_docs, sim_type='dot'):
    """
    Given query and document embeddings, compute the similarity between queries and documents
    and return the similarity matrix.
    Dot-product and cosine similarity are supported.

    Parameters
    ----------
    embeddings_queries : torch.Tensor
        Tensor of query embeddings. Can be either 2D [num_queries, emb_dim] or 1D [emb_dim].
    embeddings_docs : torch.Tensor
        Tensor of document embeddings of shape [num_docs, emb_dim].
    sim_type : str, optional
        Type of similarity to use. Options: 'dot', 'cosine'. (default is 'dot').

    Returns
    -------
    torch.Tensor
        Similarity matrix of shape [num_queries, num_docs].
    """
    print("Computing similarity...", end="")

    # If there is just one query, unsqueeze to make it 2D
    if embeddings_queries.dim() == 1:
        embeddings_queries = embeddings_queries.unsqueeze(0)

    if sim_type == 'dot':
        similarities = []
        batch_size = 32
        for query_batch in torch.split(embeddings_queries, batch_size, dim=0):
            sim_batch = query_batch @ embeddings_docs.T
            similarities.append(sim_batch)
        similarity = torch.cat(similarities, dim=0)
    elif sim_type == 'cosine':
        # Normalize embeddings along the embedding dimension
        embeddings_queries = F.normalize(embeddings_queries, p=2, dim=1)
        embeddings_docs = F.normalize(embeddings_docs, p=2, dim=1)
        similarity = (embeddings_queries @ embeddings_docs.T) * 100
    else:
        raise ValueError(f"Invalid similarity type: {sim_type}")
    
    print("Done.")
    return similarity



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


def embed_jina_faiss(
    model,
    docs,
    queries,
    doc_ids,
    query_ids,
    top_k=1000,
    sim_type="dot"
):
    """
    Embed queries and documents using the Jina embeddings model, build a FAISS index
    for document embeddings on CPU in a streaming fashion, and retrieve the top-k most
    similar documents for each query.

    Parameters
    ----------
    model : object
        The Jina embeddings model.
    docs : list
        List of document texts.
    queries : list
        List of query texts.
    doc_ids : list
        List of document IDs corresponding to docs.
    query_ids : list
        List of query IDs.
    top_k : int, optional
        Number of top similar documents to retrieve for each query (default is 1000).
    sim_type : str, optional
        Similarity type: 'dot' for dot product or 'cosine' for cosine similarity
        (default is 'dot').

    Returns
    -------
    dict
        Run dictionary mapping each query ID to a dictionary of document IDs and similarity scores.
    """
    # 1) Encode Queries in Memory
    print("Encoding queries...")
    embeddings_queries = batch_encode_jina(
        model, queries, batch_size=128, task="retrieval.query", max_length=MAX_QUERY_LEN
    )
    # Convert to CPU tensor (float32)
    embeddings_queries = torch.tensor(np.array(embeddings_queries), dtype=torch.float32, device="cpu")
    print("Done encoding queries.")

    # 2) Prepare for streaming document embeddings
    print("Initializing FAISS CPU index for doc embeddings...")
    emb_dim = None
    cpu_index = None  # Will initialize once we know the embedding dimension
    all_doc_ids = []

    # 3) Stream Document Embeddings and Add to the CPU Index
    doc_iter = batch_encode_jina_stream(
        model, data=docs, ids=doc_ids, batch_size=128,
        task="retrieval.passage", max_length=MAX_DOC_LEN
    )

    first_batch = True
    for batch_embeddings, batch_ids in doc_iter:
        # batch_embeddings is already a float32 torch.Tensor on CPU
        batch_embeddings_np = batch_embeddings.numpy()

        # If using cosine similarity, L2-normalize before indexing
        if sim_type == "cosine":
            faiss.normalize_L2(batch_embeddings_np)

        if first_batch:
            # Figure out embedding dimension from first batch
            emb_dim = batch_embeddings_np.shape[1]
            # Create a CPU index for IP
            cpu_index = faiss.IndexFlatIP(emb_dim)
            first_batch = False

        # Add to the CPU index
        cpu_index.add(batch_embeddings_np)

        # Keep track of doc IDs in the order they were added
        all_doc_ids.extend(batch_ids)

    print("Done building CPU index.")

    # 4) Convert queries to numpy for FAISS search
    query_embeddings_np = embeddings_queries.numpy()
    if sim_type == "cosine":
        faiss.normalize_L2(query_embeddings_np)

    print(f"Searching FAISS index for top {top_k} docs per query...")
    distances, faiss_indices = cpu_index.search(query_embeddings_np, top_k)
    print("FAISS search complete.")

    # 5) Build Run Dictionary
    run = {}
    for i, qid in enumerate(query_ids):
        run[str(qid)] = {}
        for rank in range(top_k):
            doc_idx = faiss_indices[i, rank]
            if doc_idx < 0 or doc_idx >= len(all_doc_ids):
                continue
            doc_id = all_doc_ids[doc_idx]
            score = distances[i, rank]
            run[str(qid)][str(doc_id)] = float(score)

    return run


def embed_jina(model, docs, queries, doc_ids, query_ids):
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
    similarity = compute_similarity(query_ids, doc_ids, embeddings_queries, embeddings_docs)
    run = retrieve(query_ids, doc_ids, similarity)
    return run


def batch_encode_jina_stream(model, data, ids, batch_size, task, max_length):
    """
    Generator that encodes data in batches using the provided Jina model,
    yielding (embeddings, ids) on each iteration.

    Parameters
    ----------
    model : object
        The Jina embeddings model.
    data : list
        List of texts to be encoded.
    ids : list
        List of IDs, same length as data.
    batch_size : int
        Number of items to encode per batch.
    task : str
        Task identifier for the encoding (e.g., 'retrieval.passage').
    max_length : int
        Maximum token length for the encoding.

    Yields
    ------
    (torch.Tensor, list)
        A tuple with (embeddings for the batch, IDs for that batch).
    """
    from tqdm import trange

    n = len(data)
    for start_idx in trange(0, n, batch_size, desc="Batch streaming data"):
        end_idx = min(start_idx + batch_size, n)
        batch = data[start_idx:end_idx]
        batch_embeddings = model.encode(batch, task=task, max_length=max_length)
        # Convert to CPU float32 tensor
        batch_embeddings = torch.tensor(np.array(batch_embeddings), dtype=torch.float32, device="cpu")

        batch_ids = ids[start_idx:end_idx]
        yield batch_embeddings, batch_ids


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
        # print("Saving embeddings...", end="")
        # np.save(path, embeddings_docs)
        # print("Done.")
    else:
        # Load embeddings
        embeddings_docs = np.load(path)

    # Compute similarities
    similarity = compute_similarity(query_ids, doc_ids, torch.tensor(embeddings_queries, dtype=torch.float32), torch.tensor(embeddings_docs, dtype=torch.float32))
    run = retrieve(query_ids, doc_ids, similarity)
    return run


def get_sim_bge(model, docs, queries):
    embeddings_queries = model.encode(queries, batch_size=8, max_length=MAX_QUERY_LEN)['dense_vecs']
    print("Embedding docs...", end="")
    embeddings_docs = model.encode(docs, batch_size=8, max_length=MAX_DOC_LEN)['dense_vecs']
    print("Done.")
    
    # Compute similarities
    similarity = compute_similarity(torch.tensor(embeddings_queries, dtype=torch.float32), torch.tensor(embeddings_docs, dtype=torch.float32))
    return similarity


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


def embed_qwen_faiss(
    model,
    tokenizer,
    docs,
    doc_ids,
    queries,
    query_ids,
    top_k=1000,
    sim_type="dot",
    max_doc_len=150,
    max_query_len=20,
    batch_size=128
):
    """
    Embed queries and docs using Qwen, build a CPU FAISS index for doc embeddings,
    then retrieve the top_k docs per query.

    Parameters
    ----------
    model : torch.nn.Module
        The Qwen model in inference mode.
    tokenizer : PreTrainedTokenizer
        Tokenizer for Qwen.
    docs : list of str
        List of document texts.
    doc_ids : list
        List of doc IDs (parallel to docs).
    queries : list of str
        List of query texts.
    query_ids : list
        List of query IDs.
    top_k : int
        Number of docs to retrieve for each query.
    sim_type : str
        'dot' or 'cosine' similarity.
    max_doc_len : int
        Max token length for documents (including appended EOS).
    max_query_len : int
        Max token length for queries (including appended EOS).
    batch_size : int
        Batch size for tokenization + embedding.

    Returns
    -------
    run : dict
        TREC-style dictionary: run[qid][doc_id] = similarity_score.
    """
    # 1) Embed Queries in Memory
    print("Embedding queries in memory...")
    all_query_embeddings = []
    all_query_ids = []

    # We can re-use the same streaming logic for queries, or just do them in one pass if memory allows.
    query_generator = batch_embed_qwen_stream(
        model, tokenizer, queries, query_ids,
        batch_size=batch_size, max_length=max_query_len
    )

    for emb, qids in query_generator:
        all_query_embeddings.append(emb)
        all_query_ids.extend(qids)

    # Combine into a single tensor
    query_embeddings = torch.cat(all_query_embeddings, dim=0)  # (num_queries, hidden_dim)

    print("Queries embedded. Building FAISS index for docs...")

    # 2) Initialize the index after we see the first doc-embedding batch dimension
    cpu_index = None
    all_docids_ordered = []
    first_batch = True

    # 3) Stream doc embeddings
    doc_generator = batch_embed_qwen_stream(
        model, tokenizer, docs, doc_ids,
        batch_size=batch_size, max_length=max_doc_len
    )

    for doc_embeds, dids in doc_generator:
        doc_embeds_np = doc_embeds.numpy()  # shape (batch_size, hidden_dim)

        # If 'cosine', L2 normalize
        if sim_type == "cosine":
            faiss.normalize_L2(doc_embeds_np)

        if first_batch:
            emb_dim = doc_embeds_np.shape[1]
            # We set up an IndexFlatIP for dot-product
            # and let 'cosine' be handled by normalization
            cpu_index = faiss.IndexFlatIP(emb_dim)
            first_batch = False

        cpu_index.add(doc_embeds_np)
        all_docids_ordered.extend(dids)

    # 4) Convert query embeddings to NumPy for searching
    query_embeds_np = query_embeddings.numpy()
    if sim_type == "cosine":
        faiss.normalize_L2(query_embeds_np)

    print(f"Searching for top-{top_k} documents per query...")
    distances, faiss_indices = cpu_index.search(query_embeds_np, top_k)
    print("Search complete.")

    # 5) Build run dictionary: run[qid][docid] = float(similarity)
    run = {}
    for q_idx, qid in enumerate(all_query_ids):
        run[str(qid)] = {}
        for rank in range(top_k):
            doc_idx = faiss_indices[q_idx, rank]
            if doc_idx < 0 or doc_idx >= len(all_docids_ordered):
                continue
            doc_id = all_docids_ordered[doc_idx]
            score = distances[q_idx, rank]
            run[str(qid)][str(doc_id)] = float(score)

    return run


from tqdm import trange

def batch_embed_qwen_stream(
    model,
    tokenizer,
    texts,
    ids,
    batch_size,
    max_length
):
    """
    Generator that yields (embeddings, ids) for Qwen in CPU float32 batches.

    Parameters
    ----------
    model : torch.nn.Module
        Qwen model.
    tokenizer : PreTrainedTokenizer
        The tokenizer corresponding to the Qwen model.
    texts : list of str
        List of texts (queries or documents).
    ids : list
        List of corresponding IDs (query_ids or doc_ids).
    batch_size : int
        Number of samples per batch.
    max_length : int
        Maximum token length for each text in the batch.

    Yields
    ------
    tuple of (torch.Tensor, list)
        A tuple (embeddings of shape (batch_size, hidden_dim), list of IDs).
    """
    n = len(texts)
    for start_idx in trange(0, n, batch_size, desc="Embedding in batches"):
        end_idx = min(start_idx + batch_size, n)
        batch_texts = texts[start_idx:end_idx]
        batch_ids = ids[start_idx:end_idx]

        # Tokenize
        tokenized = tokenize_with_manual_eos(tokenizer, batch_texts, max_length=max_length)
        input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)

        # Move to GPU or CPU (depending on your model device)
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            embeddings = get_eos_embeddings(model, input_ids, attention_mask, tokenizer)
        # Move embeddings to CPU and float32
        embeddings = embeddings.cpu().float()

        yield embeddings, batch_ids


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


from contextlib import redirect_stdout

def embed_chunkwise_old(model, get_sim_func, docs, queries, doc_ids, query_ids, window_size=256):
    # Chunk the documents into smaller pieces.
    docs_chunked = []
    for doc in docs:
        doc_words = doc.split(" ")
        doc_chunked = [" ".join(doc_words[i:i+window_size]) 
                       for i in range(0, len(doc_words), window_size)]
        docs_chunked.append(doc_chunked)

    # Calculate the similarity between all queries and each document's chunks in parallel.
    similarities = []  # will store one similarity vector (one per query) for each document
    for doc_chunks in tqdm(docs_chunked, desc="Calculating similarities"):
        # get_sim_func returns a similarity matrix of shape [n_queries, n_chunks]
        with open(os.devnull, "w") as fnull, redirect_stdout(fnull):
            sim_matrix = get_sim_func(model, doc_chunks, queries)
        # For each query, take the maximum similarity over all chunks
        doc_sims, _ = sim_matrix.max(dim=1)  # shape: [n_queries]
        similarities.append(doc_sims)
    
    # Stack all document similarity vectors to form a [n_queries, n_docs] tensor.
    similarities = torch.stack(similarities, dim=1)
    
    return retrieve(query_ids, doc_ids, similarities)


def chunk_by_paragraphs(doc, window_size=256):
    """
    Chunk a document by paragraphs without splitting inside a paragraph.
    Paragraphs are defined by two consecutive newlines ("\n\n"). Consecutive paragraphs
    are merged until adding another paragraph would exceed the window_size in terms of word count.

    Parameters
    ----------
    doc : str
        The document text.
    window_size : int, optional
        Maximum number of words allowed per chunk (default is 256).

    Returns
    -------
    list of str
        List of text chunks, each containing one or more complete paragraphs.
    """
    # Split document into paragraphs and remove empty ones.
    paragraphs = [p.strip() for p in doc.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = []
    current_count = 0

    for para in paragraphs:
        word_count = len(para.split())
        # If adding this paragraph exceeds the window and we already have a chunk,
        # flush the current chunk.
        if current_chunk and (current_count + word_count > window_size):
            chunks.append(" ".join(current_chunk))
            current_chunk = [para]
            current_count = word_count
        else:
            current_chunk.append(para)
            current_count += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def embed_chunkwise(model, get_sim_func, docs, queries, doc_ids, query_ids, chunk_func, window_size=256):
    """
    Chunk documents using the provided chunking function and compute the similarity between all queries and 
    each document's chunks in one batch. Then, for each document, take the maximum similarity across its chunks.

    Parameters
    ----------
    model : object
        The model used for encoding.
    get_sim_func : callable
        A function that computes similarity given (model, docs, queries) and returns a similarity matrix.
        In your case, this is something like get_sim_bge.
    docs : list of str
        List of documents.
    queries : list of str
        List of queries.
    doc_ids : list
        List of document IDs.
    query_ids : list
        List of query IDs.
    chunk_func : callable
        Function that takes a document (str) and window_size, and returns a list of chunks.
    window_size : int, optional
        Parameter to pass to the chunking function (default is 256).

    Returns
    -------
    dict
        Retrieval run in the expected format (as returned by the retrieve function).
    """
    # Step 1: Chunk each document using the provided chunking function.
    docs_chunked = []
    doc_chunk_counts = []  # keep track of the number of chunks per document
    for doc in docs:
        chunks = chunk_func(doc, window_size)
        docs_chunked.append(chunks)
        doc_chunk_counts.append(len(chunks))
    
    # Step 2: Flatten all chunks into a single list.
    flatten_chunks = []
    for chunks in docs_chunked:
        flatten_chunks.extend(chunks)
    
    # Step 3: Compute similarity between all queries and all document chunks in one call.
    # sim_all will have shape [num_queries, total_chunks]
    sim_all = get_sim_func(model, flatten_chunks, queries)
    
    # Step 4: For each document, select its corresponding chunk columns and take max over chunks.
    num_queries = sim_all.shape[0]
    num_docs = len(docs)
    sim_matrix = torch.empty((num_queries, num_docs))
    
    start_idx = 0
    for doc_idx, count in enumerate(doc_chunk_counts):
        # For this document, select its corresponding columns.
        sim_doc = sim_all[:, start_idx:start_idx+count]  # shape: [num_queries, count]
        # Maximum similarity for each query for this document.
        sim_max, _ = sim_doc.max(dim=1)
        sim_matrix[:, doc_idx] = sim_max
        start_idx += count
    
    return retrieve(query_ids, doc_ids, sim_matrix)


def get_sim_sbert(model, docs, queries):
    embeddings_queries = model.encode(queries)
    embeddings_docs = model.encode(docs)
    similarity = model.similarity(embeddings_queries, embeddings_docs)
    return similarity


def embed_s_transformers(model, docs, queries, doc_ids, query_ids):
    embeddings_queries = model.encode(queries)
    embeddings_docs = model.encode(docs)
    similarity = model.similarity(embeddings_queries, embeddings_docs)
    return retrieve(query_ids, doc_ids, similarity)


def embed_s_transformers_faiss(
    model,
    docs: list,
    doc_ids: list,
    queries: list,
    query_ids: list,
    top_k: int = 1000,
    batch_size: int = 512,
    similarity: str = "dot",
):
    """
    Embed documents + queries with SentenceTransformers in a streaming fashion,
    build a FAISS index of doc embeddings, and retrieve top_k results per query.

    Parameters
    ----------
    model : SentenceTransformer
        The SentenceTransformers model for embedding.
    docs : list of str
        List of document texts.
    doc_ids : list
        List of document IDs, parallel to docs.
    queries : list of str
        List of query texts.
    query_ids : list
        List of query IDs.
    top_k : int, optional
        Number of docs to retrieve per query (default 1000).
    batch_size : int, optional
        Batch size for embedding docs (default 512).
    similarity : str, optional
        Either "dot" or "cosine". Defaults to "dot".

    Returns
    -------
    run : dict
        A run dict mapping query_id -> {doc_id: score, ...}
        suitable for TREC-style evaluation (pytrec_eval).
    """
    # ---- 1) Embed Queries in memory ----
    print("Embedding queries in memory...")
    query_embeddings = model.encode(queries, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    query_embeddings = query_embeddings.astype(np.float32)  # FAISS prefers float32

    if similarity == "cosine":
        # L2-normalize queries to mimic cosine with IndexFlatIP
        faiss.normalize_L2(query_embeddings)

    print("Finished embedding queries. Now building doc index...")

    # ---- 2) Create a FAISS index in CPU memory ----
    # For dot product or "IP", we can keep it simple with IndexFlatIP
    # If you're truly at 8.8M docs, consider IVFPQ or HNSW for memory usage
    index = None
    emb_dim = None
    all_doc_ids = []

    # ---- 2a) Stream doc embeddings in BATCHES ----
    # We'll embed docs in chunks to avoid OOM
    num_docs = len(docs)
    for start_idx in tqdm(range(0, num_docs, batch_size), desc="Indexing docs"):
        end_idx = min(start_idx + batch_size, num_docs)
        doc_batch = docs[start_idx:end_idx]
        doc_id_batch = doc_ids[start_idx:end_idx]

        # Embed the documents
        doc_embeddings = model.encode(doc_batch, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        doc_embeddings = doc_embeddings.astype(np.float32)

        if similarity == "cosine":
            # L2-normalize doc embeddings if using "cosine"
            faiss.normalize_L2(doc_embeddings)

        # On first batch, init the index
        if index is None:
            emb_dim = doc_embeddings.shape[1]
            index = faiss.IndexFlatIP(emb_dim)  # Dot-product index
        # Add embeddings
        index.add(doc_embeddings)
        all_doc_ids.extend(doc_id_batch)

    print(f"Done building index with {index.ntotal} documents.")

    # ---- 3) Search for top_k docs per query ----
    print("Searching index for each query...")
    distances, faiss_indices = index.search(query_embeddings, top_k)
    print("Done searching.")

    # ---- 4) Build the run dictionary (for TREC eval) ----
    run = {}
    for q_idx, qid in enumerate(query_ids):
        run[str(qid)] = {}
        for rank in range(top_k):
            doc_idx = faiss_indices[q_idx, rank]
            if doc_idx < 0 or doc_idx >= len(all_doc_ids):
                continue
            doc_id = all_doc_ids[doc_idx]
            score = distances[q_idx, rank]
            run[str(qid)][str(doc_id)] = float(score)

    return run


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


def rerank_cross_encoder(model, tokenizer, run, top_k, queries, query_ids, docs, doc_ids, max_length, batch_size=8):
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
            batch_docs = [doc_dict[int(doc_id)] for doc_id in batch_doc_ids]
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

    with open("metrics_per_query.pkl", "wb") as f:
        pickle.dump(results, f)

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


def create_predictions_file(run, run_id="my_run"):
    """
    Generates a TREC-formatted predictions file from the run dictionary.

    The run dictionary has the following structure:
        {
            str(query_id): {str(doc_id): float(score), ...},
            ...
        }
    
    The TREC format for a result is:
        <query_id> Q0 <doc_id> <rank> <score> <run_id>
    
    Parameters
    ----------
    run : dict
        Dictionary mapping query IDs to dictionaries of document IDs and their scores.
    run_id : str, optional
        Identifier for the run (default is "my_run").
    
    Returns
    -------
    None
        Writes the results to a file named "predictions.tsv".
    """
    with open("predictions.tsv", "w") as f:
        for query_id, doc_scores in run.items():
            # Sort documents by score in descending order and take top 10
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
                # Write in TREC format: query_id, Q0, doc_id, rank, score, run_id
                f.write(f"{query_id}\tQ0\t{doc_id}\t{rank}\t{score:.4f}\t{run_id}\n")


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