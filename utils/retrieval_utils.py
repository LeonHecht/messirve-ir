import os
import numpy as np
from tqdm import tqdm
import torch
import pytrec_eval
from .train_utils import tokenize_with_manual_eos, get_eos_embeddings
from config import MAX_QUERY_LEN, MAX_DOC_LEN


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
    import torch.nn.functional as F
    
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
        model, docs, batch_size=256, task="retrieval.passage", max_length=MAX_DOC_LEN
    )
    print("Done.")

    embeddings_queries = torch.tensor(embeddings_queries, dtype=torch.float32)
    embeddings_docs = torch.tensor(embeddings_docs, dtype=torch.float32)

    # Compute similarities
    run = compute_similarity(query_ids, doc_ids, embeddings_queries, embeddings_docs)
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
    run = compute_similarity(query_ids, doc_ids, embeddings_queries, embeddings_docs)
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
    import torch.nn.functional as F
    batch_size = 32
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
        # qrels[str(row["query_id"])] = {str(docid): 0 for docid in all_docids}
        qrels[str(row["query_id"])] = {}
        qrels[str(row["query_id"])][str(row["doc_id"])] = 1
    
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
