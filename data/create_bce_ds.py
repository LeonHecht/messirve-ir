from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import json
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import os
import sys
print(f"Executable: {sys.executable}")
from typing import List, Dict, Set

import subprocess
from typing import List, Set, Tuple, Dict

JAVA_HOME = os.path.expanduser("~/.jdks/jdk-21.0.5+11")
os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["JVM_PATH"] = os.path.join(JAVA_HOME, "lib", "server", "libjvm.so")
os.environ["LD_LIBRARY_PATH"] = os.path.join(JAVA_HOME, "lib", "server") + ":" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["PATH"] = os.path.join(JAVA_HOME, "bin") + ":" + os.environ.get("PATH", "")

from pyserini.search.lucene import LuceneSearcher  # pip install pyserini


def configure_python_path():
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    print(f"Adding {project_root} to sys.path")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

configure_python_path()

from config.config import STORAGE_DIR
from src.utils.retrieval_utils import get_legal_dataset, get_legal_queries


# utility to load qid lists
def load_qids(path):
    df = pd.read_csv(path, header=None, usecols=[0], dtype={0: str})
    return set(df[0].tolist())


def inspect_bm25_hits(index_dir: str, query: str, k: int = 10, language: str = "es"):
    """
    Utility to print the top-k BM25 hits for a query, to sanity check indexing and retrieval.

    Args:
        index_dir (str): Path to the Pyserini Lucene index.
        query (str): Query string (in Spanish in your case).
        k (int): Number of top documents to show (default=10).
        language (str): Analyzer language to use (default="es").
    """
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(index_dir)
    # if language:
    #     searcher.set_language(language)
    # Optionally configure BM25 params
    searcher.set_bm25(k1=0.9, b=0.4)

    hits = searcher.search(query, k)
    print(f"\n🔎 Query: {query}")
    print(f"Top-{k} BM25 hits:")
    for i, h in enumerate(hits, 1):
        print(f"{i:2d}. {h.docid}\tScore={h.score:.4f}")
        # If you want to peek into the raw text:
        # raw = searcher.doc(h.docid).raw
        # print(raw[:200], "...")


def _ensure_pyserini_index_from_json(
    corpus_path: str,
    index_dir: str,
    tmp_dir: str,
    threads: int = 8,
    rebuild: bool = False,
) -> None:
    import json, os, subprocess
    from pathlib import Path

    index_dir = str(index_dir)
    if os.path.isdir(index_dir) and not rebuild and any(Path(index_dir).glob("**/segments_*")):
        return

    Path(index_dir).mkdir(parents=True, exist_ok=True)
    tmp_in = Path(tmp_dir)
    tmp_in.mkdir(parents=True, exist_ok=True)
    jsonl_path = tmp_in / "docs.jsonl"

    doc_ids, docs = get_legal_dataset(corpus_path)

    # write Pyserini-friendly JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f_out:
        for did, text in zip(doc_ids, docs):
            rec = {"id": str(did), "contents": text if text is not None else ""}
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # build Lucene index
    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(tmp_in),
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(threads),
        "--storePositions", "--storeDocvectors", "--storeRaw",
        "--language", "es",
    ]
    subprocess.run(cmd, check=True)


def _load_qrels(qrels_path: str, pos_labels: List[int], neg_labels: List[int]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    df_q = pd.read_csv(qrels_path, sep="\t", header=None, names=["qid", "run", "doc_id", "label"])
    df_q["qid"] = df_q["qid"].astype(str)
    df_q["doc_id"] = df_q["doc_id"].astype(str)
    positives = df_q[df_q.label.isin(pos_labels)].groupby("qid")["doc_id"].apply(list).to_dict()
    annotated_negs = df_q[df_q.label.isin(neg_labels)].groupby("qid")["doc_id"].apply(list).to_dict()
    return positives, annotated_negs


def _load_queries(queries_path: str) -> Dict[str, str]:
    df = pd.read_csv(queries_path, sep="\t")
    # expects columns ['id', 'query']
    df["id"] = df["id"].astype(str)
    return dict(zip(df["id"], df["query"]))


def build_ce_dataset_fast(
    qrels_path: str,
    pos_labels: List[int],
    neg_labels: List[int],
    corpus_path: str,
    queries_path: str,
    output_path: str,
    qid_filter: Set[str] | None = None,
    neg_ratio: int = 12,
    med_cap: int = 3,
    med_offset: int = 10,
    med_top_k: int = 150,
    seed: int = 42,
    # Pyserini-specific
    index_dir: str = "indexes/legal_bm25",
    tmp_dir: str = "tmp/pyserini_jsonl",
    rebuild_index: bool = False,
    bm25_k1: float = 0.9,
    bm25_b: float = 0.4,
    threads: int = 8,
    easy_mode: str = "no_terms",  # 'no_terms' (attempt Lucene -*), or 'random'
) -> pd.DataFrame:
    """
    Drop-in faster version backed by Pyserini (Lucene).
    - Builds (or reuses) a Lucene index for the corpus.
    - Uses LuceneSearcher for BM25 top-k retrieval to get 'medium' negatives.
    - 'easy' negatives: either sample docs that match *:* but NOT any query term,
      or fall back to random sampling if the parser path isn't available.

    Notes:
      * Index build follows Pyserini's JsonCollection + DefaultLuceneDocumentGenerator pattern. :contentReference[oaicite:2]{index=2}
      * Retrieval uses LuceneSearcher.set_bm25() and .search(). :contentReference[oaicite:3]{index=3}
    """
    random.seed(seed)

    doc_ids, docs = get_legal_dataset(corpus_path)
    doc_dict = {did: doc for did, doc in zip(doc_ids, docs)}

    # 0) Load supervision and queries
    positives, annotated_negs = _load_qrels(qrels_path, pos_labels, neg_labels)
    query_dict = _load_queries(queries_path)
    if qid_filter:
        query_dict = {qid: q for qid, q in query_dict.items() if qid in qid_filter}

    # 1) Ensure Lucene index
    _ensure_pyserini_index_from_json(
        corpus_path=corpus_path,
        index_dir=index_dir,
        tmp_dir=tmp_dir,
        threads=threads,
        rebuild=rebuild_index,
    )

    # 2) Open searcher
    searcher = LuceneSearcher(index_dir)
    searcher.set_language("es")
    searcher.set_bm25(bm25_k1, bm25_b)  # typical defaults; adjust if desired. :contentReference[oaicite:4]{index=4}

    # Build a universe of all docids quickly from index (iterate once)
    # Pyserini exposes raw stored docs via searcher.doc(docid).docid lookup is int->ext id via internal mappings;
    # we'll get ext ids from a *:* query as a simple method.
    # To avoid pulling millions, we lazily collect as we see hits + fallbacks.
    all_docs_cache: Set[str] = set()

    def _gather_docids_from_hits(hits):
        ids = [h.docid for h in hits]
        all_docs_cache.update(ids)
        return ids

    # Helper: get top hits for a text query
    def _top_docids(q: str, k: int) -> List[str]:
        if k <= 0:
            return []
        hits = searcher.search(q, k)
        return _gather_docids_from_hits(hits)

    # Helper: sample random docids (fallback). We approximate the universe by
    # first issuing a broad query if cache is small.
    def _random_docids(exclude: Set[str], k: int) -> List[str]:
        if k <= 0:
            return []
        need = k
        while len(all_docs_cache) < max(10000, k * 5):
            # Expand cache by pulling more docs from a broad query
            more = _top_docids("*:*", 20000 - len(all_docs_cache))  # MatchAllDocsQuery via Lucene syntax. :contentReference[oaicite:5]{index=5}
            if not more:
                break
        pool = list(all_docs_cache.difference(exclude))
        if not pool:
            # last resort: pull some hits and retry
            pool = _top_docids("*:*", max(10000, k * 5))
            pool = list(set(pool).difference(exclude))
        take = min(need, len(pool))
        return random.sample(pool, take) if take > 0 else []

    rows = []
    n_annotated = n_medium = n_easy = 0
    total_pos = 0

    for qid, q_text in tqdm(query_dict.items(), desc="Processing queries (Pyserini)"):
        pos_list = positives.get(qid, [])
        if not pos_list:
            continue
        pos_set = set(pos_list)
        total_pos += len(pos_list)

        # total negatives desired
        desired_neg = neg_ratio * len(pos_list)
        if desired_neg == 0:
            desired_neg = len(annotated_negs.get(qid, []))

        # hard (annotated)
        hard_pool = annotated_negs.get(qid, [])
        if len(hard_pool) >= desired_neg:
            hard_used = random.sample(hard_pool, desired_neg)
            med_used, easy_used = [], []
        else:
            hard_used = list(hard_pool)
            rem = desired_neg - len(hard_used)

            # MEDIUM via BM25 top-k with offset
            K = med_offset + med_top_k + 50  # small cushion
            top_docids = _top_docids(q_text, K)
            med_candidates = [
                d for d in top_docids[med_offset: med_offset + med_top_k]
                if d not in pos_set and d not in hard_used
            ]
            m_needed = min(med_cap * len(pos_list), rem)
            med_used = med_candidates[:m_needed]
            rem -= len(med_used)

            # EASY: docs that avoid query terms (if possible), else random
            easy_used = []
            if rem > 0:
                if easy_mode == "no_terms":
                    # Try Lucene query parser: "*:* -t1 -t2 ..."
                    # Lucene QueryParser supports NOT via '-' and match-all via '*:*'. :contentReference[oaicite:6]{index=6}
                    q_terms = [t for t in q_text.lower().split() if len(t) > 3]
                    if q_terms:
                        try:
                            # sample randomly from doc_dict
                            # to get a random sample of candidate docs for the easy negatives
                            # that will then be filtered to not include any query terms
                            cand = random.sample(doc_ids, rem*100)

                            # filter out overlaps
                            cand = [
                                d for d in cand
                                if d not in pos_set and d not in hard_used and d not in med_used
                                and all(t not in doc_dict[d].lower() for t in q_terms)
                            ]
                            if len(cand) == 0:
                                print("Warning: Did not find any easy candidates.")
                            take = min(rem, len(cand))
                            easy_used = random.sample(cand, take) if take > 0 else []
                            rem -= len(easy_used)
                        except Exception:
                            # Fall back to random if the query parser route is unavailable
                            pass

                if rem > 0:
                    exclude = pos_set.union(hard_used).union(med_used).union(set(easy_used))
                    extra = _random_docids(exclude, rem)
                    easy_used.extend(extra)
                    rem -= len(extra)
                    if rem > 0:
                        print(f"[WARN] qid {qid}: short {rem} negatives after fallback")

        # bookkeeping
        n_annotated += len(hard_used)
        n_medium += len(med_used)
        n_easy += len(easy_used)

        # write rows
        rows.extend((qid, d, 1) for d in pos_list)
        rows.extend((qid, d, 0) for d in (hard_used + med_used + easy_used))

    df_out = pd.DataFrame(rows, columns=["qid", "doc_id", "label"])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, sep="\t", index=False)

    print(f"[✓] Written {len(df_out)} rows to {output_path}")
    print(f"Negatives breakdown: annotated={n_annotated}, medium={n_medium}, easy={n_easy}")
    print(f"Total positives={total_pos}")
    return df_out



def chunk_text(
    text: str,
    max_length: int,
    stride: int = None
) -> List[str]:
    """
    Split text into overlapping chunks of at most `max_length` tokens.

    If `stride` is not provided, defaults to 50% overlap (max_length // 2).

    Parameters
    ----------
    text : str
        The full document text.
    max_length : int
        Maximum number of tokens per chunk.
    stride : int, optional
        Number of tokens to advance for each new chunk.

    Returns
    -------
    List[str]
        List of text chunks.
    """
    tokens = text.split()
    if stride is None:
        stride = max_length // 2
    chunks = [
        " ".join(tokens[i : i + max_length])
        for i in range(0, len(tokens), stride)
        if tokens[i : i + max_length]
    ]
    return chunks


def build_ce_dataset_chunked(
    qrels_path: str,
    pos_labels: List[int],
    neg_labels: List[int],
    corpus_path: str,
    queries_path: str,
    output_path: str,
    max_length: int,
    stride: int = None,
    qid_filter: Set[str] = None,
    neg_ratio: int = 6,
    med_cap: int = 3,
    med_offset: int = 10,
    med_top_k: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a binary-label cross-encoder dataset with overlapping chunking.

    Documents are split into chunks of up to `max_length` tokens with
    a sliding window of `stride` tokens (default 50% overlap).

    Query and document IDs are consistently handled as strings.

    Parameters
    ----------
    qrels_path : str
        Path to TSV with columns [qid, run, doc_id, label].
    pos_labels : list of int
        Labels treated as positives.
    neg_labels : list of int
        Labels treated as annotated negatives.
    corpus_path : str
        Path to JSON mapping doc_id to document text.
    queries_path : str
        Path to CSV with ['topic_id', 'Query'] columns.
    output_path : str
        Destination TSV file (qid, chunk_id, label).
    max_length : int
        Maximum token length per document chunk.
    stride : int, optional
        Tokens to slide between chunks. Defaults to max_length // 2.
    qid_filter : set of str, optional
        If provided, only include these query IDs.
    neg_ratio : int, optional
        Number of negatives per positive (default 6).
    med_cap : int, optional
        Max BM25 negatives per positive (default 3).
    med_offset : int, optional
        BM25 ranking offset (default 10).
    med_top_k : int, optional
        BM25 top-k window size (default 150).
    seed : int, optional
        Random seed (default 42).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['qid', 'chunk_id', 'label'].
    """
    random.seed(seed)

    # 1) Load relevance judgments as strings
    df_q = pd.read_csv(
        qrels_path, sep="\t", header=None,
        names=["qid", "run", "doc_id", "label"]
    )
    df_q["qid"] = df_q["qid"].astype(str)
    df_q["doc_id"] = df_q["doc_id"].astype(str)

    positives = (
        df_q[df_q.label.isin(pos_labels)]
        .groupby("qid")["doc_id"].apply(list).to_dict()
    )
    annotated_negs = (
        df_q[df_q.label.isin(neg_labels)]
        .groupby("qid")["doc_id"].apply(list).to_dict()
    )

    # 2) Load and chunk corpus with doc_id as string
    raw_ids, docs = get_legal_dataset(corpus_path)
    corpus = {str(did): text for did, text in zip(raw_ids, docs)}

    chunk_map: Dict[str, str] = {}
    chunk_texts: Dict[str, str] = {}

    for doc_id, text in tqdm(corpus.items(), desc="Chunking docs"):
        for idx, chunk in enumerate(chunk_text(text, max_length, stride)):
            cid = f"{doc_id}__chunk{idx}"
            chunk_map[cid] = doc_id
            chunk_texts[cid] = chunk

    chunk_ids = list(chunk_texts.keys())
    tokenized_chunks = [chunk_texts[cid].split() for cid in chunk_ids]
    bm25 = BM25Okapi(tokenized_chunks)

    # 3) Load queries with qid as string
    raw_qids, queries = get_legal_queries(queries_path)
    qids = [str(q) for q in raw_qids]
    query_dict = dict(zip(qids, queries))

    rows = []
    n_annotated = n_medium = n_easy = 0

    for qid, q_text in tqdm(query_dict.items(), desc="Processing queries"):
        if qid_filter and qid not in qid_filter:
            continue

        pos_docs = positives.get(qid, [])
        if not pos_docs:
            continue

        pos_chunk_ids = [
            cid for cid, did in chunk_map.items() if did in pos_docs
        ]

        desired_neg = neg_ratio * len(pos_chunk_ids)
        if desired_neg == 0:
            desired_neg = len(annotated_negs.get(qid, []))

        hard_docs = annotated_negs.get(qid, [])
        hard_chunk_ids = [
            cid for cid, did in chunk_map.items() if did in hard_docs
        ]

        if len(hard_chunk_ids) >= desired_neg:
            hard_used = random.sample(hard_chunk_ids, desired_neg)
            med_used = []
            easy_used = []
        else:
            hard_used = hard_chunk_ids.copy()
            rem = desired_neg - len(hard_used)

            # Medium negatives via BM25
            tok_q = q_text.split()
            scores = bm25.get_scores(tok_q)
            ranked_idx = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )
            med_candidates = [
                chunk_ids[i] for i in ranked_idx[med_offset : med_offset + med_top_k]
                if chunk_map[chunk_ids[i]] not in pos_docs
                and chunk_ids[i] not in hard_used
            ]
            m_needed = min(med_cap * len(pos_chunk_ids), rem)
            med_used = med_candidates[:m_needed]
            rem -= len(med_used)

            # Easy negatives by query-term exclusion
            q_terms = {t for t in tok_q if len(t) > 3}
            easy_candidates = [
                cid for cid in chunk_ids
                if chunk_map[cid] not in pos_docs
                and cid not in hard_used
                and cid not in med_used
                and all(t.lower() not in chunk_texts[cid].lower() for t in q_terms)
            ]
            take = min(rem, len(easy_candidates))
            easy_used = random.sample(easy_candidates, take) if take > 0 else []
            rem -= len(easy_used)

            # Fallback negatives
            if rem > 0:
                fallback = [
                    cid for cid in chunk_ids
                    if chunk_map[cid] not in pos_docs
                    and cid not in hard_used
                    and cid not in med_used
                    and cid not in easy_used
                ]
                fb_take = min(rem, len(fallback))
                fb = random.sample(fallback, fb_take) if fb_take > 0 else []
                easy_used.extend(fb)
                rem -= len(fb)
                if rem > 0:
                    print(f"[WARN] qid {qid}: short {rem} negatives")

        n_annotated += len(hard_used)
        n_medium += len(med_used)
        n_easy += len(easy_used)

        rows.extend((qid, cid, 1) for cid in pos_chunk_ids)
        rows.extend((qid, cid, 0) for cid in hard_used + med_used + easy_used)

    # 4) Save output
    df_out = pd.DataFrame(rows, columns=["qid", "chunk_id", "label"])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, sep="\t", index=False)

    print(f"[✓] Written {len(df_out)} rows to {output_path}")
    print(
        f"Negatives breakdown: annotated={n_annotated}, "
        f"medium={n_medium}, easy={n_easy}"
    )
    print(f"Total positives={len(df_out[df_out.label == 1])}")

    return df_out


def build_ce_dataset(
    qrels_path: str,
    pos_labels: list,
    neg_labels: list,
    corpus_path: str,
    queries_path: str,
    output_path: str,
    qid_filter: set = None,
    neg_ratio: int = 6,
    med_cap: int = 3,
    med_offset: int = 10,
    med_top_k: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a binary‑label cross‑encoder dataset where `neg_ratio` controls the total
    number of negatives per positive, including both annotated and mined negatives.

    Parameters
    ----------
    qrels_path : str
        Path to TSV with columns [qid, run, doc_id, label].
    corpus_path : str
        Path to JSON mapping doc_id -> full document text.
    queries_path : str
        Path to CSV with columns ['topic_id', 'Query'].
    output_path : str
        Destination TSV file (qid, doc_id, label).
    qid_filter : set, optional
        If provided, only include queries in this set.
    neg_ratio : int, optional
        Total number of negatives (annotated + mined) per positive (default: 6).
    med_cap : int, optional
        Maximum BM25‑mined negatives per positive (default: 3).
    med_top_k : int, optional
        How many top BM25 hits to consider for medium negatives (default: 150).
    seed : int, optional
        RNG seed for reproducibility (default: 42).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['qid', 'doc_id', 'label'].
    """
    random.seed(seed)

    # Counters for negative types
    n_annotated = 0
    n_medium = 0
    n_easy = 0

    # 1) Load relevance judgments
    df_q = pd.read_csv(
        qrels_path, sep="\t", header=None,
        names=["qid", "run", "doc_id", "label"]
    )
    df_q["qid"] = df_q["qid"].astype(str)
    df_q["doc_id"] = df_q["doc_id"].astype(str)

    positives = (
        df_q[df_q.label.isin(pos_labels)]
        .groupby("qid")["doc_id"]
        .apply(list)
        .to_dict()
    )
    annotated_negs = (
        df_q[df_q.label.isin(neg_labels)]
        .groupby("qid")["doc_id"]
        .apply(list)
        .to_dict()
    )

    # 2) Load corpus and build BM25 index
    doc_ids, docs = get_legal_dataset(corpus_path)
    # corpus is a mapping of doc_id -> full document text
    corpus = dict(zip(doc_ids, docs))
    
    tokenized_docs = [corpus[d].split() for d in tqdm(doc_ids, desc="Tokenizing corpus")]
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_docs)
    print("BM25 index built.")

    # 3) Load queries
    qids, queries = get_legal_queries(queries_path)
    query_dict = dict(zip(qids, queries))

    rows = []

    for qid, q_text in tqdm(query_dict.items(), desc="Processing queries"):
        if qid_filter and qid not in qid_filter:
            continue
        pos_list = positives.get(qid, [])
        if not pos_list:
            continue

        # Determine total negatives needed
        desired_neg = neg_ratio * len(pos_list)
        if desired_neg == 0:
            # Take only annotated negatives when desired_neg is 0
            desired_neg = len(annotated_negs.get(qid, []))

        # Fetch all annotated (hard) negatives
        hard_pool = annotated_negs.get(qid, [])
        if len(hard_pool) >= desired_neg:
            # If there are more annotated negatives than needed, sample them
            hard_used = random.sample(hard_pool, desired_neg)
            med_used = []
            easy_used = []
        else:
            # Otherwise take all annotated and mine the rest
            hard_used = hard_pool
            rem = desired_neg - len(hard_used)

            # 3a) Mine medium negatives via BM25
            tok_q = q_text.split()
            scores = bm25.get_scores(tok_q)
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            med_candidates = [
                doc_ids[i]
                for i in ranked[med_offset:med_offset + med_top_k]
                if doc_ids[i] not in pos_list and doc_ids[i] not in hard_used
            ]
            m_needed = min(med_cap * len(pos_list), rem)
            med_used = med_candidates[:m_needed]
            rem -= len(med_used)

            # 3b) Mine easy negatives by excluding query terms
            q_terms = {t for t in tok_q if len(t) > 3}
            easy_candidates = [
                d for d in doc_ids
                if d not in pos_list
                and d not in hard_used
                and d not in med_used
                and all(t.lower() not in corpus[d].lower() for t in q_terms)
            ]
            take = min(rem, len(easy_candidates))
            easy_used = random.sample(easy_candidates, take) if take > 0 else []
            rem -= len(easy_used)

            # 3c) Fallback to any remaining docs if still short
            if rem > 0:
                fallback = [
                    d for d in doc_ids
                    if d not in pos_list
                    and d not in hard_used
                    and d not in med_used
                    and d not in easy_used
                ]
                fb_take = min(rem, len(fallback))
                fb = random.sample(fallback, fb_take) if fb_take > 0 else []
                easy_used.extend(fb)
                rem -= len(fb)
                if rem > 0:
                    print(f"[WARN] qid {qid}: short {rem} negatives after fallback")

        # Update counters
        n_annotated += len(hard_used)
        n_medium += len(med_used)
        n_easy += len(easy_used)

        # Append positives and negatives to rows
        rows.extend((qid, d, 1) for d in pos_list)
        rows.extend((qid, d, 0) for d in hard_used + med_used + easy_used)

    # 4) Create DataFrame and save
    df_out = pd.DataFrame(rows, columns=["qid", "doc_id", "label"])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, sep="\t", index=False)
    print(f"[✓] Written {len(df_out)} rows to {output_path}")
    print(f"Negatives breakdown: annotated={n_annotated}, medium={n_medium}, easy={n_easy}")
    print(f"Total positives={len(df_out[df_out.label == 1])}")

    return df_out


def main_create_scenario_datasets():
    """
    Generate BCE cross‑encoder TSVs for four scenarios:
      S1: random split, only annotated negatives
      S2: random split, annotated + 6× extra negatives
      S3: query‑wise split, only annotated negatives
      S4: query‑wise split, annotated + extras at 2× and 3×
    """
    base = Path(STORAGE_DIR) / "legal_ir" / "data"
    # ann  = base / "annotations" / "qrels_54.tsv"
    ann  = base / "annotations" / "mistral_inpars_v2_corpus_NEW_qrels_dedup.tsv"
    # ann  = base / "annotations" / "inpars_mistral-small-2501_qrels_Q1.tsv"
    # corp = base / "corpus" / "corpus.jsonl"
    corp = base / "corpus" / "corpus_NEW.jsonl"
    # qry  = base / "corpus" / "queries_54.tsv"
    qry  = base / "corpus" / "mistral_inpars_v2_corpus_NEW_queries_dedup.tsv"
    out  = base / "datasets" / "dual_encoder"
    out.mkdir(parents=True, exist_ok=True)

    seed = 42
    
    # train_qids = load_qids(base / "qids_train.txt")
    # dev_qids   = load_qids(base / "qids_dev.txt")
    # test_qids  = load_qids(base / "qids_test.txt")
    train_qids = load_qids(base / "qids_inpars_v2_dedup_train.txt")
    dev_qids   = load_qids(base / "qids_inpars_v2_dedup_dev.txt")
    test_qids  = load_qids(base / "qids_inpars_v2_dedup_test.txt")
    for split_name, qids in [("train", train_qids), ("dev", dev_qids), ("test", test_qids)]:
        # build_ce_dataset(
        #     qrels_path=str(ann),
        #     pos_labels=[1],
        #     neg_labels=[0],
        #     corpus_path=str(corp),
        #     queries_path=str(qry),
        #     output_path=str(out / f"bce_6x_synthetic_{split_name}.tsv"),
        #     qid_filter=qids,
        #     neg_ratio=6,
        #     seed=seed
        # )
        # build_ce_dataset(
        #     qrels_path=str(ann),
        #     pos_labels=[1],
        #     neg_labels=[0],
        #     corpus_path=str(corp),
        #     queries_path=str(qry),
        #     output_path=str(out / f"bge_finetune_6x_inpars_v2_corta_{split_name}.tsv"),
        #     qid_filter=qids,
        #     neg_ratio=6,
        #     seed=seed,
        #     # max_length=512,
        #     # stride=256,
        # )
        build_ce_dataset_fast(
            qrels_path=str(ann),
            pos_labels=[1],
            neg_labels=[0],
            corpus_path=str(corp),
            queries_path=str(qry),
            output_path=str(out / f"bge_finetune_12x_inpars_v2_dedup_{split_name}.tsv"),
            qid_filter=qids,
            neg_ratio=12,
            seed=seed,
            # max_length=512,
            # stride=256,
        )

if __name__ == "__main__":
    main_create_scenario_datasets()
    # inspect_bm25_hits("indexes/legal_bm25", "acción de inconstitucionalidad ley 1626", k=10)

