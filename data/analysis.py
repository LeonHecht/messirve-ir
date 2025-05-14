import pickle
import os

from config.config import STORAGE_DIR
from src.utils.retrieval_utils import get_legal_queries


qids, queries = get_legal_queries(
    os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "consultas_sinteticas_380_filtered.tsv"),
)

query_dict = dict(zip(qids, queries))

with open("run_bm25.pkl", "rb") as f:
    run = pickle.load(f)

queries_with_no_0_docs = {}

for qid, doc_score_dict in run.items():
    scores = doc_score_dict.values()
    if all(scores) > 0:
        queries_with_no_0_docs[qid] = query_dict[qid]

with open("queries_with_no_0_docs.txt", "w") as f:
    for qid, query in queries_with_no_0_docs.items():
        f.write(f"{qid} {query}\n")

print(len(queries_with_no_0_docs))
print(len(run.keys()))