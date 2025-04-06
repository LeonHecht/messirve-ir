import torch
import pandas as pd
import os
import ir_datasets
import pickle
import json
from datasets import load_dataset
import src.scripts.msmarco_eval_ranking as msmarco_eval_ranking
from sentence_transformers import SentenceTransformer
from config.config import STORAGE_DIR
from utils.retrieval_utils import (
    embed_bge,
    embed_jina,
    embed_jina_faiss,
    embed_mamba,
    retrieve_bm25,
    embed_s_transformers,
    embed_s_transformers_faiss,
    rerank_cross_encoder,
    embed_qwen,
    get_eval_metrics,
    create_results_file,
    get_legal_dataset,
    get_legal_queries,
    create_predictions_file,
    embed_chunkwise,
    get_sim_bge,
    chunk_by_paragraphs,
)

from utils.train_utils import (
    get_msmarco_queries,
    get_msmarco_passages
)

from models.model_setup import get_bge_m3_model, get_jinja_model, get_mamba_model


class Evaluator:
    """
        Initialize the evaluator with required parameters.

        Parameters
        ----------
        ds : str
            The dataset to use (e.g., 'msmarco', 'legal').
        model_name : str
            The model identifier (e.g., 'bm25', 'bge', 'jinja', etc.).
        metric_names : set
            A set of metric names for evaluation.
        limit : int
            The maximum number of documents to process.
        reuse_run : bool
            Flag to determine if a cached run should be reused.
        model_instance : object
            The model instance for embedding (if applicable).
        rerank : bool
            Whether to perform reranking.
        tokenizer : object, optional
            Tokenizer instance for models that require it.
        reranker_model : object, optional
            Reranker model instance.
    """
    def __init__(self, ds, model_name, metric_names, limit=None, reuse_run=False, model_instance=None, rerank=False, tokenizer=None, reranker_model=None):
        self.ds = ds
        self.model_name = model_name
        self.metric_names = metric_names
        self.limit = limit
        self.reuse_run = reuse_run
        self.model_instance = model_instance
        self.rerank = rerank
        self.tokenizer = tokenizer
        self.reranker_model = reranker_model

        self.device = torch.device("cuda:1")
        self.docs = None
        self.doc_ids = None
        self.queries = None
        self.query_ids = None
        self.qrels_dev_df = None
        self.run = None
        self.path_to_reference_qrels = None

    def load_data(self):
        if self.ds == "msmarco":
            qid_to_query = get_msmarco_queries()
            pid_to_passage = get_msmarco_passages()

            if self.limit:
                self.docs = list(pid_to_passage.values())[:self.limit]
                self.doc_ids = list(pid_to_passage.keys())[:self.limit]
            else:
                self.docs = list(pid_to_passage.values())
                self.doc_ids = list(pid_to_passage.keys())
            
            qrels_dev_path = os.path.join(STORAGE_DIR, "ms_marco_passage", "data", "qrels.dev.tsv")
            self.qrels_dev_df = pd.read_csv(
                qrels_dev_path,
                sep="\t",                # TREC qrels are usually tab-separated
                names=["query_id", "iteration", "doc_id", "relevance"],
                header=None,            # There's no header in qrels files
                dtype={"query_id": int, "iteration": int, "doc_id": int, "relevance": int}
            )

            dataset_dev_small = ir_datasets.load("msmarco-passage/dev/small")
            qids_dev_small = []
            for qid, query in dataset_dev_small.queries_iter():
                qids_dev_small.append(int(qid))
            
            print("Length of qids_dev_small:", len(qids_dev_small))

            # filter qrels to only include doc_ids that are in the top limit docs
            self.qrels_dev_df = self.qrels_dev_df[self.qrels_dev_df["query_id"].isin(qids_dev_small)]
            self.qrels_dev_df = self.qrels_dev_df[self.qrels_dev_df["doc_id"].isin(self.doc_ids)]

            # save qrels_dev_df to tsv file
            self.path_to_reference_qrels = f"qrels_dev_full.tsv"
            self.qrels_dev_df.to_csv(self.path_to_reference_qrels, sep="\t", index=False, header=False)

            with open(os.path.join('data', 'external', 'BatchAPI_outputs', 'gpt_responses_dev_small.json'), 'r', encoding='utf-8') as f:
                qid_to_response = json.load(f)
            
            # filter qid_to_response to the ones present in qrels_dev_df
            qid_to_response = {qid: response for qid, response in qid_to_response.items() if int(qid) in self.qrels_dev_df["query_id"].unique()}

            self.query_ids = list(qid_to_response.keys())
            self.queries = list(qid_to_response.values())

            # query_ids = qrels_dev_df["query_id"].unique()
            # queries = [qid_to_query[qid] for qid in query_ids]

            print("Number of queries:", len(self.queries))
            print("Number of docs:", len(self.docs))
            # rel_doc_ids = qrels_dev_df["doc_id"].unique()
        elif self.ds == "messirve":
            self.docs, self.queries, self.doc_ids, self.query_ids = get_messirve_corpus("ar")
        elif self.ds == "legal":
            self.doc_ids, self.docs = get_legal_dataset(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus_py.csv"))
            self.query_ids, self.queries = get_legal_queries(os.path.join(STORAGE_DIR, "legal_ir", "data", "queries_57.csv"))

            qrels_dev_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "qrels_py.tsv")
            self.qrels_dev_df = pd.read_csv(
                qrels_dev_path,
                sep="\t",                # TREC qrels are usually tab-separated
                names=["query_id", "iteration", "doc_id", "relevance"],
                header=None,            # There's no header in qrels files
                dtype={"query_id": int, "iteration": int, "doc_id": int, "relevance": int}
            )
            
            self.path_to_reference_qrels = os.path.join(STORAGE_DIR, "legal_ir", "data", "qrels_py.tsv")
        else:
            raise ValueError("Dataset not supported.")
        print("Data prepared.")

    def _process_run(self, run_path, compute_fn):
        """
        Checks if the run file exists; if not, computes the run using the provided function,
        then pickles the result.

        Parameters
        ----------
        run_path : str
            File path for storing/loading the run.
        compute_fn : callable
            Function that computes the run.
        """
        if not os.path.exists(run_path) or not self.reuse_run:
            self.run = compute_fn()
            with open(run_path, "wb") as f:
                pickle.dump(self.run, f)
            print(f"Run computed and saved to {run_path}")
        else:
            with open(run_path, "rb") as f:
                self.run = pickle.load(f)
            print(f"Run loaded from {run_path}")

    def get_run(self):
        """
        Retrieves or computes the run based on the model name.
        """
        # Mapping of model names to their respective computation lambdas.
        model_mapping = {
            "bm25": lambda: retrieve_bm25(self.docs, self.queries, self.doc_ids, self.query_ids),
            "bge": lambda: embed_bge(
                get_bge_m3_model('BAAI/bge-m3'),
                self.docs, self.queries, self.doc_ids, self.query_ids, self.reuse_run
            ),
            "jina": lambda: embed_jina(
                get_jinja_model().to(self.device),
                self.docs, self.queries, self.doc_ids, self.query_ids
            ),
            "mamba": lambda: embed_mamba(
                *get_mamba_model(), self.docs, self.queries, self.doc_ids, self.query_ids
            ),
            "bge-finetuned": lambda: embed_bge(
                get_bge_m3_model('test_encoder_only_m3_bge-m3_sd'),
                self.docs, self.queries, self.doc_ids, self.query_ids, self.reuse_run
            ),
            "sentence-transformer": lambda: embed_s_transformers(
                self.model_instance, self.docs, self.queries, self.doc_ids, self.query_ids
            ),
            "qwen": lambda: embed_qwen(
                self.model_instance, self.tokenizer, self.docs, self.queries, self.doc_ids, self.query_ids
            ),
            "bge-chunkwise": lambda: embed_chunkwise(get_bge_m3_model('BAAI/bge-m3'), get_sim_bge, self.docs, self.queries, self.doc_ids, self.query_ids, chunk_func=chunk_by_paragraphs, window_size=256)
        }

        if self.model_name not in model_mapping:
            raise ValueError("Model not supported.")

        # Example: using 'ar' as a placeholder for country code.
        run_path = f"run_{self.model_name}_ar.pkl"
        self._process_run(run_path, model_mapping[self.model_name])

    def rerank_run(self):
        # for each query rerank the top 1000 docs
        self.run = rerank_cross_encoder(self.reranker_model, self.tokenizer, self.run, 100, self.queries, self.query_ids, self.docs, self.doc_ids,
                                max_length=2048)

    def get_metrics(self):
        self.metrics = get_eval_metrics(self.run, self.qrels_dev_df, self.doc_ids, self.metric_names)
        create_results_file(self.run)
        create_predictions_file(self.run)   # create TREC style qrel file (contains same info as results.txt)
        msmarco_eval_ranking.main(self.path_to_reference_qrels, "results.txt")

    def evaluate(self):
        self.load_data()
        self.get_run()
        if self.rerank:
            self.rerank_run()
        self.get_metrics()


def get_messirve_corpus(country):
    ds = load_dataset("spanish-ir/messirve", country)
    docs = ds["test"]["docid_text"]
    queries = ds["test"]["query"]
    doc_ids = ds["test"]["docid"]
    query_ids = ds["test"]["id"]
    return docs, queries, doc_ids, query_ids


def main():
    from unsloth import FastLanguageModel
    from peft import AutoPeftModelForCausalLM
    
    model_save_path = "/media/discoexterno/leon/ms_marco_passage/results/IR_unsloth_qwen0.5_5negs_rslora_100k/saved_model"

    model, tokenizer = FastLanguageModel.from_pretrained(
        # model_save_path,
        model_save_path,
        max_seq_length = 150,
        dtype = "bf16",
        load_in_4bit = False
    )

    FastLanguageModel.for_inference(model)

    print(tokenizer.pad_token_id)   # 128004
    print(tokenizer.pad_token)      # <|finetune_right_pad_id|>

    # model = AutoPeftModelForCausalLM.from_pretrained(
    #     model_save_path,
    #     torch_dtype=torch.bfloat16
    # )
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    # # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    # if tokenizer.pad_token_id != tokenizer_unsloth.pad_token_id:
    #     tokenizer.add_special_tokens({"pad_token": tokenizer_unsloth.pad_token})
    #     tokenizer.pad_token_id = tokenizer_unsloth.pad_token_id
    #     model.resize_token_embeddings(len(tokenizer))
    # run("qwen", metrics={'ndcg', 'ndcg_cut.10', 'recall_1000', 'recall_100', 'recall_10', 'recip_rank'}, ds="msmarco", model_instance=model, tokenizer=tokenizer, reuse_run=False)
    evaluator = Evaluator(ds="msmarco",
                          model_name="jina",
                          metric_names={'ndcg', 'ndcg_cut.10', 'recall_1000', 'recall_100', 'recall_10', 'recip_rank'},
                          model_instance=model,
                          tokenizer=tokenizer,
                          limit=10_000
                )
    evaluator.evaluate()

if __name__ == "__main__":
    # reranker_model = AutoModelForSequenceClassification.from_pretrained("/media/discoexterno/leon/legal_ir/results/cross_encoder_2048/checkpoint-320")
    # tokenizer = AutoTokenizer.from_pretrained("/media/discoexterno/leon/legal_ir/results/cross_encoder_2048")
    # model = SentenceTransformer("multirun/2025-01-30/11-45-41/1/finetuned_models/distiluse-base-multilingual-cased-v1-exp_20250130_123521")
    # model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")
    # model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # model = SentenceTransformer("dariolopez/bge-m3-es-legal-tmp-6")
    # model = SentenceTransformer("Stern5497/sbert-legal-xlm-roberta-base")
    evaluator = Evaluator(ds="legal",
                          model_name="bge-chunkwise",
                          metric_names={'ndcg', 'ndcg_cut.10', 'recall_1000', 'recall_100', 'recall_10', 'recip_rank'},
                        #   model_instance=model
                )
    evaluator.evaluate()
    # main()
    # model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    # model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    # tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    # run(model="sentence-transformer", model_instance=model, metrics={'ndcg', 'ndcg_cut.10', 'recall_100', 'recall_10', 'recip_rank'}, country="ar", reuse_run=False, rerank=False)
    # run(model="bge-finetuned", metrics={'ndcg', 'ndcg_cut.10', 'recall_100', 'recall_10', 'recip_rank'}, country="ar")
    # run(model="jinja", metrics={'ndcg', 'ndcg_cut.10', 'recall_1000', 'recall_100', 'recall_10', 'recip_rank'}, reuse_run=False, ds="msmarco", limit=200_000)
    # run(model="mamba", metrics={'ndcg', 'ndcg_cut.10', 'recall_100', 'recip_rank'}, country="ar")
    # run("sentence-transformer", metrics={'ndcg', 'ndcg_cut.10', 'recall_1000', 'recall_100', 'recall_10', 'recip_rank'}, ds="legal", model_instance=model, reuse_run=False)
    # run("bm25", metrics={'ndcg', 'ndcg_cut.10', 'recall_100', 'recall_10', 'recip_rank'}, ds="legal", reuse_run=False, rerank=True, reranker_model=reranker_model, tokenizer=tokenizer)