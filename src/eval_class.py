import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import pandas as pd
import sys
import ir_datasets
import pickle
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def configure_python_path():
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Apply the path tweak before any project imports
configure_python_path()

import src.scripts.msmarco_eval_ranking as msmarco_eval_ranking
from sentence_transformers import SentenceTransformer
from config.config import STORAGE_DIR, MAX_QUERY_LEN

from src.utils.retrieval_utils import (
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
    get_legal_dataset_norm,
    get_legal_queries,
    create_predictions_file,
    embed_chunkwise,
    get_sim_bge,
    chunk_by_paragraphs,
    embed_bge_sparse,
    embed_bge_sliding_window,
    rerank_cross_encoder_chunked,
    retrieve_exact_match,
    evaluate_with_chunking,
    embed_qwen_chunked,
    embed_colbert,
    embed_bge_sparse_sliding_window,
    embed_qwen_sliding_window,
    embed_bge_colbert_sliding_window,
    embed_bge_sparse_sliding_window,
    embed_bge_paragraph_chunking_dense,
)

from src.utils.train_utils import (
    get_msmarco_queries,
    get_msmarco_passages
)

from src.models.model_setup import get_bge_m3_model, get_jinja_model, get_mamba_model


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
    def __init__(self, ds, model_name, metric_names, limit=None, reuse_run=False, model_instance=None, rerank=False, tokenizer=None, reranker_model=None, reranker_model_type=None, max_length=-1, rerank_chunkwise=False, checkpoint=None):
        self.ds = ds
        self.model_name = model_name
        self.metric_names = metric_names
        self.limit = limit
        self.reuse_run = reuse_run
        self.model_instance = model_instance
        self.rerank = rerank
        self.tokenizer = tokenizer
        self.reranker_model = reranker_model
        self.reranker_model_type = reranker_model_type
        self.max_length = max_length
        self.rerank_chunkwise = rerank_chunkwise
        self.checkpoint = checkpoint

        self.device = torch.device("cuda:0")
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

        elif self.ds in ("legal-54", "legal-inpars", "legal-synthetic"):
            # self.doc_ids, self.docs = get_legal_dataset(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus.jsonl"))
            self.doc_ids, self.docs = get_legal_dataset(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus_tesseract.jsonl"))
            # self.doc_ids, self.docs = get_legal_dataset(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus_mistral_summaries_1024.jsonl"))
            # self.doc_ids, self.docs = get_legal_dataset_norm(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus.jsonl"), normalize=True)
            # self.doc_ids, self.docs = get_legal_dataset(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus_raw_google_ocr.jsonl"))
            # self.doc_ids, self.docs = get_legal_dataset(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "corpus_Gpt4o-mini_cleaned.jsonl"))
            self.doc_dict = {doc_id: doc for doc_id, doc in zip(self.doc_ids, self.docs)}
            
            if self.ds == "legal-54":
                queries_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "queries_54.tsv")
                self.path_to_reference_qrels = os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "qrels_54.tsv")
                test_qids_path = None
            elif self.ds == "legal-inpars":
                queries_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "inpars_mistral-small-2501_queries.tsv")
                self.path_to_reference_qrels = os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "inpars_mistral-small-2501_qrels.tsv")
                test_qids_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "qids_inpars_test_Q1.txt")
            elif self.ds == "legal-synthetic":
                queries_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "consultas_sinteticas_380_filtered.tsv")
                self.path_to_reference_qrels = os.path.join(STORAGE_DIR, "legal_ir", "data", "annotations", "qrels_synthetic_mistral-small-2501_filtered.tsv")
                test_qids_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "qids_synthetic_test.txt")

            self.query_ids, self.queries = get_legal_queries(queries_path)

            self.qrels_dev_df = pd.read_csv(
                self.path_to_reference_qrels,
                sep="\t",                # TREC qrels are usually tab-separated
                names=["query_id", "iteration", "doc_id", "relevance"],
                header=None,            # There's no header in qrels files
                dtype={"query_id": str, "iteration": int, "doc_id": str, "relevance": int}
            )

            if test_qids_path is not None:
                with open(test_qids_path, "r") as f:
                    test_qids = f.readlines()
                test_qids = [qid.strip() for qid in test_qids]

                # filter queries and query_ids to only include the ones in the test set
                self.queries = [query for query_id, query in zip(self.query_ids, self.queries) if query_id in test_qids]
                self.query_ids = [query_id for query_id in self.query_ids if query_id in test_qids]

                # filter qrels to only include the ones in the test set
                self.qrels_dev_df = self.qrels_dev_df[self.qrels_dev_df["query_id"].isin(test_qids)]

                # Right after filtering qrels_dev_df:
                filtered_qrels_path = os.path.join(STORAGE_DIR,
                    "legal_ir", "data", "annotations", f"{self.ds}_qrels_test.tsv")
                self.qrels_dev_df.to_csv(
                    filtered_qrels_path,
                    sep="\t",
                    header=False,
                    index=False
                )
                self.path_to_reference_qrels = filtered_qrels_path

            # create a dictionary of query_id to query
            self.query_dict = {query_id: query for query_id, query in zip(self.query_ids, self.queries)}
        else:
            raise ValueError("Dataset not supported.")
        print("Data prepared.")

    def _process_run(self, compute_fn):
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
        self.run = compute_fn()
        
    def get_run(self):
        """
        Retrieves or computes the run based on the model name.
        """
        # Mapping of model names to their respective computation lambdas.
        model_mapping = {
            "bm25": lambda: retrieve_bm25(self.docs, self.queries, self.doc_ids, self.query_ids),
            "bge-sliding": lambda: embed_bge_sliding_window(
                get_bge_m3_model('BAAI/bge-m3') if self.checkpoint is None else get_bge_m3_model(self.checkpoint),
                self.doc_dict, self.query_dict, self.reuse_run
            ),
            "bge": lambda: embed_bge(
                get_bge_m3_model('BAAI/bge-m3') if self.checkpoint is None else get_bge_m3_model(self.checkpoint),
                self.docs, self.queries, self.doc_ids, self.query_ids, self.reuse_run
            ),
            "bge-sparse": lambda: embed_bge_sparse(
                get_bge_m3_model('BAAI/bge-m3') if self.checkpoint is None else get_bge_m3_model(self.checkpoint),
                self.docs, self.queries, self.doc_ids, self.query_ids, self.reuse_run
            ),
            "bge-sparse-sliding": lambda: embed_bge_sparse_sliding_window(
                get_bge_m3_model('BAAI/bge-m3') if self.checkpoint is None else get_bge_m3_model(self.checkpoint),
                self.doc_dict, self.query_dict, self.reuse_run
            ),
            "bge-colbert-sliding": lambda: embed_bge_colbert_sliding_window(
                get_bge_m3_model('BAAI/bge-m3') if self.checkpoint is None else get_bge_m3_model(self.checkpoint),
                self.doc_dict, self.query_dict, self.reuse_run
            ),
            "jina": lambda: embed_jina(
                get_jinja_model().to(self.device),
                self.docs, self.queries, self.doc_ids, self.query_ids
            ),
            "mamba": lambda: embed_mamba(
                *get_mamba_model(), self.docs, self.queries, self.doc_ids, self.query_ids
            ),
            "sentence-transformer": lambda: embed_s_transformers(
                self.model_instance, self.docs, self.queries, self.doc_ids, self.query_ids
            ),
            "qwen": lambda: embed_qwen(
                self.model_instance, self.tokenizer, self.docs, self.queries, self.doc_ids, self.query_ids
            ),
            "qwen-sliding": lambda: embed_qwen_sliding_window(
                self.model_instance, self.tokenizer, self.doc_dict, self.query_dict
            ),
            "qwen-chunkwise": lambda: embed_qwen_chunked(self.model_instance, self.tokenizer, self.doc_dict, self.query_dict, self.doc_ids, self.query_ids),
            "bge-chunkwise": lambda: embed_chunkwise(get_bge_m3_model('BAAI/bge-m3'), get_sim_bge, self.docs, self.queries, self.doc_ids, self.query_ids, chunk_func=chunk_by_paragraphs, window_size=256),
            "exact-match": lambda: retrieve_exact_match(self.docs, self.queries, self.doc_ids, self.query_ids),
            "colbert": lambda: embed_colbert(
                self.model_instance, self.docs, self.queries, self.doc_ids, self.query_ids
            ),
            "bge-sparse-sliding": lambda: embed_bge_sparse_sliding_window(
                get_bge_m3_model('BAAI/bge-m3') if self.checkpoint is None else get_bge_m3_model(self.checkpoint),
                self.doc_dict, self.query_dict, self.reuse_run
            ),
            "bge-paragraph": lambda: embed_bge_paragraph_chunking_dense(
                get_bge_m3_model('BAAI/bge-m3') if self.checkpoint is None else get_bge_m3_model(self.checkpoint),
                self.doc_dict, self.query_dict, self.reuse_run
            ),
        }

        if self.model_name not in model_mapping:
            raise ValueError("Model not supported.")

        self._process_run(model_mapping[self.model_name])

    def write_run_to_tsv(self, qid: str, out_path):
        run_Q1 = self.run[qid]
        run_Q1 = dict(sorted(run_Q1.items(), key=lambda x: x[1], reverse=True))
        with open(out_path, "w") as f:
            f.write("doc_id\tscore\n")
            for did, score in run_Q1.items():
                f.write(f"{did}\t{str(score)}\n")

    def rerank_run(self):
        # self.write_run_to_tsv(qid="1", out_path="run_Q1_before.tsv")
        # for each query rerank the top 100 docs
        if self.rerank_chunkwise:
            self.run = rerank_cross_encoder_chunked(self.reranker_model, self.reranker_model_type, self.tokenizer, self.run, 100, self.query_dict, self.doc_dict,
                                                    max_length=self.max_length, stride=self.max_length//2, aggregator="top3")
        else:
            self.run = rerank_cross_encoder(self.reranker_model, self.reranker_model_type, self.tokenizer, self.run, 50, self.query_dict, self.doc_dict,
                                    max_length=self.max_length)
        # self.write_run_to_tsv(qid="1", out_path="run_Q1_after.tsv")

    def get_metrics(self):
        run_qids = set(self.run.keys())
        ref_qids = set(self.qrels_dev_df["query_id"].astype(str).unique())
        print(f"Run queries:   {len(run_qids)}")
        print(f"QREL queries:  {len(ref_qids)}")
        print(f"Intersection:  {len(run_qids & ref_qids)}")
        self.metrics = get_eval_metrics(self.run, self.qrels_dev_df, self.doc_ids, self.metric_names)
        result_paths = create_results_file(self.run)
        prediction_paths = create_predictions_file(self.run)   # create TREC style qrel file (contains same info as results.txt)
        if type(result_paths) == str:
            msmarco_eval_ranking.main(self.path_to_reference_qrels, [result_paths])
        elif type(result_paths) == list:
            msmarco_eval_ranking.main(self.path_to_reference_qrels, result_paths)
        else:
            raise Exception("Invalid result paths")


    def create_run_df(self, run, top_k=5):
        """
        Create a DataFrame with queries as columns and rows representing the top_k retrieved documents.

        For each query in the input dictionary, the documents are sorted in descending order
        according to their scores. Only the top_k document identifiers are kept. In the returned
        DataFrame, each column corresponds to a query and each row represents the rank (e.g., Rank 1,
        Rank 2, ..., Rank top_k). If a query has fewer than top_k documents, the missing entries are
        set to None.

        Parameters
        ----------
        run : dict
            A dictionary where each key is a query identifier and each value is a dictionary of document
            identifiers mapped to their corresponding scores. Example format:
            {
                "query1": {"doc1": score1, "doc2": score2, ...},
                "query2": {"doc3": score3, "doc4": score4, ...},
                ...
            }
        top_k : int, optional
            The number of top documents to retrieve for each query (default is 5).

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame with queries as columns and rows representing the top_k document ranks.
            The index labels (e.g., 'Rank 1', 'Rank 2', etc.) indicate the rank order.

        Raises
        ------
        ValueError
            If the input run is not a dictionary, or if its values are not dictionaries.
        """
        if not isinstance(run, dict):
            raise ValueError("The run input must be a dictionary.")
        
        data = {}
        for query_id, doc_scores in run.items():
            if not isinstance(doc_scores, dict):
                raise ValueError("Each value in the run dictionary must be a dictionary of document scores.")
            # Sort documents for the given query based on their scores (higher first)
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, score in sorted_docs[:top_k]]
            # Pad with None if fewer than top_k documents exist
            if len(top_docs) < top_k:
                top_docs.extend([None] * (top_k - len(top_docs)))
            data[query_id] = top_docs

        # Create DataFrame and label the rows by rank.
        df = pd.DataFrame(data)
        df.index = [f"Rank {i + 1}" for i in range(top_k)]
        return df

    def evaluate(self):
        import pickle
        self.load_data()
        self.get_run()
        # # save run using pickle
        # with open("run_bm25.pkl", "wb") as f:
        #     pickle.dump(self.run, f)
        # self.run_df = self.create_run_df(self.run, top_k=10)
        if self.rerank:
            self.rerank_run()
        self.get_metrics()
        # with open("metrics.pkl", "wb") as f:
        #     pickle.dump(self.metrics, f)


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
    
    # model_save_path = "/media/discoexterno/leon/ms_marco_passage/results/IR_unsloth_qwen0.5_5negs_rslora_100k/saved_model"
    model_save_path = "/media/discoexterno/leon/legal_ir/results/legal_eos_full_synthetic/saved_model"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_save_path,
        max_seq_length = 512,
        dtype = torch.bfloat16,
        load_in_4bit = False
    )

    FastLanguageModel.for_inference(model)

    print("Pad token ID and token:")
    print(tokenizer.pad_token_id)   # 128004
    print(tokenizer.pad_token)      # <|finetune_right_pad_id|>

    print("EOS token ID and token:")
    print(tokenizer.eos_token_id)
    print(tokenizer.eos_token)

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
    
    # evaluator = Evaluator(ds="msmarco",
    #                       model_name="jina",
    #                       metric_names={'ndcg', 'ndcg_cut.10', 'recall_1000', 'recall_100', 'recall_10', 'recip_rank'},
    #                       model_instance=model,
    #                       tokenizer=tokenizer,
    #                       limit=10_000
    #             )
    evaluator = Evaluator(
        ds="legal-inpars",
        model_name="qwen-sliding",
        metric_names={'ndcg', 'ndcg_cut.10', 'recall_1000', 'recall_100', 'recall_10', 'recip_rank', 'map'},
        model_instance=model,
        # checkpoint=model_path,
        rerank=False,
        # reranker_model=model,
        tokenizer=tokenizer,
        # max_length=512,
        # rerank_chunkwise=True,
        # reranker_model_type="binary"
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
    
    # evaluator = Evaluator(ds="legal",
    #                       model_name="bm25",
    #                       metric_names={'ndcg', 'ndcg_cut.10', 'recall_1000', 'recall_100', 'recall_10', 'recip_rank'},
    #                     #   model_instance=model
    #             )
    # evaluator.evaluate()

    # from transformers import AutoTokenizer, AutoModelForSequenceClassification
    # reranker_model = AutoModelForSequenceClassification.from_pretrained("/media/discoexterno/leon/legal_ir/results/cross_encoder_weighted_stride_inpars_Q1_v4")
    # tokenizer = AutoTokenizer.from_pretrained("/media/discoexterno/leon/legal_ir/results/cross_encoder_weighted_stride_inpars_Q1_v4")
    
    # # model_path = "dariolopez/bge-m3-es-legal-tmp-6"
    # model_path = "/media/discoexterno/leon/legal_ir/results/baai_finetuning/bge-m3_6x_54_multi_epoch"
    # model_path = "/media/discoexterno/leon/legal_ir/results/baai_finetuning/bge-m3_6x_54_summary_1024"
    # model_path = "/media/discoexterno/leon/legal_ir/results/baai_finetuning/bge-m3_full_6x_summary_1024"
    # # model_path = "/media/discoexterno/leon/legal_ir/results/baai_finetuning/bge-m3_full_6x"
    # # model_path = "/media/discoexterno/leon/legal_ir/results/baai_finetuning/bge-m3_full_chunked_6x"
    # # model_path = "/media/discoexterno/leon/legal_ir/results/legal_eos_full_synthetic/saved_model"
    # # model_path = "/media/discoexterno/leon/legal_ir/results/cross_encoder_weighted_stride_inpars_Q1_v4/checkpoint-4276"
    # # model = AutoModelForSequenceClassification.from_pretrained(model_path)
    # # tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Evaluate IR metrics.
    evaluator = Evaluator(
        ds="legal-inpars",
        model_name="bge",
        # metric_names={'ndcg', 'ndcg_cut.10', 'recall_1000', 'recall_100', 'recall_10', 'recip_rank', 'map'},
        metric_names={'ndcg_cut.10', 'recall_100', 'recall_10'},
        # model_instance=model,
        # checkpoint=model_path,
        rerank=False,
        # reranker_model=reranker_model,
        # tokenizer=tokenizer,
        # max_length=512,
        # rerank_chunkwise=True,
        # reranker_model_type="binary"
    )
    evaluator.evaluate()

    # main()
    # model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    # model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")