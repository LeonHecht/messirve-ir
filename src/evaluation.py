import sys
print("Executable: ", sys.executable)
from datasets import load_dataset
import pandas as pd
import pickle
import json
import os
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import src.scripts.msmarco_eval_ranking as msmarco_eval_ranking
import ir_datasets
from utils.retrieval_utils import (
    embed_bge,
    embed_jinja,
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
)

from utils.train_utils import (
    get_msmarco_queries,
    get_msmarco_passages
)
# make only GPU0 visible
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from config.config import STORAGE_DIR

from models.model_setup import get_bge_m3_model, get_jinja_model
from models import model_setup


def get_messirve_corpus(country):
    ds = load_dataset("spanish-ir/messirve", country)
    docs = ds["test"]["docid_text"]
    queries = ds["test"]["query"]
    doc_ids = ds["test"]["docid"]
    query_ids = ds["test"]["id"]
    return docs, queries, doc_ids, query_ids


def run(model, metrics, model_instance=None, tokenizer=None, reranker_model=None, reuse_run=False, rerank=False, ds="msmarco", country=None, limit=None):
    print("Starting Evaluation")
    if ds == "msmarco":
        qid_to_query = get_msmarco_queries()
        pid_to_passage = get_msmarco_passages()

        if limit:
            docs = list(pid_to_passage.values())[:limit]
            doc_ids = list(pid_to_passage.keys())[:limit]
        else:
            docs = list(pid_to_passage.values())
            doc_ids = list(pid_to_passage.keys())
        
        qrels_dev_path = os.path.join(STORAGE_DIR, "ms_marco_passage", "data", "qrels.dev.tsv")
        qrels_dev_df = pd.read_csv(
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
        qrels_dev_df = qrels_dev_df[qrels_dev_df["query_id"].isin(qids_dev_small)]
        qrels_dev_df = qrels_dev_df[qrels_dev_df["doc_id"].isin(doc_ids)]

        # save qrels_dev_df to tsv file
        path_to_reference = f"qrels_dev_full.tsv"
        qrels_dev_df.to_csv(path_to_reference, sep="\t", index=False, header=False)

        with open('gpt_responses_dev_small.json', 'r', encoding='utf-8') as f:
            qid_to_response = json.load(f)
        
        # filter qid_to_response to the ones present in qrels_dev_df
        qid_to_response = {qid: response for qid, response in qid_to_response.items() if int(qid) in qrels_dev_df["query_id"].unique()}

        query_ids = list(qid_to_response.keys())
        queries = list(qid_to_response.values())

        # query_ids = qrels_dev_df["query_id"].unique()
        # queries = [qid_to_query[qid] for qid in query_ids]

        print("Number of queries:", len(queries))
        print("Number of docs:", len(docs))
        # rel_doc_ids = qrels_dev_df["doc_id"].unique()
    elif ds == "messirve":
        docs, queries, doc_ids, query_ids = get_messirve_corpus(country)
    elif ds == "legal":
        doc_ids, docs = get_legal_dataset(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus_py.csv"))
        query_ids, queries = get_legal_queries(os.path.join(STORAGE_DIR, "legal_ir", "data", "queries_57.csv"))

        qrels_dev_path = os.path.join(STORAGE_DIR, "legal_ir", "data", "qrels_py.tsv")
        qrels_dev_df = pd.read_csv(
            qrels_dev_path,
            sep="\t",                # TREC qrels are usually tab-separated
            names=["query_id", "iteration", "doc_id", "relevance"],
            header=None,            # There's no header in qrels files
            dtype={"query_id": int, "iteration": int, "doc_id": int, "relevance": int}
        )
        
        # save qrels_dev_df to tsv file
        path_to_reference = os.path.join(STORAGE_DIR, "legal_ir", "data", "qrels_py.tsv")
    else:
        raise ValueError("Dataset not supported.")
    print("Data prepared.")

    # # read corpus from pickle file
    # with open("/media/discoexterno/leon/messirve-ir/datasets/messIRve_corpus.pkl", "rb") as f:
    #     corpus = pickle.load(f)

    # docs = corpus["corpus"]["text"]
    # doc_ids = corpus["corpus"]["docid"]

    device = torch.device("cuda:1")

    if model == "bm25":
        run_path = f"run_bm25_{country}.pkl"
        if not os.path.exists(run_path) or not reuse_run:
            run = retrieve_bm25(docs, queries, doc_ids, query_ids)
            # save qrels_model to disk using pickle
            with open(run_path, "wb") as f:
                print("Dumping run to pickle file...", end="")
                pickle.dump(run, f)
                print("Done")
        else:
            with open(run_path, "rb") as f:
                run = pickle.load(f)
        print("BM25 retrieval done.")
    elif model == "bge":
        run_path = f"run_bge_{country}.pkl"
        if not os.path.exists(run_path) or not reuse_run:
            checkpoint = 'BAAI/bge-m3'
            # checkpoint = 'BAAI/bge-m3-unsupervised'
            # checkpoint = 'BAAI/bge-m3-retromae'
            model = get_bge_m3_model(checkpoint)
            run = embed_bge(model, docs, queries, doc_ids, query_ids, reuse_run)
            # save run to disk using pickle
            with open(run_path, "wb") as f:
                print("Dumping run to pickle file...")
                pickle.dump(run, f)
        else:
            with open(run_path, "rb") as f:
                run = pickle.load(f)
    elif model == "jinja":
        run_path = f"run_jinja_{country}.pkl"
        if not os.path.exists(run_path) or not reuse_run:
            model = get_jinja_model()
            model.to(device)
            run = embed_jina_faiss(model, docs, queries, doc_ids, query_ids)
            # run = embed_jinja(model, docs, queries, doc_ids, query_ids)
            # save run to disk using pickle
            with open(run_path, "wb") as f:
                print("Dumping run to pickle file...")
                pickle.dump(run, f)
        else:
            with open(run_path, "rb") as f:
                run = pickle.load(f)
    elif model == "mamba":
        run_path = f"run_mamba_{country}.pkl"
        if not os.path.exists(run_path) or not reuse_run:
            model, tokenizer = model_setup.get_mamba_model()
            model.to(device)
            run = embed_mamba(model, tokenizer, docs, queries, doc_ids, query_ids)
            # save run to disk using pickle
            with open(run_path, "wb") as f:
                print("Dumping run to pickle file...")
                pickle.dump(run, f)
        else:
            with open(run_path, "rb") as f:
                run = pickle.load(f)
    elif model == "bge-finetuned":
        run_path = f"run_bge_finetuned_{country}.pkl"
        if not os.path.exists(run_path) or not reuse_run:
            checkpoint = 'test_encoder_only_m3_bge-m3_sd'
            # checkpoint = 'BAAI/bge-m3-unsupervised'
            # checkpoint = 'BAAI/bge-m3-retromae'
            model = get_bge_m3_model(checkpoint)
            run = embed_bge(model, docs, queries, doc_ids, query_ids, reuse_run)
            # save run to disk using pickle
            with open(run_path, "wb") as f:
                print("Dumping run to pickle file...")
                pickle.dump(run, f)
        else:
            with open(run_path, "rb") as f:
                run = pickle.load(f)
    elif model == "sentence-transformer":
        run_path = f"run_sentence_transformer_{country}.pkl"
        if not os.path.exists(run_path) or not reuse_run:
            model = model_instance
            run = embed_s_transformers(model, docs, queries, doc_ids, query_ids)
            # run = embed_s_transformers_faiss(model, docs, doc_ids, queries, query_ids)
            # save run to disk using pickle
            # with open(run_path, "wb") as f:
            #     print("Dumping run to pickle file...")
            #     pickle.dump(run, f)
        else:
            with open(run_path, "rb") as f:
                run = pickle.load(f)
    elif model == "qwen":
        from transformers import AutoModelForCausalLM
        run_path = f"run_qwen_{country}.pkl"
        if not os.path.exists(run_path) or not reuse_run:
            # checkpoint = '/media/discoexterno/leon/qwen-2-vec/run_2000_texts/output-model/checkpoint-560'
            # tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
            # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # model = AutoModelForCausalLM.from_pretrained(checkpoint)
            # model.resize_token_embeddings(len(tokenizer))
            device = "cuda"
            model_instance.to(device)
            run = embed_qwen(model_instance, tokenizer, docs, queries, doc_ids, query_ids)
    else:
        raise ValueError("Model not supported.")
    
    if rerank:
        # for each query rerank the top 1000 docs
        run = rerank_cross_encoder(reranker_model, tokenizer, run, 100, queries, query_ids, docs, doc_ids,
                                   max_length=2048)
    
    ir_metrics = get_eval_metrics(run, qrels_dev_df, doc_ids, metrics)
    create_results_file(run)
    msmarco_eval_ranking.main(path_to_reference, "results.txt")
    return ir_metrics


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
    run("qwen", metrics={'ndcg', 'ndcg_cut.10', 'recall_1000', 'recall_100', 'recall_10', 'recip_rank'}, ds="msmarco", model_instance=model, tokenizer=tokenizer, reuse_run=False)


if __name__ == "__main__":
    # reranker_model = AutoModelForSequenceClassification.from_pretrained("/media/discoexterno/leon/legal_ir/results/cross_encoder_2048/checkpoint-320")
    # tokenizer = AutoTokenizer.from_pretrained("/media/discoexterno/leon/legal_ir/results/cross_encoder_2048")
    # model = SentenceTransformer("multirun/2025-01-30/11-45-41/1/finetuned_models/distiluse-base-multilingual-cased-v1-exp_20250130_123521")
    # model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    # model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    # tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    # run(model="sentence-transformer", model_instance=model, metrics={'ndcg', 'ndcg_cut.10', 'recall_100', 'recall_10', 'recip_rank'}, country="ar", reuse_run=False, rerank=False)
    # run(model="bge-finetuned", metrics={'ndcg', 'ndcg_cut.10', 'recall_100', 'recall_10', 'recip_rank'}, country="ar")
    # run(model="jinja", metrics={'ndcg', 'ndcg_cut.10', 'recall_1000', 'recall_100', 'recall_10', 'recip_rank'}, reuse_run=False, ds="msmarco", limit=200_000)
    # run(model="mamba", metrics={'ndcg', 'ndcg_cut.10', 'recall_100', 'recip_rank'}, country="ar")
    run("sentence-transformer", metrics={'ndcg', 'ndcg_cut.10', 'recall_1000', 'recall_100', 'recall_10', 'recip_rank'}, ds="legal", model_instance=model, reuse_run=False)
    # run("bm25", metrics={'ndcg', 'ndcg_cut.10', 'recall_100', 'recall_10', 'recip_rank'}, ds="legal", reuse_run=False, rerank=True, reranker_model=reranker_model, tokenizer=tokenizer)
    # main()