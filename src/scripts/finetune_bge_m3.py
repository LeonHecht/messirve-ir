import os
from datasets import load_from_disk
import random
import json
import pickle
from tqdm import tqdm
import sys
import json
import csv
from typing import List, Dict


def configure_python_path():
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    print(f"Adding {project_root} to sys.path")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

configure_python_path()

from config.config import STORAGE_DIR
    

def finetune():
    base_dir = os.path.join(STORAGE_DIR, "legal_ir", "data")

    train_ds_path = "path"
    
    train_ds_path = "bce_6x_inpars_train_baai.jsonl"
    dev_ds_path = "bce_6x_inpars_dev_baai.jsonl"
    test_ds_path = "bce_6x_inpars_test_baai.jsonl"
    
    train_ds_path = os.path.join(base_dir, "datasets", "dual_encoder", train_ds_path)
    dev_ds_path = os.path.join(base_dir, "datasets", "dual_encoder", dev_ds_path)
    test_ds_path = os.path.join(base_dir, "datasets", "dual_encoder", test_ds_path)

    # run command in terminal
    command = f"""torchrun --nproc_per_node 2 \
	-m FlagEmbedding.finetune.embedder.encoder_only.m3 \
	--model_name_or_path BAAI/bge-m3 \
    --cache_dir ./cache/model \
    --train_data {train_ds_path}\
    --cache_path ./cache/data \
    --train_group_size 4 \
    --query_max_len 48 \
    --passage_max_len 1024 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
    --same_dataset_within_batch True \
    --small_threshold 0 \
    --drop_threshold 0 \
    --deepspeed /home/leon/tesis/messirve-ir/ds_stage0.json \
    --output_dir ./test_encoder_only_m3_bge-m3_sd \
    --overwrite_output_dir \
    --learning_rate 5e-6 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --logging_steps 20 \
    --save_steps 1000 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type m3_kd_loss \
    --unified_finetuning False \
    --use_self_distill False \
    --fix_encoder False \
    --self_distill_start_step 0 \
    --max_grad_norm 20 \
    --bf16 \
"""

    # execute command in terminal
    os.system(command)


def evaluate():
    base_dir = os.path.join(STORAGE_DIR, "legal_ir", "data")

    baai_dir = os.path.join(base_dir, "baai_bce_inpars_test")
    embd_dir = os.path.join(base_dir, "results", "baai_embeddings")
    eval_dir = os.path.join(base_dir, "results", "baai_eval")

    # model_path = "home/leon/tesis/messirve-ir/src/scripts/test_encoder_only_m3_bge-m3_sd/checkpoint-1308"
    model_path = "./src/scripts/test_encoder_only_m3_bge-m3_sd/checkpoint-1308"
    print(os.path.isdir(model_path))

    command = f"""python3 -m FlagEmbedding.evaluation.custom \
    --eval_name inpars_test_baai \
    --dataset_dir {baai_dir} \
    --splits test \
    --corpus_embd_save_dir {embd_dir} \
    --output_dir {eval_dir} \
    --cache_path ./cache/data \
    --overwrite True \
    --search_top_k 2048 \
    --k_values 10 100 \
    --eval_output_method markdown \
    --eval_output_path {os.path.join(eval_dir, "eval_results.md")} \
    --eval_metrics ndcg_at_10 recall_at_10 recall_at_100 \
    --devices cuda:0 cuda:1 \
    --cache_dir ./cache/model \
    --embedder_query_max_length 48 \
    --embedder_passage_max_length 1024 \
    --embedder_batch_size 2 \
    --embedder_model_class encoder-only-m3 \
    --embedder_name_or_path {model_path} \
"""
    # --embedder_name_or_path BAAI/bge-m3 \
    # --reranker_query_max_length 512 \
    # --reranker_max_length 1024 \
    # --reranker_name_or_path BAAI/bge-reranker-v2-m3 \
    # --rerank_top_k 100 \
    
    os.system(command)


if __name__ == "__main__":
    # finetune()
    evaluate()