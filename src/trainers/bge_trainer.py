from datasets import load_from_disk
from config.config import STORAGE_DIR
import os
import pandas as pd
import json


class BgeCrossEncoderTrainer:
    def __init__(self):
        pass

    def create_dataset(self):
        ds = load_from_disk(os.path.join(STORAGE_DIR, "legal_ir", "data", "triplet_ds_standard_ranknet"))

        # delete pos_label and neg_label columns
        ds = ds.remove_columns(["pos_label", "neg_label"])

        corpus_df = pd.read_csv(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus_py.csv"))
        doc_dict = {str(doc_id): doc for doc_id, doc in zip(corpus_df["Codigo"], corpus_df["text"])}

        queries_df = pd.read_csv(os.path.join(STORAGE_DIR, "legal_ir", "data", "queries_57.csv"))
        query_dict = {str(query_id): query for query_id, query in zip(queries_df["topic_id"], queries_df["Query"])}

        # create json file in format {"query": str, "pos": List[str], "neg":List[str], "prompt": str} prompt describing relation between query and doc
        json_dict = []
        for i in range(len(ds)):
            query_id = ds["query_id"][i]
            doc_id = ds["pos_doc_id"][i]
            pos_doc = doc_dict[doc_id]
            neg_doc_id = ds["neg_doc_id"][i]
            neg_doc = doc_dict[neg_doc_id]
            prompt = ""
            json_dict.append({"query": f"Caso legal que trata del siguiente tema: {query_dict[query_id]}", "pos": [pos_doc], "neg": [neg_doc], "prompt": prompt})
        
        # save json_dict to file
        with open(os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "train_bge_LLM_reranker_triplet_ds.json"), "w", encoding="utf-8") as f:
            for line in json_dict:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def train(self):
        import os

        # run command in terminal
        command = f"""torchrun --nproc_per_node 2 \
-m FlagEmbedding.llm_reranker.finetune_for_instruction.run \
--output_dir ./bge_train_output \
--model_name_or_path BAAI/bge-reranker-v2-gemma \
--train_data {os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "train_bge_LLM_reranker_triplet_ds.json")} \
--learning_rate 2e-4 \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \
--dataloader_drop_last True \
--query_max_len 40 \
--passage_max_len 2048 \
--train_group_size 16 \
--logging_steps 1 \
--save_steps 100 \
--save_total_limit 2 \
--ddp_find_unused_parameters False \
--gradient_checkpointing \
--deepspeed ds_stage0.json \
--warmup_ratio 0.1 \
--bf16 \
--use_lora True \
--lora_rank 32 \
--lora_alpha 64 \
--use_flash_attn True \
--target_modules q_proj k_proj v_proj o_proj
"""
        # --fp16 \

        # execute command in terminal
        os.system(command)


    def evaluate(self):
        pass


if __name__ == "__main__":
    trainer = BgeCrossEncoderTrainer()
    trainer.create_dataset()
    trainer.train()
    # trainer.evaluate()
    