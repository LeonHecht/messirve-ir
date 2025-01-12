from datasets import load_dataset
import random
import json
from tqdm import tqdm
import sys
print(sys.executable)


def get_negatives(train_ds, query_id, k):
    """ Randomly sample negatives from the dataset."""
    indices = []
    while len(indices) < k:
        rand_i = random.randint(0, len(train_ds)-1)
        if rand_i not in indices and train_ds[rand_i]["id"] != query_id:
            indices.append(rand_i)
    # Use a generator to filter out examples with the given query_id
    negs = [train_ds[i]["docid_text"] for i in indices]
    return negs

def convert():
    country = "ar"
    ds = load_dataset("spanish-ir/messirve", country)

    # ['id', 'query', 'docid', 'docid_text', 'query_date', 'answer_date', 'match_score', 'expanded_search', 'answer_type']
    train_ds = ds["train"]

    # {"query": str, "pos": List[str], "neg":List[str], "pos_scores": List[int], "neg_scores": List[int], "prompt": str, "type": str}
    json_out = []

    for i, example in tqdm(enumerate(train_ds), total=len(train_ds)):
        query = example["query"]
        query_id = example["id"]
        docid_text = example["docid_text"]

        negs = get_negatives(train_ds, query_id, k=10)

        if query not in json_out:
            json_out.append({"query": query, "pos": [docid_text], "neg": negs, "neg_scores": [], "prompt": "", "type": ""})

    # dump json_dict to file
    with open("messirve.json", "w", encoding="utf-8") as f:
        for line in json_out:
            json.dump(line, f, ensure_ascii=False)
            f.write("\n")


def finetune():
    import os

    # run command in terminal
    command = """torchrun --nproc_per_node 2 \
	-m FlagEmbedding.finetune.embedder.encoder_only.m3 \
	--model_name_or_path BAAI/bge-m3 \
    --cache_dir ./cache/model \
    --train_data ./messirve.json\
    --cache_path ./cache/data \
    --train_group_size 8 \
    --query_max_len 256 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
    --same_dataset_within_batch True \
    --small_threshold 0 \
    --drop_threshold 0 \
    --deepspeed ds_stage0.json \
    --output_dir ./test_encoder_only_m3_bge-m3_sd \
    --overwrite_output_dir \
    --learning_rate 5e-6 \
    --fp16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --logging_steps 1 \
    --save_steps 1000 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type m3_kd_loss \
    --unified_finetuning True \
    --use_self_distill True \
    --fix_encoder False \
    --self_distill_start_step 0"""

    # execute command in terminal
    os.system(command)

if __name__ == "__main__":
    # convert()
    finetune()