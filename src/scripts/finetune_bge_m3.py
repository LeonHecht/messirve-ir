from datasets import load_from_disk
import random
import json
from tqdm import tqdm
import sys
print(sys.executable)
import pickle


def get_negatives(ds, query_id, k):
    """ Randomly sample negatives from the dataset."""
    indices = []
    while len(indices) < k:
        rand_i = random.randint(0, len(ds)-1)
        if rand_i not in indices and ds[rand_i]["id"] != query_id:
            indices.append(rand_i)
    # Use a generator to filter out examples with the given query_id
    negs = [ds[i]["docid_text"] for i in indices]
    return negs
    

def convert(split):
    print("Loading dataset...", end="")
    ds = load_from_disk("messirve_ar")
    print("Done")

    print("Loading hard negatives...", end="")
    with open("hard_negatives_test_bge_ar.pkl", "rb") as f:
        hard_negatives = pickle.load(f)
        print("Done")

    # ['id', 'query', 'docid', 'docid_text', 'query_date', 'answer_date', 'match_score', 'expanded_search', 'answer_type']
    ds = ds[split]

    # {"query": str, "pos": List[str], "neg":List[str], "pos_scores": List[int], "neg_scores": List[int], "prompt": str, "type": str}
    json_out = []

    for i in tqdm(range(len(ds["query"])), total=len(ds["query"])):
        query = ds["query"][i]
        query_id = ds["id"][i]
        docid_text = ds["docid_text"][i]
        doc_id = ds["docid"][i]

        # negs = get_negatives(ds, query_id, k=10)
        neg_doc_ids = hard_negatives[str(query_id)]     # hard_negatives = {query_id: [doc_id1, doc_id2, ...]}
        # get the text of the negativess
        neg_doc_indices = []
        for j in range(len(ds["id"])):
            if ds["docid"][j] in neg_doc_ids:
                neg_doc_indices.append(j)
        
        negs = [ds[index]["docid_text"] for index in neg_doc_indices if ds[index]["docid_text"] not in negs]        
        assert len(negs) == 15
        
        if query not in json_out:
            json_out.append({"query": query, "pos": [docid_text], "neg": negs, "neg_scores": [], "prompt": "", "type": ""})

    # dump json_dict to file
    with open("messirve_test_ar_bge_finetune.json", "w", encoding="utf-8") as f:
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
    --query_max_len 48 \
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
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
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
    --max_grad_norm 20"""
    # --fp16 \

    # execute command in terminal
    os.system(command)

if __name__ == "__main__":
    # convert("test")
    finetune()