from models.model_setup import get_mamba_model, get_auto_model
from datasets import load_from_disk
from trainers.info_nce_trainer import InfoNCERetrievalTrainerHNLLM
from transformers import TrainingArguments
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel
from utils.train_utils import (
    tokenize_with_hard_negatives_msmarco,
    tokenize_with_hard_negatives_messirve,
    custom_data_collator
)
from peft import LoraConfig, TaskType, get_peft_model
import pickle
import evaluation
from datasets import load_dataset
import sys
print("Executable", sys.executable)
import os
# make only GPU0 visible
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from config import MAX_QUERY_LEN, MAX_DOC_LEN, STORAGE_DIR

print("Max query len:", MAX_QUERY_LEN)
print("Max doc len:", MAX_DOC_LEN)


def get_msmarco_queries():
    print("Loading MS MARCO queries...", end="")
    save_path = os.path.join(STORAGE_DIR, "ms_marco_passage", "data", "qid_to_query.pkl")
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            qid_to_query = pickle.load(f)
    else:
        query_dataset = load_dataset("sentence-transformers/msmarco-corpus", "query", split="train")
        qid_to_query = dict(zip(query_dataset["qid"], query_dataset["text"]))
        # print(qid_to_query[571018])
        # => "what are the liberal arts?"
        with open(save_path, "wb") as f:
            pickle.dump(qid_to_query, f)
    print("Done")
    return qid_to_query

def get_msmarco_passages():
    print("Loading MS MARCO passages...", end="")
    save_path = os.path.join(STORAGE_DIR, "ms_marco_passage", "data", "pid_to_passage.pkl")
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            pid_to_passage = pickle.load(f)
    else:
        passage_dataset = load_dataset("sentence-transformers/msmarco-corpus", "passage", split="train")
        pid_to_passage = dict(zip(passage_dataset["pid"], passage_dataset["text"]))
        # print(pid_to_passage[7349777])
        # => "liberal arts. 1. the academic course of instruction at a college 
        with open(save_path, "wb") as f:
            pickle.dump(pid_to_passage, f)
    print("Done")
    return pid_to_passage


def get_msmarco_hard_negatives(num_negs, reload=False):
    print("Loading hard negatives...", end="")
    save_path = os.path.join(STORAGE_DIR, "ms_marco_passage", "data", f"negatives_{num_negs}_msmarco.pkl")
    if os.path.exists(save_path) and not reload:
        with open(save_path, "rb") as f:
            negs_ds = pickle.load(f)
    else:
        negs_ds = load_dataset("sentence-transformers/msmarco-msmarco-distilbert-base-tas-b", "triplet-50-ids")
        negs_ds = negs_ds["train"]
        remove_cols = [f"negative_{i+1}" for i in range(num_negs, 50)]
        negs_ds = negs_ds.map(lambda x: x, remove_columns=remove_cols)
        # save to disk
        with open(save_path, "wb") as f:
            pickle.dump(negs_ds, f)
    print("Done")
    return negs_ds


def tokenize_train_ds_msmarco(tokenizer, train_ds_pre, qid_to_query, pid_to_passage, num_negs):
    print("Tokenizing train dataset...", end="")
    save_path = os.path.join(STORAGE_DIR, "ms_marco_passage", "data", f"train_ds_msmarco_{num_negs}negs_50k.pkl")
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            train_ds = pickle.load(f)
    else:
        train_ds = train_ds_pre.map(lambda x: tokenize_with_hard_negatives_msmarco(tokenizer, x, qid_to_query, pid_to_passage, num_negs, MAX_QUERY_LEN, MAX_DOC_LEN), batched=True)
        with open(save_path, "wb") as f:
            pickle.dump(train_ds, f)
    print("Done")
    return train_ds

    
def tokenize_test_ds_msmarco(tokenizer, test_ds_pre, qid_to_query, pid_to_passage, num_negs):
    print("Tokenizing test dataset...", end="")
    save_path = os.path.join(STORAGE_DIR, "ms_marco_passage", "data", f"test_ds_msmarco_{num_negs}negs_50k.pkl")
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            test_ds = pickle.load(f)
    else:
        test_ds = test_ds_pre.map(lambda x: tokenize_with_hard_negatives_msmarco(tokenizer, x, qid_to_query, pid_to_passage, num_negs, MAX_QUERY_LEN, MAX_DOC_LEN), batched=True)
        with open(save_path, "wb") as f:
            pickle.dump(test_ds, f)
    print("Done")
    return test_ds
    

def train():
    which = "msmarco"

    if which == "msmarco":
        qid_to_query = get_msmarco_queries()
        pid_to_passage = get_msmarco_passages()
        num_negs = 5
        print("Training on MS MARCO dataset...")
        negs_ds = get_msmarco_hard_negatives(num_negs, reload=True)
        negs_ds = negs_ds.select(range(50_000))

        # split negs_ds into train and eval sets with a 90/10 ratio
        split = negs_ds.train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"]
        test_ds = split["test"]

        checkpoint = STORAGE_DIR + "/qwen-2-vec/run_89622_texts_1_epoch/output-model/checkpoint-2500"
    else:
        country = "ar"
        print(f"Training on {country} dataset...")

        # load train_df from disk
        train_ds = load_from_disk(f"messirve_train_{country}_hard_negatives")
        # train_ds = train_ds.select(range(5000))

        test_ds = load_from_disk(f"messirve_test_{country}_hard_negatives")
        # test_ds = test_ds.select(range(1000))

        checkpoint = STORAGE_DIR + "/qwen-2-vec/run_89622_texts_1_epoch/output-model/checkpoint-2500"

    # if 'helga_g' in STORAGE_DIR:
    if False:
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            # Can select any from the below:
            # "unsloth/Qwen2.5-0.5B", "unsloth/Qwen2.5-1.5B", "unsloth/Qwen2.5-3B"
            # "unsloth/Qwen2.5-14B",  "unsloth/Qwen2.5-32B",  "unsloth/Qwen2.5-72B",
            # And also all Instruct versions and Math. Coding verisons!
            model_name = "unsloth/Qwen2.5-0.5B",
            max_seq_length = MAX_DOC_LEN,
            dtype = "bf16",
            load_in_4bit = False,
            # device_map="cuda:1",
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            # use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
    
    if which == "msmarco":
        # Apply tokenization to the dataset
        train_ds = tokenize_train_ds_msmarco(tokenizer, train_ds, qid_to_query, pid_to_passage, num_negs)
        test_ds = tokenize_test_ds_msmarco(tokenizer, test_ds, qid_to_query, pid_to_passage, num_negs)
    else:
        # Apply tokenization to the dataset
        train_ds = train_ds.map(lambda x: tokenize_with_hard_negatives_messirve(tokenizer, x, append_eos=True), batched=True)
        test_ds = test_ds.map(lambda x: tokenize_with_hard_negatives_messirve(tokenizer, x, append_eos=True), batched=True)

    output_dir = os.path.join(STORAGE_DIR, "ms_marco_passage", "results", "results_IR_ms_marco")

    batch_size = 16
    gradient_accumulation_steps = 4
    epochs = 1

    # Compute total steps given your dataset and hyperparameters
    total_steps = (len(train_ds) // (batch_size * gradient_accumulation_steps)) * epochs
    eval_steps = total_steps // epochs // 5

    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,           # Directory to save checkpoints
        evaluation_strategy="steps",     # Evaluate at the end of each epoch
        eval_steps=eval_steps,                  # Evaluate every 500 steps
        learning_rate=5e-5,              # Learning rate
        per_device_train_batch_size=batch_size,  # Batch size for training
        per_device_eval_batch_size=batch_size,   # Batch size for evaluation
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,              # Number of epochs
        weight_decay=0.01,               # Weight decay
        max_grad_norm=30,                # Maximum gradient norm
        save_strategy="steps",           # Save model checkpoints at the end of each epoch
        save_steps=eval_steps,                  # Save checkpoints every 500 stepss
        logging_dir="./logs",            # Directory for logs
        logging_steps=10,                # Log every 10 steps
        save_total_limit=1,              # Save only the last checkpoint
        remove_unused_columns=False,
        warmup_ratio=0.1,
        fp16=False,
        bf16=True,
        # gradient_checkpointing=True,
    )

    # Create Trainer Instance
    trainer = InfoNCERetrievalTrainerHNLLM(
        model=model,
        args=training_args,
        train_dataset=train_ds,     # Raw dataset for training
        eval_dataset=test_ds,       # Raw dataset for evaluation
        tokenizer=tokenizer,             # Tokenizer for on-the-fly tokenization
        data_collator=custom_data_collator,     # Handles dynamic padding and tokenization
    )

    # Train the Model
    trainer.train()

    # # Evaluate the Model
    # metrics = trainer.evaluate()
    # print(metrics)

    # Save the Model
    trainer.save_model(os.path.join(output_dir, "saved_model"))
    torch.save(trainer.state.log_history, os.path.join(output_dir, "training_metrics_hf.pth"))
    tokenizer.save_pretrained(output_dir)

    # evaluation.run("qwen", metrics={'ndcg', 'ndcg_cut.10', 'recall_100', 'recall_10', 'recip_rank'}, country="ar", model_instance=model, tokenizer=tokenizer, reuse_run=False)

if __name__ == "__main__":
    train()