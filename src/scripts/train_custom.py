from models.model_setup import get_mamba_model, get_auto_model
from datasets import load_from_disk
from trainers.info_nce_trainer import InfoNCERetrievalTrainerHNLLM
from transformers import TrainingArguments
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel
from utils.train_utils import (
    tokenize_with_hard_negatives_messirve,
    custom_data_collator,
    get_msmarco_queries,
    get_msmarco_passages,
    get_msmarco_hard_negatives,
    tokenize_train_ds_msmarco,
    tokenize_test_ds_msmarco
)
from peft import LoraConfig, TaskType, get_peft_model
import pickle
import src.evaluation as evaluation
from datasets import load_dataset
import sys
print("Executable", sys.executable)
import json
import os
import pandas as pd
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random

from config.config import MAX_QUERY_LEN, MAX_DOC_LEN, STORAGE_DIR

print("Max query len:", MAX_QUERY_LEN)
print("Max doc len:", MAX_DOC_LEN)


def train():
    which = "legal"

    if which == "msmarco":
        qid_to_query = get_msmarco_queries()
        pid_to_passage = get_msmarco_passages()
        
        with open("qid_to_response.json", "r", encoding='utf-8') as f:
            qid_to_response = json.load(f)

        num_negs = 5
        print("Training on MS MARCO dataset...")
        negs_ds = get_msmarco_hard_negatives(num_negs, reload=True)
        negs_ds = negs_ds.select(range(50_000))

        for example in negs_ds:
            qid = example["query"]
            response = qid_to_response[str(qid)]
            example["query"] = response

        # split negs_ds into train and eval sets with a 90/10 ratio
        split = negs_ds.train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"]
        test_ds = split["test"]

        # checkpoint = STORAGE_DIR + "/qwen-2-vec/run_89622_texts_1_epoch/output-model/checkpoint-2500"
    elif which == "messirve":
        country = "ar"
        print(f"Training on {country} dataset...")

        # load train_df from disk
        train_ds = load_from_disk(f"messirve_train_{country}_hard_negatives")
        # train_ds = train_ds.select(range(5000))

        test_ds = load_from_disk(f"messirve_test_{country}_hard_negatives")
        # test_ds = test_ds.select(range(1000))

        checkpoint = STORAGE_DIR + "/qwen-2-vec/run_89622_texts_1_epoch/output-model/checkpoint-2500"
    elif which == "legal":
        # Read corpus and queries
        corpus = pd.read_csv(
            os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus_py.csv"),
            usecols=["Codigo", "text"]
        )
        pid_to_passage = dict(zip(corpus["Codigo"], corpus["text"]))
        doc_ids = list(corpus["Codigo"])

        queries = pd.read_csv(
            os.path.join(STORAGE_DIR, "legal_ir", "data", "queries_57.csv"),
            usecols=["topic_id", "Query"]
        )
        qid_to_query = dict(zip(queries["topic_id"], queries["Query"]))

        # Read qrels
        with open(os.path.join(STORAGE_DIR, "legal_ir", "data", "qrels_py.tsv"), "r") as f:
            qrels = f.readlines()

        # Build dataset dictionary with positives and negatives for each query
        ds = {}
        num_negs = 5

        for row in qrels:
            row = row.split("\t")
            row[-1] = row[-1].replace("\n", "")
            row = [int(val) for val in row]
            curr_query = row[0]
            if curr_query not in ds:
                ds[curr_query] = {"positives": [], "negatives": []}

            if row[-1] == 2 or row[-1] == 3:
                ds[curr_query]["positives"].append(row[2])
            elif row[-1] == 0 or row[-1] == 1:
                ds[curr_query]["negatives"].append(row[2])

        # Ensure each query has at least 5 negatives (filling with random ones if needed)
        for qid, doc_dict in ds.items():
            if len(doc_dict["negatives"]) < 5:
                for _ in range(5 - len(doc_dict["negatives"])):
                    neg = doc_ids[0]
                    while neg in doc_dict["negatives"]:
                        neg = doc_ids[random.randint(0, len(doc_ids) - 1)]
                    doc_dict["negatives"].append(neg)

        # Build the final dataset as a list of dictionaries
        dataset = []
        for qid, doc_dict in ds.items():
            for pos in doc_dict["positives"]:
                record = {"query": qid, "positive": pos}
                for n, neg in enumerate(doc_dict["negatives"]):
                    if n == num_negs:
                        break
                    record[f"negative_{n+1}"] = neg
                dataset.append(record)

        # --- Query-level split --- #
        # 1. Get unique queries from the dataset
        unique_queries = list({entry["query"] for entry in dataset})
        # 2. Shuffle and split unique queries (e.g., 70% train, 30% test)
        random.seed(42)
        random.shuffle(unique_queries)
        split_index = int(0.7 * len(unique_queries))
        train_queries = set(unique_queries[:split_index])
        test_queries = set(unique_queries[split_index:])

        # 3. Filter dataset entries by query membership
        train_data = [entry for entry in dataset if entry["query"] in train_queries]
        test_data = [entry for entry in dataset if entry["query"] in test_queries]

        # Optionally, write out each split as TSV files
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        train_df.to_csv(
            os.path.join(STORAGE_DIR, "legal_ir", "data", "train_dataset.tsv"),
            sep="\t",
            index=False
        )
        test_df.to_csv(
            os.path.join(STORAGE_DIR, "legal_ir", "data", "test_dataset.tsv"),
            sep="\t",
            index=False
        )

        # Load splits into Hugging Face Dataset objects if needed:
        train_ds = Dataset.from_pandas(train_df)
        test_ds = Dataset.from_pandas(test_df)

        # Save datasets to disk
        train_ds.save_to_disk(os.path.join(STORAGE_DIR, "legal_ir", "data", "train_ds"))
        test_ds.save_to_disk(os.path.join(STORAGE_DIR, "legal_ir", "data", "test_ds"))
    else:
        raise ValueError("Dataset not supported")
    # if 'helga_g' in STORAGE_DIR:
    if False:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        lora_config = LoraConfig(
            r=16,
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
            model_name = "unsloth/Qwen2.5-0.5B-Instruct",
            max_seq_length = MAX_DOC_LEN,
            dtype = "bf16",
            load_in_4bit = False,
            # device_map="cuda:1",
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )

        print(tokenizer.pad_token_id)   # 128004
        print(tokenizer.pad_token)      # <|finetune_right_pad_id|>

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
            use_rslora = True,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
    
    if which == "msmarco" or which == "legal":
        # Apply tokenization to the dataset
        train_ds = tokenize_train_ds_msmarco(tokenizer, train_ds, qid_to_query, pid_to_passage, num_negs, reuse=False)
        test_ds = tokenize_test_ds_msmarco(tokenizer, test_ds, qid_to_query, pid_to_passage, num_negs, reuse=False)
    elif which == "messirve":
        # Apply tokenization to the dataset
        train_ds = train_ds.map(lambda x: tokenize_with_hard_negatives_messirve(tokenizer, x, append_eos=True), batched=True)
        test_ds = test_ds.map(lambda x: tokenize_with_hard_negatives_messirve(tokenizer, x, append_eos=True), batched=True)
    else:
        raise ValueError("Dataset not supported")

    # output_dir = os.path.join(STORAGE_DIR, "ms_marco_passage", "results", "IR_unsloth_qwen0.5_5negs_rslora_50k_SFT_GPT")
    output_dir = os.path.join(STORAGE_DIR, "legal_ir", "results", "test_py")
    print("Output dir:", output_dir)

    batch_size = 2
    gradient_accumulation_steps = 4
    epochs = 4

    # Compute total steps given your dataset and hyperparameters
    total_steps = (len(train_ds) // (batch_size * gradient_accumulation_steps)) * epochs
    eval_steps = total_steps // epochs // 4

    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,           # Directory to save checkpoints
        evaluation_strategy="steps",     # Evaluate at the end of each epoch
        eval_steps=eval_steps,                  # Evaluate every 500 steps
        learning_rate=5e-4,              # Learning rate
        per_device_train_batch_size=batch_size,  # Batch size for training
        per_device_eval_batch_size=batch_size,   # Batch size for evaluation
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,              # Number of epochs
        weight_decay=0.01,               # Weight decay
        max_grad_norm=30,                # Maximum gradient norm
        save_strategy="steps",           # Save model checkpoints at the end of each epoch
        save_steps=eval_steps,                  # Save checkpoints every 500 stepss
        logging_dir="./logs",            # Directory for logs
        logging_steps=2,                # Log every 10 steps
        save_total_limit=1,              # Save only the last checkpoint
        remove_unused_columns=False,
        warmup_ratio=0.1,
        fp16=False,
        bf16=True,
        # gradient_checkpointing=True,
        # optim = "adamw_8bit",
    )

    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

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
    model.save_pretrained(os.path.join(output_dir, "saved_model"))
    torch.save(trainer.state.log_history, os.path.join(output_dir, "training_metrics_hf.pth"))
    tokenizer.save_pretrained(os.path.join(output_dir, "saved_model"))

    evaluation.run("qwen", metrics={'ndcg', 'ndcg_cut.10', 'recall_1000', 'recall_100', 'recall_10', 'recip_rank'}, ds="legal", model_instance=model, tokenizer=tokenizer, reuse_run=False)

if __name__ == "__main__":
    train()