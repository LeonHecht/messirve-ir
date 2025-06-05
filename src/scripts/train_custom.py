import os
import unsloth
import random
from datasets import load_from_disk, load_dataset
from transformers import TrainingArguments
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from unsloth import FastLanguageModel
from peft import LoraConfig, TaskType, get_peft_model
import pickle
import json
import pandas as pd
import uuid
import sys

def configure_python_path():
    """
    Add the project root directory to sys.path.

    This function finds the directory two levels up from this file
    (the repo root) and inserts it at the front of sys.path so that
    `config.config` can be imported without errors.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Apply the path tweak before any project imports
configure_python_path()

from src.trainers.info_nce_trainer import InfoNCERetrievalTrainerHNLLM
from config.config import MAX_QUERY_LEN, MAX_DOC_LEN, STORAGE_DIR
from src.models.model_setup import get_mamba_model, get_auto_model
from src.utils.train_utils import (
    tokenize_with_hard_negatives_messirve,
    custom_data_collator,
    get_msmarco_queries,
    get_msmarco_passages,
    get_msmarco_hard_negatives,
    tokenize_train_ds_msmarco,
    tokenize_test_ds_msmarco,
    tokenize_ds_legal
)
from src.utils.retrieval_utils import (
    get_legal_dataset,
    get_legal_queries
)
from src.utils.log_experiment import log_csv, log_md, log_plot
from src.eval_class import Evaluator

print("Max query len:", MAX_QUERY_LEN)
print("Max doc len:", MAX_DOC_LEN)


def log_experiment(trainer, training_args, model_checkpoint, exp_id, dataset_name, loss_name, gpu_name, ir_metrics):
        """
        Log experiment details using Markdown, CSV, and plot logging.

        Parameters
        ----------
        exp_id : str
            Experiment identifier.
        dataset_name : str
            Name of the dataset used.
        loss_name : str
            Name of the loss function used.
        gpu_name : str
            Name of the GPU used.
        ir_metrics : dict
            Dictionary containing IR metrics (e.g., nDCG, Recall, MRR).
        """
        # Retrieve training results from trainer state
        training_results = trainer.state.log_history
        # Convert training arguments to a dictionary
        training_args_dict = training_args.to_dict()
        # Determine experiment directory from output_dir
        exp_dir = training_args_dict.get("output_dir", "./results")
        model_name = model_checkpoint

        # Log Markdown file
        log_md(exp_dir, exp_id, model_name, dataset_name, loss_name,
               training_args_dict, gpu_name, training_results, ir_metrics)
        # Log CSV file
        log_csv(exp_id, model_name, dataset_name, loss_name,
                training_args_dict, gpu_name, training_results, ir_metrics)
        # Log training plot
        log_plot(exp_dir, exp_id, training_results)


def convert_jsonl_to_hf_dataset(jsonl_path: str) -> Dataset:
    """
    Convert a JSONL file into a HuggingFace Dataset.

    Parameters
    ----------
    jsonl_path : str
        Path to the input JSONL file. Each line must be a JSON object with
        keys: 'query', 'pos', 'neg', 'pos_scores', 'neg_scores', 'prompt',
        and 'type'.

    Returns
    -------
    Dataset
        A HuggingFace Dataset with columns:
        - 'query' (str)
        - 'positive' (str or None)
        - 'negative_1', 'negative_2', ..., up to the max number of negatives.
    """
    rows = []
    max_negatives = 0

    with open(jsonl_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            entry = json.loads(line)
            query = entry.get('query')
            pos_list = entry.get('pos', [])
            positive = pos_list[0] if pos_list else None

            negatives = entry.get('neg', [])
            max_negatives = max(max_negatives, len(negatives))

            row = {
                'query': query,
                'positive': positive,
            }
            for j, neg in enumerate(negatives, start=1):
                row[f'negative_{j}'] = neg

            rows.append(row)

    # Pad missing negative_i fields with None for consistency
    for row in rows:
        for j in range(1, max_negatives + 1):
            row.setdefault(f'negative_{j}', None)

    return Dataset.from_list(rows)


def make_dual_encoder_dataset(tsv_path: str,
                              neg_per_pos: int,
                              shuffle_negatives: bool = False,
                              seed: int = 42) -> Dataset:
    
    header = pd.read_csv(tsv_path, sep="\t", nrows=0).columns.tolist()

    if "doc_id" in header:
        doc_col = "doc_id"
    elif "chunk_id" in header:
        doc_col = "chunk_id"
    else:
        raise ValueError("TSV must contain either 'doc_id' or 'chunk_id' column.")

    df = pd.read_csv(tsv_path, sep="\t", dtype={"qid": str, doc_col: str, "label": int})

    # 2. Build rows for new dataset
    all_rows = []
    for qid, sub in df.groupby('qid', sort=False):
        positives = sub.loc[sub['label']==1, doc_col].tolist()
        negatives = sub.loc[sub['label']==0, doc_col].tolist()

        # sanity check: must exactly match your provided ratio
        total_negs = len(negatives)
        assert total_negs == len(positives) * neg_per_pos, (
            f"Query {qid} has {len(positives)} positives and "
            f"{total_negs} negatives; expected {len(positives)*neg_per_pos}"
        )

        if shuffle_negatives:
            import random
            random.seed(seed)
            random.shuffle(negatives)

        # 3. Partition negatives into contiguous chunks of size neg_per_pos
        for i, pos in enumerate(positives):
            start = i * neg_per_pos
            chunk = negatives[start : start + neg_per_pos]
            row = {'query': qid, 'positive': pos}
            for j, neg in enumerate(chunk, start=1):
                row[f'negative_{j}'] = neg
            all_rows.append(row)

    # 4. Convert back into a Hugging Face Dataset
    result_df = pd.DataFrame(all_rows)
    new_ds = Dataset.from_pandas(result_df, preserve_index=False)
    return new_ds


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
        ds = convert_jsonl_to_hf_dataset(
            jsonl_path=os.path.join(STORAGE_DIR, "legal_ir", "data", "datasets", "dual_encoder", "bce_6x_inpars_synthetic_chunked_baai.jsonl"),
        )
        
        # Split the dataset into train and test sets
        split = ds.train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"]
        test_ds = split["test"]
        print(f"Train dataset size: {len(train_ds)}")
        print(f"Test dataset size: {len(test_ds)}")

        # train_ds = train_ds.select(range(100))
        # test_ds = test_ds.select(range(100))

        query_ids, queries = get_legal_queries(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "consultas_sinteticas_380_filtered.tsv"))
        qid_to_query = {query_id: query for query_id, query in zip(query_ids, queries)}

        query_ids2, queries2 = get_legal_queries(os.path.join(STORAGE_DIR, "legal_ir", "data", "corpus", "inpars_mistral-small-2501_queries_Q1.tsv"))
        qid_to_query2 = {query_id: query for query_id, query in zip(query_ids2, queries2)}
        qid_to_query.update(qid_to_query2)

        num_negs = 6
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
        # checkpoint = "unsloth/Qwen2.5-0.5B-Instruct"
        # adapter_checkpoint = "/media/discoexterno/leon/ms_marco_passage/results/IR_ms_marco_peft_pretrain_100k/checkpoint-2812"
        adapter_checkpoint = "/media/discoexterno/leon/ms_marco_passage/results/IR_ms_marco_peft_pretrain_100k/saved_model"

        base_model = "unsloth/Qwen2.5-0.5B-Instruct"  # or "unsloth/Qwen2.5-0.5B", "unsloth/Qwen2.5-1.5B", etc.

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        processor = AutoProcessor.from_pretrained(base_model)
        
        model, _ = FastLanguageModel.from_pretrained(
            # Can select any from the below:
            # "unsloth/Qwen2.5-0.5B", "unsloth/Qwen2.5-1.5B", "unsloth/Qwen2.5-3B"
            # "unsloth/Qwen2.5-14B",  "unsloth/Qwen2.5-32B",  "unsloth/Qwen2.5-72B",
            # And also all Instruct versions and Math. Coding verisons!
            adapter_checkpoint,
            max_seq_length = 512,
            dtype = torch.bfloat16,
            load_in_4bit = False,
            # device_map="cuda:1",
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )

        # model.load_adapter(adapter_checkpoint)

        # model.train()

        # for name, param in model.named_parameters():
        #     if "lora_" in name:  # o el sufijo/prefijo que use tu adapter
        #         param.requires_grad = True

        # print("=== Parámetros y requires_grad ===")
        # for name, param in model.named_parameters():
        #     print(f"{name:60s} | requires_grad = {param.requires_grad}")
        # print("=== fin lista ===\n")

        print(tokenizer.pad_token_id)   # 128004
        print(tokenizer.pad_token)      # <|finetune_right_pad_id|>

        # model = FastLanguageModel.get_peft_model(
        #     model,
        #     r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        #     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
        #                     "gate_proj", "up_proj", "down_proj",],
        #     lora_alpha = 16,
        #     lora_dropout = 0, # Supports any, but = 0 is optimized
        #     bias = "none",    # Supports any, but = "none" is optimized
        #     # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        #     use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        #     random_state = 3407,
        #     use_rslora = True,  # We support rank stabilized LoRA
        #     loftq_config = None, # And LoftQ
        # )
    
    if which == "msmarco":
        # Apply tokenization to the dataset
        train_ds = tokenize_train_ds_msmarco(tokenizer, train_ds, qid_to_query, pid_to_passage, num_negs, reuse=False)
        test_ds = tokenize_test_ds_msmarco(tokenizer, test_ds, qid_to_query, pid_to_passage, num_negs, reuse=False)
    elif which == "messirve":
        # Apply tokenization to the dataset
        train_ds = train_ds.map(lambda x: tokenize_with_hard_negatives_messirve(tokenizer, x, append_eos=True), batched=True)
        test_ds = test_ds.map(lambda x: tokenize_with_hard_negatives_messirve(tokenizer, x, append_eos=True), batched=True)
    elif which == "legal":
        # Apply tokenization to the dataset
        train_ds = tokenize_ds_legal(tokenizer, train_ds, num_negs, max_doc_len=512, max_query_len=48)
        test_ds = tokenize_ds_legal(tokenizer, test_ds, num_negs, max_doc_len=512, max_query_len=48)
    else:
        raise ValueError("Dataset not supported")

    # output_dir = os.path.join(STORAGE_DIR, "ms_marco_passage", "results", "IR_unsloth_qwen0.5_5negs_rslora_50k_SFT_GPT")
    output_dir = os.path.join(STORAGE_DIR, "legal_ir", "results", "legal_eos_full_synthetic")
    print("Output dir:", output_dir)

    batch_size = 2
    gradient_accumulation_steps = 8
    epochs = 1

    # Compute total steps given your dataset and hyperparameters
    total_steps = (len(train_ds) // (batch_size * gradient_accumulation_steps)) * epochs
    eval_steps = total_steps // epochs // 4

    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,           # Directory to save checkpoints
        evaluation_strategy="steps",     # Evaluate at the end of each epoch
        eval_steps=eval_steps,                  # Evaluate every 500 steps
        learning_rate=5e-5,              # Learning rate
        per_device_train_batch_size=batch_size,  # Batch size for training
        per_device_eval_batch_size=1,   # Batch size for evaluation
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,              # Number of epochs
        weight_decay=0.01,               # Weight decay
        max_grad_norm=1,                # Maximum gradient norm
        save_strategy="steps",           # Save model checkpoints at the end of each epoch
        save_steps=eval_steps,                  # Save checkpoints every 500 stepss
        logging_dir="./logs",            # Directory for logs
        logging_steps=20,                # Log every 10 steps
        save_total_limit=1,              # Save only the last checkpoint
        remove_unused_columns=False,
        warmup_ratio=0.1,
        fp16=False,
        bf16=False,
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
        # tokenizer=tokenizer,             # Tokenizer for on-the-fly tokenization
        processing_class = processor,
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

    # evaluation.run("qwen", metrics={'ndcg', 'ndcg_cut.10', 'recall_1000', 'recall_100', 'recall_10', 'recip_rank'}, ds="legal", model_instance=model, tokenizer=tokenizer, reuse_run=False)

    # Evaluate IR metrics.
    evaluator = Evaluator(
        ds="legal-inpars",
        model_name="qwen",
        metric_names={'ndcg', 'ndcg_cut.10', 'recall_1000', 'recall_100', 'recall_10', 'recip_rank'},
        rerank=False,
        model_instance=model,
        tokenizer=tokenizer,
        reuse_run=False,
    )
    evaluator.evaluate()

    ir_metrics = {
        "ndcg": evaluator.metrics["ndcg"],
        "ndcg_cut_10": evaluator.metrics["ndcg_cut_10"],
        "recall_1000": evaluator.metrics["recall_1000"],
        "recall_100": evaluator.metrics["recall_100"],
        "recall_10": evaluator.metrics["recall_10"],
        "recip_rank": evaluator.metrics["recip_rank"],
    }

    gpu_name = "RTX A5000"
    exp_id = "exp_" + uuid.uuid4().hex[:8]
    dataset_name = "bce_ds"
    loss_name = "InfoNCE"
    log_experiment(trainer, training_args, base_model, exp_id, dataset_name, loss_name, gpu_name, ir_metrics)

if __name__ == "__main__":
    train()