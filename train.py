import sys
print(sys.executable)
import os
# Define storage location for datasets, models and results
# For Miztli cluster: f"/tmpu/helga_g/{os.getenv('USER')}/messirve-ir"
# For linux server: "/media/discoexterno/leon/messirve-ir"

STORAGE_DIR = os.getenv("STORAGE_DIR")
print(f"STORAGE_DIR: {STORAGE_DIR}")
# STORAGE_DIR = os.getenv("STORAGE_DIR", "$HOME/tesis/messirve-ir")
import json
from datetime import datetime
import torch
from datasets import load_from_disk, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

from utils import log_experiment

import hydra
from omegaconf import DictConfig

import evaluation

print("All imports successful")


def make_log_dir():
    os.makedirs(os.path.join(STORAGE_DIR, "experiment_logs"), exist_ok=True)
    # make a directory with the format YYYY-MM-DD for todays experiments
    today_dir = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(os.path.join(STORAGE_DIR, "experiment_logs", today_dir), exist_ok=True)
    # make a directory with the format HH-MM-SS for the current experiment
    experiment_dir = datetime.now().strftime("%H-%M-%S")
    os.makedirs(os.path.join(STORAGE_DIR, "experiment_logs", today_dir, experiment_dir), exist_ok=True)
    return os.path.join(STORAGE_DIR, "experiment_logs", today_dir, experiment_dir)


def get_model(checkpoint):
    # 1. Load a model to finetune with 2. (Optional) model card data
    model = SentenceTransformer(
        checkpoint,
        model_card_data=SentenceTransformerModelCardData(
            license="apache-2.0"
        )
    )
    return model


def get_dataset():
    train_path = os.path.join(STORAGE_DIR, "datasets", "messirve_train_ar_hard_negatives_sbert")
    eval_path = os.path.join(STORAGE_DIR, "datasets", "messirve_test_ar_hard_negatives_sbert")
    train_dataset = load_from_disk(train_path)
    eval_dataset = load_from_disk(eval_path)

    # cut train dataset to 1000 samples
    train_dataset = train_dataset.select(range(1000))
    # cut eval dataset to 300 samples
    eval_dataset = eval_dataset.select(range(300))
    
    return train_dataset, eval_dataset


def get_evaluator(eval_dataset, model):
    # # 6. (Optional) Create an evaluator & evaluate the base model
    # dev_evaluator = TripletEvaluator(
    #     anchors=eval_dataset["anchor"],
    #     positives=eval_dataset["positive"],
    #     negatives=eval_dataset["negative"],
    #     name="all-nli-dev",
    # )
    # dev_evaluator(model)
    return None


def get_trainer(model, args, train_dataset, eval_dataset, loss, dev_evaluator):
    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=None,
    )
    return trainer


def extract_training_metrics(output_dir):
    """Load training and evaluation logs from trainer_state.json"""
    checkpoint_dir = os.listdir(output_dir)[0]
    trainer_state_path = os.path.join(output_dir, checkpoint_dir, "trainer_state.json")

    if not os.path.exists(trainer_state_path):
        raise FileNotFoundError(f"No trainer_state.json found in {output_dir}")

    with open(trainer_state_path, "r") as f:
        trainer_state = json.load(f)

    training_results = trainer_state.get("log_history", [])
    
    return training_results


@hydra.main(config_path=".", config_name="config")
def train(cfg: DictConfig):
    
    experiment_dir = make_log_dir()

    # Access parameters from the Hydra config
    checkpoint = cfg.experiment.checkpoint
    dataset_name = cfg.experiment.dataset_name
    loss_name = cfg.experiment.loss_name
    learning_rate = cfg.experiment.learning_rate
    batch_size = cfg.experiment.batch_size
    num_epochs = cfg.experiment.num_epochs
    warmup_ratio = cfg.experiment.warmup_ratio
    weight_decay = cfg.experiment.weight_decay
    max_grad_norm = cfg.experiment.max_grad_norm
    fp16 = cfg.experiment.fp16
    bf16 = cfg.experiment.bf16
    eval_steps = cfg.experiment.eval_steps
    save_steps = cfg.experiment.save_steps

    experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(experiment_dir, "checkpoints", f"{checkpoint}")
    os.makedirs(output_dir, exist_ok=True)

    print("\n\n---------------------------------------------")
    print("Starting experiment:", experiment_id)
    print("With parameters:")
    print(f"Checkpoint: {checkpoint}")
    print(f"Dataset: {dataset_name}")
    print(f"Loss: {loss_name}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Warmup Ratio: {warmup_ratio}")
    print(f"Weight Decay: {weight_decay}")
    print(f"Max Grad Norm: {max_grad_norm}")
    print(f"FP16: {fp16}")
    print(f"BF16: {bf16}")
    print("---------------------------------------------\n")

    model = get_model(checkpoint)
    print("Model loaded")
    train_dataset, eval_dataset = get_dataset()
    print("Datasets loaded")
    loss = MultipleNegativesRankingLoss(model)
    print("Loss function defined")

    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=output_dir,  # Output directory
        # Optional training parameters:
        max_grad_norm=max_grad_norm,  # Clip gradients to prevent exploding gradients
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,  # Warmup ratio for the learning rate scheduler
        weight_decay=weight_decay,  # Strength of weight decay
        fp16=fp16,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=bf16,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=1,
        logging_steps=20,
        # run_name="distiluse-base-multilingual-cased-v2-5-HN",  # Will be used in W&B if `wandb` is installed
    )
    print("Training arguments defined")

    dev_evaluator = get_evaluator(eval_dataset, model)

    trainer = get_trainer(model, args, train_dataset, eval_dataset, loss, dev_evaluator)
    
    trainer.train()

    # Store training results
    training_metrics = extract_training_metrics(output_dir)

    ir_metrics = evaluation.run("sentence-transformer", metrics={'ndcg', 'ndcg_cut.10', 'recall_100', 'recall_10', 'recip_rank'}, country="ar", model_instance=model, reuse_run=False)

    # # (Optional) Evaluate the trained model on the test set
    # test_evaluator = TripletEvaluator(
    #     anchors=test_dataset["anchor"],
    #     positives=test_dataset["positive"],
    #     negatives=test_dataset["negative"],
    #     name="all-nli-test",
    # )
    # test_evaluator(model)

    dataset_name = "messirve_ar_hard_negatives5_sbert"
    loss_name = "MultipleNegativesRankingLoss"
    hardware = torch.cuda.get_device_name(0)
    model_name = checkpoint.split("/")[-1]
    log_experiment.log_md(experiment_dir, experiment_id, model_name, dataset_name, loss_name, args.to_dict(), hardware, training_metrics, ir_metrics)
    log_experiment.log_csv(experiment_id, model_name, dataset_name, loss_name, args.to_dict(), hardware, training_metrics, ir_metrics)
    log_experiment.log_plot(experiment_dir, experiment_id, training_metrics)

    # 8. Save the trained model
    model_save_path = os.path.join(experiment_dir, "finetuned_models", f"{model_name}-{experiment_id}")
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    print(f"Model saved to: {model_save_path}")

    # # 9. (Optional) Push it to the Hugging Face Hub
    # model.push_to_hub("mpnet-base-all-nli-triplet")


if __name__ == "__main__":
    train()