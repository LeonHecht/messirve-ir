import sys
print(sys.executable)

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

print("All imports successful")


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
    # 3. Load a dataset to finetune on
    # dataset = load_dataset("sentence-transformers/all-nli", "triplet")
    train_dataset = load_from_disk("messirve_train_ar_hard_negatives_sbert")
    eval_dataset = load_from_disk("messirve_test_ar_hard_negatives_sbert")
    # test_dataset = dataset["test"]
    return train_dataset, eval_dataset


def get_loss(model):
    # 4. Define a loss function
    loss = MultipleNegativesRankingLoss(model)
    return loss


def get_training_args():
    # 5. (Optional) Specify training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir="finetuned_models/distiluse-base-multilingual-cased-v2",
        # Optional training parameters:
        # max_grad_norm=0.5,  # Clip gradients to prevent exploding gradients
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        logging_steps=20,
        run_name="distiluse-base-multilingual-cased-v2-5-HN",  # Will be used in W&B if `wandb` is installed
    )
    return args


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


def train():
    checkpoint = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    model = get_model(checkpoint)
    print("Model loaded")
    train_dataset, eval_dataset = get_dataset()
    print("Datasets loaded")
    loss = get_loss(model)
    print("Loss function defined")
    args = get_training_args()
    print("Training arguments defined")
    dev_evaluator = get_evaluator(eval_dataset, model)

    trainer = get_trainer(model, args, train_dataset, eval_dataset, loss, dev_evaluator)

    # Store training results
    training_results = []

    # Callback function to log results dynamically
    def log_callback(logs):
        if "loss" in logs and "epoch" in logs:
            training_results.append({
                "loss": logs["loss"],
                "grad_norm": logs.get("grad_norm", float('nan')),  # Some trainers don't log grad_norm explicitly
                "learning_rate": logs.get("learning_rate", float('nan')),
                "epoch": logs["epoch"],
            })

    trainer.add_callback(log_callback)

    # Store evaluation results
    eval_results = []

    def eval_callback(logs):
        """Capture evaluation logs dynamically."""
        if "loss" in logs and "epoch" in logs:
            eval_results.append({
                "epoch": logs["epoch"],
                "loss": logs["loss"],
            })

    # Attach the callback to the evaluation process
    trainer.add_eval_callback(eval_callback)

    trainer.train()

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
    experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_name = checkpoint.split("/")[-1]
    log_experiment.log_md(experiment_id, model_name, dataset_name, loss_name, args.to_dict(), hardware, training_results)
    log_experiment.log_csv(experiment_id, training_results, args)
    log_experiment.log_plot(experiment_id, training_results, eval_results)

    # 8. Save the trained model
    model.save_pretrained(f"finetuned_models/{model_name}-{experiment_id}")

    # # 9. (Optional) Push it to the Hugging Face Hub
    # model.push_to_hub("mpnet-base-all-nli-triplet")


if __name__ == "__main__":
    train()