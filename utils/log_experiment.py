import os
from datetime import datetime

STORAGE_DIR = "/tmpu/helga_g/leonh_a/messirve-ir"  # Change this path based on the environment
output_dir = os.path.join(STORAGE_DIR, "experiment_logs")
os.makedirs(output_dir, exist_ok=True)


def log_md(exp_id, model_name, dataset_name, loss_name, training_args, gpu_name, training_results, ir_metrics):
    # Define experiment parameters
    experiment_date = datetime.now().strftime('%Y-%m-%d')

    # Format the training results as a Markdown table
    training_table = "| Loss | Grad Norm | Learning Rate | Epoch \n"
    training_table += "|-------|-------|-----------|---------------|\n"
    for result in training_results:
        if 'loss' in result and 'grad_norm' in result:
            training_table += f"| {result['loss']:.4f} | {result['grad_norm']:.2e} | {result['learning_rate']:.2e} | {result['epoch']:.2f} |\n"
        elif 'loss' in result:
            training_table += f"| {result['loss']:.4f} | - | {result['learning_rate']:.2e} | {result['epoch']:.2f} |\n"
        elif 'eval_loss' in result:
            training_table += f"| {result['eval_loss']:.4f} | - | - | {result['epoch']:.2f} |\n"

    # Create Markdown content
    markdown_content = f"""# Experiment Log

## Experiment ID: {exp_id}
- **Date**: {experiment_date}
- **Model**: `{model_name}`
- **Dataset**: `{dataset_name}`
- **Loss Function**: `{loss_name}`
- **GPU Name**: `{gpu_name}`

### Training Arguments:
- **Batch Size**: {training_args["per_device_train_batch_size"]}
- **Learning Rate**: {training_args["learning_rate"]}
- **FP16**: {training_args["fp16"]}
- **BF16**: {training_args["bf16"]}
- **Gradient Clipping**: {training_args.get("max_grad_norm", "None")}
- **Warmup Ratio**: {training_args.get("warmup_ratio", "None")}
- **Weight Decay**: {training_args.get("weight_decay", "None")}
- **Epochs**: {training_args["num_train_epochs"]}

### Training Results:
{training_table}

### IR Metrics:
{ir_metrics}

### Observations:
(Write observations here manually)
    """
    filename = os.path.join(output_dir, f"{exp_id}.md")

    # Write to file
    with open(filename, "w") as file:
        file.write(markdown_content)

    print(f"Experiment log saved to: {filename}")


import csv

def log_csv(exp_id, model_name, dataset_name, loss_name, training_args, gpu_name, training_results, ir_metrics):
    """Logs experiment details as a new row in a CSV file."""
    
    # Define CSV file path
    csv_filename = os.path.join(STORAGE_DIR, "experiment_logs.csv")
    file_exists = os.path.isfile(csv_filename)  # Check if file already exists

    # Extract experiment metadata
    date = datetime.now().strftime('%Y-%m-%d')

    # Prepare the CSV row (Flatten the training args)
    csv_row = {
        "Experiment ID": exp_id,
        "Date": date,
        "Model": model_name,
        "Dataset": dataset_name,
        "Loss Function": loss_name,
        "Batch Size": training_args["per_device_train_batch_size"],
        "Learning Rate": training_args["learning_rate"],
        "FP16": training_args["fp16"],
        "BF16": training_args["bf16"],
        "Gradient Clipping": training_args.get("max_grad_norm", "None"),
        "Warmup Ratio": training_args.get("warmup_ratio", "None"),
        "Weight Decay": training_args.get("weight_decay", "None"),
        "Epochs": training_args["num_train_epochs"],
        "GPU": gpu_name,
    }

    
    last_epoch_result = training_results[-1]
    if "loss" in last_epoch_result:
        csv_row["Final Train Loss"] = last_epoch_result["loss"]
    elif "eval_loss" in last_epoch_result:
        csv_row["Final Test Loss"] = last_epoch_result["eval_loss"]
        previous_epoch_result = training_results[-2]
        csv_row["Final Train Loss"] = previous_epoch_result["loss"]
    else:
        csv_row["Final Train Loss"] = "N/A"
    
    csv_row["nDCG"] = ir_metrics["ndcg"]
    csv_row["nDCG@10"] = ir_metrics["ndcg_cut_10"]
    csv_row["Recall@100"] = ir_metrics["recall_100"]
    csv_row["Recall@10"] = ir_metrics["recall_10"]
    csv_row["MRR"] = ir_metrics["recip_rank"]

    # Define CSV headers
    headers = list(csv_row.keys())

    # Write to CSV file
    with open(csv_filename, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)

        # If the file doesn't exist, write the header first
        if not file_exists:
            writer.writeheader()

        # Write experiment data
        writer.writerow(csv_row)

    print(f"Experiment logged in: {csv_filename}")


import matplotlib.pyplot as plt

def log_plot(exp_id, training_results):
    """Plots training and evaluation loss curves and saves them to a file."""
  
    filename = os.path.join(output_dir, f"{exp_id}.png")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    train_epochs = []
    train_losses = []
    eval_epochs = []
    eval_losses = []

    for result in training_results:
        if "loss" in result:
            train_epochs.append(result["epoch"])
            train_losses.append(result["loss"])
        elif "eval_loss" in result:
            eval_epochs.append(result["epoch"])
            eval_losses.append(result["eval_loss"])

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_epochs, train_losses, marker="o", linestyle="-", label="Training Loss")
    plt.plot(eval_epochs, eval_losses, marker="o", linestyle="--", label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Evaluation Loss Curves for {exp_id}")
    plt.legend()
    plt.grid(True)

    # Save the plot to file
    plt.savefig(filename)
    plt.close()  # Close the plot to free memory

    print(f"Loss plot saved to: {filename}")

