import os
from datetime import datetime


def log_md(exp_id, model_name, dataset_name, loss_name, training_args, gpu_name, training_results, eval_results):
    # Define experiment parameters
    experiment_date = datetime.now().strftime('%Y-%m-%d')

    # Format the training results as a Markdown table
    training_table = "| Epoch | Loss  | Grad Norm | Learning Rate |\n"
    training_table += "|-------|-------|----------|--------------|\n"
    for result in training_results:
        training_table += f"| {result['loss']:.4f} | {result['grad_norm']:.2e} | {result['learning_rate']:.2e} | {result['epoch']:.2f} |\n"

    # Format the evaluation results as a Markdown table
    eval_table = "| Epoch | Eval Loss |\n"
    eval_table += "|-------|----------|\n"
    for result in eval_results:
        eval_table += f"| {result['loss']:.4f} | {result['epoch']:.2f} |\n"

    # Create Markdown content
    markdown_content = f"""# Experiment Log

    ## Experiment ID: {exp_id}
    - **Date**: {experiment_date}
    - **Model**: `{model_name}`
    - **Dataset**: `{dataset_name}`
    - **Loss Function**: `{loss_name}`
    - **GPU Name**: `{gpu_name}`

    ### Training Arguments:
    {training_args}

    ### Training Results:
    {training_table}
    
    ### Evaluation Results:
    {training_table}

    ### Observations:
    (Write your observations here manually)
    """

    # Define output directory and filename
    output_dir = "experiment_logs"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{exp_id}.md")

    # Write to file
    with open(filename, "w") as file:
        file.write(markdown_content)

    print(f"Experiment log saved to: {filename}")


import csv

def log_csv(exp_id, model_name, dataset_name, loss_name, training_results, eval_results, args, gpu_name):
    """Logs experiment details as a new row in a CSV file."""
    
    # Define CSV file path
    csv_filename = "experiment_logs.csv"
    file_exists = os.path.isfile(csv_filename)  # Check if file already exists

    # Extract experiment metadata
    date = datetime.now().strftime('%Y-%m-%d')
    
    # Extract training arguments
    training_args = args.to_dict()

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

    # Get last epoch results (final loss & grad norm)
    if training_results:
        last_epoch_result = training_results[-1]
        csv_row["Final Train Loss"] = last_epoch_result["loss"]
        csv_row["Final Grad Norm"] = last_epoch_result["grad_norm"]
    else:
        csv_row["Final Train Loss"] = "N/A"
        csv_row["Final Grad Norm"] = "N/A"
    
    # Get last epoch results (final loss & grad norm)
    if eval_results:
        last_epoch_result = eval_results[-1]
        csv_row["Final Eval Loss"] = last_epoch_result["loss"]
    else:
        csv_row["Final Eval Loss"] = "N/A"
    
    csv_row["nDCG"] = ""
    csv_row["nDCG@10"] = ""
    csv_row["Recall@100"] = ""
    csv_row["Recall@10"] = ""
    csv_row["MRR"] = ""

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

def log_plot(exp_id, training_results, eval_results):
    """Plots training and evaluation loss curves and saves them to a file."""
    
    output_dir = "experiment_logs"
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f"{exp_id}.png")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Extract training data
    train_epochs = [result["epoch"] for result in training_results]
    train_losses = [result["loss"] for result in training_results]

    # Extract evaluation data
    eval_epochs = [result["epoch"] for result in eval_results]
    eval_losses = [result["loss"] for result in eval_results]

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

