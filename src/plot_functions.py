import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from scripts.dataset_loader import get_dataloaders
from models.model_resnet import ResNetWithCAW
from scripts.evaluate import evaluate_model  # ✅ Import only the function

# === Set Matplotlib to Agg Mode for Non-GUI Environments ===
import matplotlib
matplotlib.use('Agg')

# === Helper Functions ===
def create_plot_dir(output_dir):
    """Creates output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)

# === Function 1: Plot Confusion Matrix ===
def plot_confusion_matrix(conf_matrix, dataset, model_name, output_dir):
    """Plots and saves a confusion matrix."""
    create_plot_dir(output_dir)

    labels = ["Nevus", "Melanoma"] if dataset == "Derm7pt" else ["Malignant", "Benign", "Non-Neoplastic"]

    plt.figure(figsize=(5, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {dataset} ({model_name})")
    plt.savefig(os.path.join(output_dir, f"{dataset}_{model_name}_conf_matrix.png"))
    print(f"✅ Saved Confusion Matrix Plot at {output_dir}")

# === Function 2: Generate Activation Heatmap ===
def plot_activation_heatmap(model, test_loader, output_dir):
    """Plots and saves the correlation heatmap of CAW feature activations."""
    model.eval()
    create_plot_dir(output_dir)

    activations = []

    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Extracting activations from CAW layer"):
            images = images.to("cpu")
            
            # ✅ Corrected: Extract activations from CAW layer
            outputs, concept_masks = model(images)
            features = concept_masks  # ✅ Assign concept masks as features

            activations.append(features.cpu().numpy())

    # Compute correlation matrix and replace NaN values
    activations = np.vstack(activations)
    correlation_matrix = np.corrcoef(activations.T)
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)  # Replace NaNs with 0

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, vmin=-1, vmax=1)
    plt.title("CAW Activation Heatmap")
    plt.savefig(os.path.join(output_dir, "caw_activation_heatmap.png"))
    print(f"✅ Saved CAW Activation Heatmap at {output_dir}")

# === Function 3: Plot AUC, ACC, and F1 Scores ===
def plot_evaluation_metrics(auc_values, acc_values, f1_values, dataset, model_name, output_dir):
    """Plots and saves evaluation metrics with Mean ± Std."""
    create_plot_dir(output_dir)

    auc_mean, auc_std = np.mean(auc_values), np.std(auc_values)
    acc_mean, acc_std = np.mean(acc_values), np.std(acc_values)
    f1_mean, f1_std = np.mean(f1_values), np.std(f1_values)

    metrics = ["AUC", "ACC", "F1"]
    means = [auc_mean, acc_mean, f1_mean]
    stds = [auc_std, acc_std, f1_std]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=metrics, y=means, capsize=0.2)
    plt.errorbar(metrics, means, yerr=stds, fmt='o', color='black', capsize=5)

    plt.ylabel("Score (%)")
    plt.ylim(50, 100)
    plt.title(f"Evaluation Metrics - {dataset} ({model_name})\nMean ± Std Over 3 Runs")
    plt.savefig(os.path.join(output_dir, f"{dataset}_{model_name}_metrics.png"))
    print(f"✅ Saved Evaluation Metrics Plot at {output_dir}")

# === Function 4: Model Confidence Histogram ===
def plot_model_confidence(y_probs, dataset, model_name, output_dir):
    """Plots and saves a histogram of the model's confidence scores."""
    create_plot_dir(output_dir)

    plt.figure(figsize=(7, 5))
    plt.hist(y_probs, bins=20, color='blue', alpha=0.6, edgecolor='black')
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.title(f"Model Confidence Histogram - {dataset} ({model_name})")
    plt.savefig(os.path.join(output_dir, f"{dataset}_{model_name}_confidence_histogram.png"))
    print(f"✅ Saved Model Confidence Histogram at {output_dir}")

# === MAIN FUNCTION ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot evaluation results for CAW framework.")
    parser.add_argument("--dataset", type=str, choices=["Derm7pt", "SkinCon"], required=True)
    parser.add_argument("--model", type=str, choices=["resnet18", "resnet50"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    parser.add_argument("--use_caw", action="store_true", help="Use Concept-Attention Whitening (CAW)")
    parser.add_argument("--runs", type=int, default=3, help="Number of evaluation runs")

    args = parser.parse_args()

    # === Load Data ===
    dataloaders = get_dataloaders(args.dataset, batch_size=64, num_workers=4)
    test_loader = dataloaders["test"]

    # === Load Model ===
    print(f"🚀 Loading {args.model} model with CAW: {args.use_caw}")
    model = ResNetWithCAW(base_model=args.model, num_classes=3 if args.dataset == "SkinCon" else 2, use_caw_layers=args.use_caw)  
    model.to("cpu")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"⚠️ Checkpoint '{args.checkpoint}' not found!")

    print(f"🔄 Loading checkpoint from {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    # === Run Multiple Evaluations ===
    auc_results, acc_results, f1_results, y_probs = [], [], [], []

    for i in range(args.runs):
        print(f"🔄 Running Evaluation {i+1}/{args.runs}...")
        acc, f1, auc, loss, conf_matrix, y_prob = evaluate_model(model, test_loader, args.use_caw, return_probs=True)  # ✅ Pass use_caw argument
        acc_results.append(acc)
        f1_results.append(f1)
        auc_results.append(auc)
        y_probs.extend(y_prob)  # Collect all confidence scores

    # === Generate Plots ===
    plot_confusion_matrix(conf_matrix, args.dataset, args.model, args.output_dir)
    plot_activation_heatmap(model, test_loader, args.output_dir)
    plot_evaluation_metrics(auc_results, acc_results, f1_results, args.dataset, args.model, args.output_dir)
    plot_model_confidence(y_probs, args.dataset, args.model, args.output_dir)
