import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scripts.dataset_loader import get_dataloaders
from models.model_resnet import ResNetWithCAW

# === Model Loading Function (Fixed) ===
def load_model(model_path, model_name, num_classes, use_caw, device):
    print(f"🔍 Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model = ResNetWithCAW(base_model=model_name, num_classes=num_classes, use_caw_layers=use_caw)
    model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
    model.to(device)
    model.eval()
    return model

# === Evaluation Function (Fixed) ===
def evaluate_model(model, test_loader, device):
    """Evaluates the model and returns AUC, Accuracy, F1-score, and Concept Alignment Score."""
    y_true, y_pred, y_prob = [], [], []
    concept_mask_alignment_scores = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="🔵 Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs, concept_masks = model(images)

            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

            if concept_masks is not None:
                concept_alignment_score = concept_masks.mean().item()  
                concept_mask_alignment_scores.append(concept_alignment_score)

    acc = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, average="weighted") * 100
    y_prob = np.array(y_prob)
    auc = roc_auc_score(y_true, y_prob[:, 1]) * 100 if y_prob.shape[1] == 2 else roc_auc_score(y_true, y_prob, multi_class="ovr") * 100
    concept_alignment_score = np.mean(concept_mask_alignment_scores) * 100 if concept_mask_alignment_scores else 0  

    return auc, acc, f1, concept_alignment_score

# === Main Evaluation Script (Fixed) ===
def main():
    parser = argparse.ArgumentParser(description="Evaluate trained ResNet models with CAW")
    parser.add_argument("--dataset", type=str, choices=["SkinCon", "Derm7pt"], required=True)
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--use_caw", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of evaluation runs (1 for single, >1 for multiple runs)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔹 Using device: {device}")

    # ✅ Fix `get_dataloaders` argument order
    dataloaders = get_dataloaders(disease_dataset=args.dataset, concept_dataset=None, batch_size=args.batch_size, num_workers=args.num_workers)

    if "test" not in dataloaders:
        raise ValueError(f"❌ No test set found for dataset {args.dataset}. Check dataset structure!")

    print(f"📂 Test Dataset Size: {len(dataloaders['test'])}")

    test_loader = dataloaders["test"]
    num_classes = 3 if args.dataset == "SkinCon" else 2

    auc_scores, acc_scores, f1_scores, concept_alignment_scores = [], [], [], []

    for i in range(args.num_runs):  
        model_path = os.path.join(args.output_dir, f"{args.dataset}_{args.model}_seed_{i}_best.pth")

        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}. Skipping...")
            continue

        print(f"🔄 Evaluating Model {i} from: {model_path}")
        
        model = load_model(model_path, args.model, num_classes, args.use_caw, device)
        auc, acc, f1, concept_alignment = evaluate_model(model, test_loader, device)

        auc_scores.append(auc)
        acc_scores.append(acc)
        f1_scores.append(f1)
        concept_alignment_scores.append(concept_alignment)

        print(f"📊 Seed {i}: AUC={auc:.2f}, ACC={acc:.2f}, F1={f1:.2f}, Concept Align={concept_alignment:.2f}")

    if auc_scores:
        print("\n📌 Final Evaluation Results Across Runs:")
        print(f"AUC: {np.mean(auc_scores):.2f} ± {np.std(auc_scores):.2f}")
        print(f"ACC: {np.mean(acc_scores):.2f} ± {np.std(acc_scores):.2f}")
        print(f"F1: {np.mean(f1_scores):.2f} ± {np.std(f1_scores):.2f}")
        print(f"Concept Alignment: {np.mean(concept_alignment_scores):.2f} ± {np.std(concept_alignment_scores):.2f}")
    else:
        print("❌ No models were found for evaluation.")

if __name__ == "__main__":
    main()
