import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from tqdm import tqdm
from itertools import zip_longest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from scripts.dataset_loader import get_dataloaders
from models.model_resnet import ResNetWithCAW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.datasets as datasets

# === Set Random Seed ===
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# === Model Initialization ===
def load_model(model_name, num_classes, use_caw, device):
    """Loads ResNet model with CAW layers."""
    print(f"🚀 Initializing {model_name} with CAW: {use_caw}")
    model = ResNetWithCAW(base_model=model_name, num_classes=num_classes, use_caw_layers=use_caw)
    return model.to(device)

# === Training Function ===
def train_epoch(model, train_loader, concept_loader, optimizer, criterion, device, epoch, accumulation_steps=1):
    """Trains the model for one epoch with gradient accumulation and concept alignment."""
    model.train()
    total_loss = 0.0
    y_true_train, y_pred_train, y_prob_train = [], [], []
    
    optimizer.zero_grad()

    for batch_idx, (train_batch, concept_batch) in enumerate(
    tqdm(zip_longest(train_loader, concept_loader, fillvalue=(None, None)), desc=f"🟢 Epoch {epoch} - Training", total=len(train_loader))
    ):
        if train_batch is None or train_batch[0] is None or train_batch[1] is None:
            continue  # Skip this batch

        if concept_batch is None or concept_batch[0] is None:
            concept_images = None  # Handle missing concept images
        else:
            concept_images, _ = concept_batch

        images, labels = train_batch
        images, labels = images.to(device), labels.to(device)
        if concept_images is not None:
            concept_images = concept_images.to(device)

        outputs, concept_masks = model(images, concept_images)

        # ✅ Standard Cross-Entropy Loss
        loss_ce = criterion(outputs, labels) / accumulation_steps

        # ✅ Concept Mask Regularization
        loss_q = 0
        if concept_masks is not None and isinstance(concept_masks, torch.Tensor):
            caw_matrix = model.caw_layers[0].orthogonal_matrix
            print(f"🔍 Sample Concept Mask (epoch {epoch}): {concept_masks[0].detach().cpu().numpy()}")
            print(f"🔍 Requires Grad: {concept_masks.requires_grad}")

            # Ensure concept mask dimensions match orthogonal matrix
            if concept_masks.shape[-1] != caw_matrix.shape[0]:
                concept_masks = torch.nn.functional.interpolate(concept_masks.unsqueeze(1), size=(caw_matrix.shape[0], caw_matrix.shape[1]), mode="bilinear", align_corners=False).squeeze(1)

            loss_q = torch.norm(concept_masks - caw_matrix)
            loss_q += 0.01 * torch.norm(concept_masks)  # Add small L2 regularization

        # ✅ Total Loss (Avoid NaN)
        loss = loss_ce + 0.1 * loss_q if isinstance(loss_q, torch.Tensor) else loss_ce
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        preds = outputs.argmax(dim=1).detach().cpu().numpy()

        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(preds)
        y_prob_train.extend(probs)

    train_acc = accuracy_score(y_true_train, y_pred_train) * 100
    train_f1 = f1_score(y_true_train, y_pred_train, average="weighted") * 100
    train_auc = roc_auc_score(y_true_train, np.array(y_prob_train)[:, 1]) * 100 if np.array(y_prob_train).shape[1] == 2 else roc_auc_score(y_true_train, np.array(y_prob_train), multi_class="ovr") * 100

    return total_loss / len(train_loader), train_auc, train_acc, train_f1
#validation function
def validate_epoch(model, val_loader, criterion, device, epoch):
    """Evaluates the model on the validation set."""
    model.eval()
    total_loss = 0.0
    y_true_val, y_pred_val, y_prob_val = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"🔵 Epoch {epoch} - Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs, concept_masks = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_true_val.extend(labels.cpu().numpy())
            y_pred_val.extend(preds)
            y_prob_val.extend(probs)

    val_acc = accuracy_score(y_true_val, y_pred_val) * 100
    val_f1 = f1_score(y_true_val, y_pred_val, average="weighted") * 100
    val_auc = roc_auc_score(y_true_val, np.array(y_prob_val)[:, 1]) * 100 if np.array(y_prob_val).shape[1] == 2 else roc_auc_score(y_true_val, np.array(y_prob_val), multi_class="ovr") * 100

    return total_loss / len(val_loader), val_auc, val_acc, val_f1

# === Main Training Script ===
def main():
    parser = argparse.ArgumentParser(description="Train ResNet with CAW on SkinCon or Derm7pt")
    parser.add_argument("--dataset", type=str, choices=["SkinCon", "Derm7pt"], required=True)
    parser.add_argument("--concept_dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--use_caw", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs (1 for single run, >1 for multiple runs)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloaders = get_dataloaders(args.dataset, args.concept_dataset, args.batch_size, args.num_workers)
    
    train_loader = dataloaders["train"]
    concept_loader = dataloaders["concept"]
    val_loader = dataloaders["validation"]

    num_classes = 3 if "SkinCon" in args.dataset else 2
    # Debugging: Print dataset attributes before extracting labels
    print(f"🔍 Dataset Type: {type(train_loader.dataset)}")
    print(f"🔍 Dataset Attributes: {dir(train_loader.dataset)}")
    # ✅ Extract Labels from Subset Datasets Safely
    if isinstance(train_loader.dataset, torch.utils.data.Subset):
        # Access the original dataset inside the Subset
        base_dataset = train_loader.dataset.dataset  
        if isinstance(base_dataset, datasets.ImageFolder):
            train_labels = np.array([base_dataset.targets[i] for i in train_loader.dataset.indices])  # Extract subset labels
        else:
            train_labels = np.array([base_dataset[i][1] for i in train_loader.dataset.indices])  # Generic fallback
    else:
        train_labels = np.array(getattr(train_loader.dataset, 'targets', []))  # Default extraction

    # ✅ Check if we successfully extracted labels
    if len(train_labels) == 0:
        raise ValueError("❌ Could not extract labels from dataset. Check dataset format.")


    train_labels = np.array(train_labels)
    unique_classes = np.unique(train_labels)

    if set(unique_classes) != set(np.arange(num_classes)):
        print(f"⚠️ Warning: Some classes are missing in the training set! Found: {unique_classes}")
        class_weights = np.ones(num_classes)  # Default to uniform weights
    else:
        class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=train_labels)

    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # === Train 3 Models with Different Seeds ===
    for i in range(args.num_runs):  
        set_random_seed(i)  
        model = load_model(args.model, num_classes, args.use_caw, device)  
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=5)

        best_auc = 0.0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_auc, train_acc, train_f1 = train_epoch(model, train_loader, concept_loader, optimizer, criterion, device, epoch)
            val_loss, val_auc, val_acc, val_f1 = validate_epoch(model, val_loader, criterion, device, epoch)

            scheduler.step(val_auc)
            if val_auc > best_auc:
                best_auc = val_auc
                model_path = os.path.join(args.output_dir, f"{args.dataset}_{args.model}_seed_{i}_best.pth")
                torch.save(model.state_dict(), model_path)
                print(f"✅ Saved Best Model for Seed {i}: {model_path} (AUC: {best_auc:.2f}%)")
                print(f"📊 Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.2f}, AUC: {train_auc:.2f}%, Acc: {train_acc:.2f}%, F1: {train_f1:.2f}% | Val Loss: {val_loss:.2f}, AUC: {val_auc:.2f}%, Acc: {val_acc:.2f}%, F1: {val_f1:.2f}%\n")

if __name__ == "__main__":
    main()
