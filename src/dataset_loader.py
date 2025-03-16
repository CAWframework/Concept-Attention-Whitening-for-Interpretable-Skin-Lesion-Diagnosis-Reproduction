import os
import torch
import pandas as pd
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class SkinConDataset(Dataset):
    """ Custom Dataset Class for SkinCon to correctly load images & labels. """

    def __init__(self, image_dir, metadata_file, transform=None):
        self.image_dir = image_dir
        self.df = pd.read_csv(metadata_file)
        self.image_column = "ImageID"
        self.label_column = "three_partition_label"
        self.label_map = {"malignant": 0, "benign": 1, "non-neoplastic": 2}
        self.df = self.df[self.df[self.label_column].isin(self.label_map)]  # Filter valid labels
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row[self.image_column])
        label = self.label_map[row[self.label_column]]

        # Load image
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

class ConceptDataset(Dataset):
    """ Custom Dataset for Concept Learning with CAW. """

    def __init__(self, concept_dir, concept_label_file, transform=None):
        self.concept_dir = concept_dir
        self.transform = transform

        # ✅ Load concept labels
        self.df = pd.read_csv(concept_label_file)
        # ✅ Ensure 'ImageID' is present
        if "ImageID" not in self.df.columns:
            raise KeyError(f"❌ Missing 'ImageID' column in {concept_label_file}. Check dataset preprocessing!")

        self.image_column = "ImageID"
        self.concept_columns = [col for col in self.df.columns if col != "ImageID"]
        # ✅ Extract concept folder names
        self.concept_dirs = {concept: os.path.join(concept_dir, concept) for concept in os.listdir(concept_dir)}

        # ✅ Ensure ImageID contains only filenames (remove paths)
        self.df["ImageID"] = self.df["ImageID"].apply(lambda x: os.path.basename(str(x)))  # Strip paths

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Load image and corresponding concept labels."""
        row = self.df.iloc[idx]
        image_name = str(row[self.image_column]).strip()

        # ✅ Search for the image in all concept subfolders
        image_path = None
        for concept_folder in self.concept_dirs.values():
            possible_path = os.path.join(concept_folder, image_name)
            if os.path.exists(possible_path):
                image_path = possible_path
                break  # Stop searching once found

        # ✅ Raise an error if the image wasn't found
        if image_path is None:
            raise FileNotFoundError(f"❌ Image not found in any concept folders: {image_name}")

        # ✅ Load image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        concept_labels = row[self.concept_columns].values.astype(float)
        return image, torch.tensor(concept_labels, dtype=torch.float)


def get_dataloaders(disease_dataset, concept_dataset=None, batch_size=64, num_workers=4, shuffle=True):
    """
    Loads both disease dataset D and concept dataset X_c for CAW training.

    Args:
        disease_dataset (str): Name of the skin disease dataset (e.g., "SkinCon" or "Derm7pt").
        concept_dataset (str): Name of the concept dataset (e.g., "Derm7pt_concepts"). Default: None.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for DataLoader.
        shuffle (bool): Whether to shuffle training data.

    Returns:
        dict: Dataloaders for train, validation, test (disease) and concept dataset (if available).
    """

    dataloaders = {}

    # === Define Transformations ===
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Augmentation only for training
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    concept_transform = test_transform  # Same preprocessing for concept dataset

    # === Load Disease Dataset (D) ===
    if disease_dataset == "Derm7pt":
        disease_dataset_path = os.path.abspath(os.path.join("datasets", disease_dataset))
        for split in ["train", "validation", "test"]:
            split_path = os.path.join(disease_dataset_path, split)

            if not os.path.isdir(split_path):
                print(f"⚠️ Warning: '{split}' folder is missing in {disease_dataset_path}. Skipping...")
                dataloaders[split] = None
                continue

            transform = train_transform if split == "train" else test_transform
            dataset = datasets.ImageFolder(root=split_path, transform=transform)

            if len(dataset) == 0:
                print(f"⚠️ Warning: No images found in '{split}' folder! Skipping...")
                dataloaders[split] = None
            else:
                dataloaders[split] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=(split == "train"),  # ✅ Only shuffle train data
                    num_workers=num_workers
                )

    elif disease_dataset == "SkinCon":
        metadata_file = os.path.abspath("datasets/SkinCon/annotations_fitzpatrick17k.csv")
        image_dir = os.path.abspath("datasets/SkinCon/images")

        if not os.path.exists(metadata_file) or not os.path.exists(image_dir):
            raise FileNotFoundError(f"❌ SkinCon dataset files missing. Check {image_dir} & {metadata_file}")

        # Load all images & labels from metadata file
        full_dataset = SkinConDataset(image_dir=image_dir, metadata_file=metadata_file, transform=train_transform)

        # Split dataset into train, validation, test
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

        dataloaders["train"] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        dataloaders["validation"] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        dataloaders["test"] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    else:
        raise ValueError(f"❌ Unsupported dataset: {disease_dataset}")

    # === Load Concept Dataset (Optional) ===
    if concept_dataset:
        concept_dataset_path = os.path.abspath(os.path.join("datasets", concept_dataset))
        concept_label_file = os.path.join(concept_dataset_path, "concept_labels.csv")

        if os.path.isdir(concept_dataset_path) and os.path.exists(concept_label_file):
            concept_dataset = ConceptDataset(concept_dataset_path, concept_label_file, transform=concept_transform)

            dataloaders["concept"] = DataLoader(
                concept_dataset,
                batch_size=batch_size,
                shuffle=True,  # ✅ Always shuffle concept dataset
                num_workers=num_workers
            )
        else:
            print(f"⚠️ Warning: Concept dataset '{concept_dataset_path}' not found. Skipping concept alignment.")
            dataloaders["concept"] = None

    return dataloaders
