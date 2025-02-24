import os
import glob
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# === CONFIGURE DATASETS ===
DATASET_CONFIGS = {
    "Derm7pt": {
        "DATASET_DIR": "datasets/Derm7pt/images",
        "OUTPUT_DIR": "datasets/Derm7pt",
        "METADATA_FILE": "datasets/Derm7pt/meta/meta.csv",
        "image_column": "derm",  # Image filename column
        "diagnosis_column": "diagnosis",
        "train_split_file": "datasets/Derm7pt/meta/train_indexes.csv",
        "val_split_file": "datasets/Derm7pt/meta/valid_indexes.csv",
        "test_split_file": "datasets/Derm7pt/meta/test_indexes.csv",
        "valid_classes": {
            "nevus": [
                "clark nevus", "reed or spitz nevus", "dermal nevus",
                "blue nevus", "congenital nevus", "combined nevus", "recurrent nevus"
            ],
            "melanoma": [
                "melanoma (less than 0.76 mm)", "melanoma (in situ)", "melanoma (0.76 to 1.5 mm)",
                "melanoma (more than 1.5 mm)", "melanoma metastasis", "melanoma"
            ]
        },
        "concepts": [
            "pigment_network", "streaks", "pigmentation", "regression_structures", 
            "dots_and_globules", "blue_whitish_veil", "vascular_structures"
        ]
    },
    "SkinCon": {
        "DATASET_DIR": "datasets/SkinCon/images",
        "OUTPUT_DIR": "datasets/SkinCon",
        "METADATA_FILE": "datasets/SkinCon/annotations_fitzpatrick17k.csv",
        "image_column": "ImageID",  # Image filename column
        "diagnosis_column": "three_partition_label",
        "valid_classes": {
            "malignant": ["malignant"],
            "benign": ["benign"],
            "non-neoplastic": ["non-neoplastic"]
        },
        "concepts": [
            "Papule", "Scale", "Plaque", "Scar", "Pustule", "Friable", "Bulla", 
            "Dome-shaped", "Patch", "Brown(Hyperpigmentation)", "Nodule", 
            "Ulcer", "White(Hypopigmentation)", "Crust", "Erosion", "Purple", 
            "Atrophy", "Yellow", "Exudate", "Black", "Telangiectasia", "Erythema"
        ]
    }
}

def process_dataset(dataset_name):
    """Processes and splits the dataset correctly by first filtering, then splitting."""
    config = DATASET_CONFIGS.get(dataset_name)
    if not config:
        raise ValueError(f"Dataset '{dataset_name}' is not supported!")

    DATASET_DIR = config["DATASET_DIR"]
    OUTPUT_DIR = config["OUTPUT_DIR"]
    METADATA_FILE = config["METADATA_FILE"]
    image_column = config["image_column"]
    diagnosis_column = config["diagnosis_column"]
    valid_classes = config["valid_classes"]
    concepts = config["concepts"]

    # === STEP 1: Load Metadata ===
    df = pd.read_csv(METADATA_FILE)
    df[image_column] = df[image_column].str.strip()  # Normalize filenames

    # === STEP 2: Filter Dataset to Keep Only Valid Classes ===
    class_mapping = {v: k for k, values in valid_classes.items() for v in values}
    df = df[df[diagnosis_column].isin(class_mapping)]  # Keep only valid classes
    df[diagnosis_column] = df[diagnosis_column].map(class_mapping)  # Relabel classes
    if dataset_name == "Derm7pt":
        # === ✅ STEP 3: Filter Images Based on Concepts (Without Changing Labels) ===
        df["has_valid_concepts"] = df[concepts].apply(lambda row: any(row != "absent"), axis=1)
        df = df[df["has_valid_concepts"]]  # Keep only images that have at least one valid concept

        # === STEP 4: Load Predefined Splits (Ensure Filtered Indices Exist) ===
        train_indexes = list(set(pd.read_csv(config["train_split_file"])["indexes"]) & set(df.index))
        val_indexes = list(set(pd.read_csv(config["val_split_file"])["indexes"]) & set(df.index))
        test_indexes = list(set(pd.read_csv(config["test_split_file"])["indexes"]) & set(df.index))

        # Select only valid filtered samples for each split
        train_df = df.loc[train_indexes]
        val_df = df.loc[val_indexes]
        test_df = df.loc[test_indexes]

    elif dataset_name == "SkinCon":
        valid_concepts = [c for c in concepts if df[c].sum() >= 50]
        df = df[df[valid_concepts].apply(lambda row: any(row == 1), axis=1)]

        train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df[diagnosis_column], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df[diagnosis_column], random_state=42)
    print(f"📂 Filtered dataset: {len(df)} images remain after disease & concept filtering.")
    print(f"🔹 Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    # === STEP 5: Ensure Image Paths ===
    
    if dataset_name == "Derm7pt":
        all_images = glob.glob(os.path.join(DATASET_DIR, "**", "*.*"), recursive=True)
        image_files = {os.path.relpath(img, DATASET_DIR).replace("\\", "/"): img for img in all_images}
    else:
        all_images = glob.glob(os.path.join(DATASET_DIR, "*.*"))
        image_files = {os.path.splitext(os.path.basename(img))[0].lower(): img for img in all_images}
    print(f"✅ Found {len(image_files)} images in {DATASET_DIR}")

    # === STEP 6: Move Images to Train, Validation, Test Folders ===
    def move_images(df_subset, split_name):
        missing_images = []
        for _, row in df_subset.iterrows():
            img_filename = str(row[image_column]).strip()
            matched_file = next(
                (image_files[f] for f in image_files if os.path.splitext(f)[0].lower() == os.path.splitext(img_filename)[0].lower()),
                None
            )

            if matched_file:
                dest = os.path.join(OUTPUT_DIR, split_name, row[diagnosis_column])
                os.makedirs(dest, exist_ok=True)
                shutil.copy(matched_file, os.path.join(dest, os.path.basename(matched_file)))
            else:
                missing_images.append(img_filename)

        if missing_images:
            print(f"⚠️ Warning: {len(missing_images)} images not found.")
            with open(f"missing_images_{dataset_name}.txt", "w") as f:
                for img in missing_images:
                    f.write(img + "\n")

    move_images(train_df, "train")
    move_images(val_df, "validation")
    move_images(test_df, "test")

    print(f"✅ Dataset '{dataset_name}' processing complete!\n")

def process_concept_images(dataset_name, df):
    """Extracts and organizes concept images into Derm7pt_concepts/."""
    concept_output_dir = f"datasets/{dataset_name}_concepts"
    os.makedirs(concept_output_dir, exist_ok=True)

    concepts = DATASET_CONFIGS[dataset_name]["concepts"]
    for concept in concepts:
        concept_dir = os.path.join(concept_output_dir, concept)
        os.makedirs(concept_dir, exist_ok=True)

        # Select images where this concept is NOT 'absent'
        concept_images = df[df[concept] != "absent"]["derm"].tolist() if dataset_name == "Derm7pt" else df[df[concept] == 1]["ImageID"].tolist()

        for img in concept_images:
            img_path = os.path.join(DATASET_CONFIGS[dataset_name]["DATASET_DIR"], img)
            if os.path.exists(img_path):
                shutil.copy(img_path, os.path.join(concept_dir, os.path.basename(img_path)))

    print(f"✅ Concept dataset '{dataset_name}_concepts' created successfully!")

# === RUN SCRIPT ===
if __name__ == "__main__":
    for dataset in ["Derm7pt", "SkinCon"]:
        process_dataset(dataset)
        df = pd.read_csv(DATASET_CONFIGS[dataset]["METADATA_FILE"])
        process_concept_images(dataset, df)