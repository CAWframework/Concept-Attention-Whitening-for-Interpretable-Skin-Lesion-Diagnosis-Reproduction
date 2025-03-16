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
        "image_column": "ImageID",
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
    df[image_column] = df[image_column].str.strip()

    # === STEP 2: Filter Dataset to Keep Only Valid Classes ===
    class_mapping = {v: k for k, values in valid_classes.items() for v in values}
    df = df[df[diagnosis_column].isin(class_mapping)]
    df[diagnosis_column] = df[diagnosis_column].map(class_mapping)

    if dataset_name == "Derm7pt":
        df["has_valid_concepts"] = df[concepts].apply(lambda row: any(row != "absent"), axis=1)
        df = df[df["has_valid_concepts"]]

        # === STEP 3: Load Predefined Splits ===
        train_df = df.loc[df.index.isin(pd.read_csv(config["train_split_file"])["indexes"])]
        val_df = df.loc[df.index.isin(pd.read_csv(config["val_split_file"])["indexes"])]
        test_df = df.loc[df.index.isin(pd.read_csv(config["test_split_file"])["indexes"])]

    elif dataset_name == "SkinCon":
        valid_concepts = [c for c in concepts if df[c].sum() >= 50]
        df = df[df[valid_concepts].apply(lambda row: any(row == 1), axis=1)]

        train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df[diagnosis_column], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df[diagnosis_column], random_state=42)

    print(f"📂 Filtered dataset: {len(df)} images remain after disease & concept filtering.")
    print(f"🔹 Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

    # === STEP 4: Move Images to Train, Validation, Test Folders ===
    def move_images(df_subset, split_name):
        for _, row in df_subset.iterrows():
            img_path = os.path.join(DATASET_DIR, row[image_column])
            dest = os.path.join(OUTPUT_DIR, split_name, row[diagnosis_column])
            os.makedirs(dest, exist_ok=True)
            shutil.copy(img_path, os.path.join(dest, os.path.basename(img_path)))

    move_images(train_df, "train")
    move_images(val_df, "validation")
    move_images(test_df, "test")

    print(f"✅ Dataset '{dataset_name}' processing complete!\n")
    return df
def process_concept_images(dataset_name, df_filtered):
    """Extracts and organizes concept images into {dataset_name}_concepts/ and saves labels to a CSV."""
    concept_output_dir = f"datasets/{dataset_name}_concepts"
    os.makedirs(concept_output_dir, exist_ok=True)

    df_filtered.columns = df_filtered.columns.str.strip()  # ✅ Ensure no extra spaces in column names

    dataset_dir = DATASET_CONFIGS[dataset_name]["DATASET_DIR"]
    image_column = DATASET_CONFIGS[dataset_name]["image_column"]
    df_filtered = df_filtered.rename(columns={image_column: "ImageID"})

    # ✅ Recursively find all images in dataset_dir (subfolders included)
    all_images = {os.path.basename(f).lower(): f for f in glob.glob(f"{dataset_dir}/**/*.*", recursive=True)}

    # ✅ Normalize filenames in DataFrame (to avoid case mismatches)
    df_filtered["ImageID"] = df_filtered["ImageID"].str.lower()

    # ✅ Apply dataset-specific processing
    if dataset_name == "Derm7pt":
        # ✅ Convert multi-class concepts to binary
        df_filtered["PN_TYP"] = df_filtered["pigment_network"].apply(lambda x: 1 if str(x).strip().lower() == "typical" else 0)
        df_filtered["PN_ATP"] = df_filtered["pigment_network"].apply(lambda x: 1 if str(x).strip().lower() == "atypical" else 0)
        df_filtered["STR_REG"] = df_filtered["streaks"].apply(lambda x: 1 if str(x).strip().lower() == "regular" else 0)
        df_filtered["STR_IR"] = df_filtered["streaks"].apply(lambda x: 1 if str(x).strip().lower() == "irregular" else 0)
        df_filtered["PIG_REG"] = df_filtered["pigmentation"].apply(lambda x: 1 if "regular" in str(x).strip().lower() else 0)
        df_filtered["PIG_IR"] = df_filtered["pigmentation"].apply(lambda x: 1 if "irregular" in str(x).strip().lower() else 0)
        df_filtered["RS_PRS"] = df_filtered["regression_structures"].apply(lambda x: 1 if x != "absent" else 0)
        df_filtered["DaG_REG"] = df_filtered["dots_and_globules"].apply(lambda x: 1 if str(x).strip().lower() == "regular" else 0)
        df_filtered["DaG_IR"] = df_filtered["dots_and_globules"].apply(lambda x: 1 if str(x).strip().lower() == "irregular" else 0)
        df_filtered["BWV_PRS"] = df_filtered["blue_whitish_veil"].apply(lambda x: 1 if str(x).strip().lower() == "present" else 0)
        df_filtered["VS_REG"] = df_filtered["vascular_structures"].apply(lambda x: 1 if any(word in str(x).strip().lower() for word in ["arborizing", "hairpin", "comma"]) else 0)
        df_filtered["VS_IR"] = df_filtered["vascular_structures"].apply(lambda x: 1 if any(word in str(x).strip().lower() for word in ["dotted", "linear irregular", "within regression", "wreath"]) else 0)


        # ✅ Extract only filenames, remove subdirectories (e.g., NEL/Nel026.jpg → Nel026.jpg)
        df_filtered["ImageID"] = df_filtered["ImageID"].apply(lambda x: os.path.basename(str(x)))

        # ✅ Save concept labels for Derm7pt
        concept_label_file = os.path.join(concept_output_dir, "concept_labels.csv")
        df_concepts = df_filtered[["ImageID", "PN_TYP", "PN_ATP", "STR_REG", "STR_IR", "PIG_REG", "PIG_IR", "RS_PRS", 
                                   "DaG_REG", "DaG_IR", "BWV_PRS", "VS_REG", "VS_IR"]]
        df_concepts.to_csv(concept_label_file, index=False)
        print(f"✅ Concept label CSV saved to: {concept_label_file}")

    elif dataset_name == "SkinCon":
        # ✅ Keep only concept columns that exist in SkinCon metadata
        concepts = DATASET_CONFIGS[dataset_name]["concepts"]
        available_concepts = [c for c in concepts if c in df_filtered.columns]

        # ✅ Save concept labels for SkinCon
        concept_label_file = os.path.join(concept_output_dir, "concept_labels.csv")
        df_concepts = df_filtered[["ImageID"] + available_concepts]  # Keep only relevant columns
        df_concepts.to_csv(concept_label_file, index=False)
        print(f"✅ Concept label CSV saved to: {concept_label_file}")

    # ✅ Process images for each concept
    for concept in df_concepts.columns[1:]:  # Skip the image column
        concept_dir = os.path.join(concept_output_dir, concept)
        os.makedirs(concept_dir, exist_ok=True)

        # ✅ Use only filtered dataset images (fixing missing files issue)
        concept_images = df_filtered[df_filtered[concept] == 1]["ImageID"].tolist()

        for img in concept_images:
            img_lower = img.lower()  # Convert to lowercase for matching
            if img_lower in all_images:
                shutil.copy(all_images[img_lower], os.path.join(concept_dir, img))
            else:
                print(f"⚠️ Warning: Image not found - {img}")
    print(f"✅ Concept dataset '{dataset_name}_concepts' created successfully!")

# === RUN SCRIPT ===
if __name__ == "__main__":
    for dataset in ["Derm7pt", "SkinCon"]:
        # ✅ First, process and filter the dataset
        df_filtered = process_dataset(dataset)  # Ensure process_dataset() returns the filtered DataFrame

        # ✅ Now, pass only the **filtered** dataset to process_concept_images()
        process_concept_images(dataset, df_filtered)

