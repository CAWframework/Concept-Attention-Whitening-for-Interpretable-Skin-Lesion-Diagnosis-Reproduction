# Concept-Attention Whitening for Interpretable Skin Lesion Diagnosis – Reproduction

## 📖 Paper Details

- **Title:** Concept-Attention Whitening for Interpretable Skin Lesion Diagnosis  
- **Authors:** Junlin Hou, Jilan Xu, Hao Chen  
- **Original Paper:** [arXiv:2404.05997](https://arxiv.org/abs/2404.05997)

---

## 📌 Overview

This repository contains a **reproduction** of the **Concept-Attention Whitening (CAW)** framework, designed to improve the interpretability of deep learning models for skin lesion classification. We closely follow the methodology described in the original paper and compare our results against those reported by the authors.

---

## 📂 Repository Structure

```
Concept-Attention-Whitening-for-Interpretable-Skin-Lesion-Diagnosis-Reproduction/
├── notebooks/           # Jupyter notebooks 
├── models/              # Resnet and CAW moddel
├── src/                 # Python scripts for training, dataset processing, evaluation
├── results/             # Figures and logs
├── requirements.txt     # Dependencies to run the code
└── report/              # Final LaTeX report and related files
```

---

## ⚙️ Installation & Setup

### 🔸 Step 1: Clone the Repository

```bash
git clone https://github.com/YourUsername/Concept-Attention-Whitening-for-Interpretable-Skin-Lesion-Diagnosis-Reproduction.git
cd Concept-Attention-Whitening-for-Interpretable-Skin-Lesion-Diagnosis-Reproduction
```

### 🔸 Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔸 Step 3: Data Preparation

**Download Datasets:**

- **Derm7pt**: [Derm7pt Official Website](https://derm.cs.sfu.ca/Welcome.html)
- **SkinCon**: [SkinCon Official Website](https://skincon-dataset.github.io/)

Place the downloaded datasets inside the `datasets/` directory.

**Preprocess Datasets:**

```bash
python src/preprocess.py
```
**Dataset Structure**

This project uses two primary datasets: Derm7pt and SkinCon. These datasets include images, metadata, and concept annotations that help in training the model. After running the preprocessing script, the datasets will be organized as follows:

```
datasets/
├── Derm7pt/
│   ├── images/                       # Contains dermoscopic images
│   ├── meta/                         # Metadata files for the dataset
│   ├── test/                         # Test set
│   ├── train/                        # Training set
│   ├── validation/                   # Validation set
│   ├── Derm7pt_filtered.csv          # Filtered metadata CSV
│   ├── README.txt                    # Project readme file
├── Derm7pt_concepts/                 # Concept files (clinical concepts)
│   ├── BWV_PRS/                      # Concept folder for "BWV_PRS"
│   ├── DaG_IR/                       # Concept folder for "DaG_IR"
│   ├── PIG_IR/                       # Concept folder for "PIG_IR"
│   ├── ...                           # Other concept folders
│   └── concept_labels.csv            # Contains concept annotations for each image
├── SkinCon/
│   ├── images/                        # Contains skin disease images
│   ├── test/                          # Test set
│   ├── train/                         # Training set
│   ├── validation/                    # Validation set
│   ├── SkinCon_filtered.csv           # Filtered metadata CSV
│   ├── annotations_fitzpatrick17k.csv # Annotations for the SkinCon dataset
├── SkinCon_concepts/                  # Concept folders for SkinCon 
│   ├── Atrophy/                       # Concept folder for "Atrophy"
│   ├── Black/                         # Concept folder for "Black"
│   ├── Bulla/                         # Concept folder for "Bulla"
│   ├── ...                            # Other concept folders
│   └── concept_labels.csv             # Concept annotations for SkinCon
```
---

## 🚀 Training the Model

Run the following scripts to train the CAW model on each dataset:

**Derm7pt Dataset (ResNet18 Backbone)**

```bash
python src/train.py --dataset Derm7pt --model resnet18 --use_caw --epochs 100 --lr 2e-3 --batch_size 64 --output_dir checkpoints
```

**SkinCon Dataset (ResNet50 Backbone)**

```bash
python src/train.py --dataset SkinCon --model resnet50 --use_caw --epochs 100 --lr 2e-3 --batch_size 64 --output_dir checkpoints
```

---

## 📊 Evaluating the Model

Use the following commands to evaluate the trained models:

**Evaluate Derm7pt Model**

```bash
python src/evaluate.py --dataset Derm7pt --model resnet18 --use_caw --checkpoint checkpoints/Derm7pt_resnet18_caw.pth --runs 3
```

**Evaluate SkinCon Model**

```bash
python src/evaluate.py --dataset SkinCon --model resnet50 --use_caw --checkpoint checkpoints/SkinCon_resnet50_caw.pth --runs 3
```

---

## 📑 Citation

If you find this reproduction useful, please cite the original paper:

```bibtex
@misc{hou2024conceptattention,
  title={Concept-Attention Whitening for Interpretable Skin Lesion Diagnosis},
  author={Junlin Hou and Jilan Xu and Hao Chen},
  year={2024},
  eprint={2404.05997},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

---

## 📍 Reference Links

- [Original Paper (arXiv)](https://arxiv.org/abs/2404.05997)
- [Derm7pt Dataset](https://derm.cs.sfu.ca/Welcome.html)
- [SkinCon Dataset](https://skincon-dataset.github.io/)

