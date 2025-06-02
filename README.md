**Capstone Project: Dual-Task X-Ray Analysis**

A PyTorch-based deep learning pipeline for simultaneous bone fracture detection (binary classification) and multi-label chest disease classification on X-ray images. This repository contains all code, trained model weights, sample data mappings, training notebooks, and a simple inference app.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Repository Structure](#repository-structure)
4. [Setup & Installation](#setup--installation)
5. [Usage](#usage)

   * [1. Data Preparation](#1-data-preparation)
   * [2. Training & Evaluation](#2-training--evaluation)
   * [3. Inference App](#3-inference-app)
6. [File Descriptions](#file-descriptions)
7. [Model Checkpoints](#model-checkpoints)
8. [Requirements](#requirements)
9. [Acknowledgments & References](#acknowledgments--references)

---

## Project Overview

This capstone implements a **dual-task DenseNet** model that ingests X-ray images and outputs:

* **Binary output**: fracture vs. no-fracture (on bone X-rays).
* **Multi-label output**: presence/absence of 14+ chest conditions (e.g., Atelectasis, Effusion, etc.) on chest X-rays.

Key ideas:

* **Shared feature encoder** (DenseNet backbone) with two separate “heads.”
* **CLAHE-based preprocessing** for contrast enhancement.
* **Fracture-specific augmentation** (rotation, flips, random crops) to balance classes.
* **Balanced batch sampling** to mitigate class imbalance.
* **Custom loss functions**: Binary Cross-Entropy for fractures; BCE-with-Logits for multi-label chest conditions.
* **Web-based inference** via a simple Flask (or Streamlit) app.

---

## Features

* End-to-end Jupyter notebooks showing data loading, preprocessing, training loops, and evaluation metrics (accuracy, precision, recall, F1, AUC).
* Model definition and helper functions separated into modular Python scripts (`mod.py`, `func.py`).
* Precomputed **label mapping** JSON for chest conditions (`chest_condition_map.json`).
* Two trained model weights (`model_epoch_8.pth`, `model_epoch_12.pth`) for quick inference.
* A lightweight inference app (`app.py`) that lets you upload a bone or chest X-ray image and get real-time predictions.
* Step-by-step instructions in `step.txt` guiding through data preparation and training.

---

## Repository Structure

```
Capstone_project/
├── .gitignore
├── Archive.zip
├── app.py
├── cap.ipynb
├── capstone_trials.ipynb
├── capy.ipynb
├── chest_condition_map.json
├── func.py
├── mod.py
├── model_epoch_12.pth
├── model_epoch_8.pth
├── requirements.txt
├── step.txt
└── capstone.key
```

* **.gitignore**    : Contains files/directories to exclude from Git (e.g., `__pycache__/`, large checkpoints).
* **Archive.zip**   : (Optional) Any original data snapshots or backup files.
* **app.py**      : Inference script (Flask/Streamlit) for uploading images and viewing predictions.
* **cap.ipynb**    : Main training notebook (data loading → preprocessing → model training → evaluation).
* **capstone\_trials.ipynb** : Experimental notebook showing hyperparameter sweeps and trial runs.
* **capy.ipynb**    : Alternate Jupyter notebook (early-stage prototyping).
* **chest\_condition\_map.json** : Maps numeric indices ↔ chest condition names (e.g., `"0": "Atelectasis", ...`).
* **func.py**      : Utility functions (data transforms, custom losses, dataset class definitions, balanced sampler).
* **mod.py**      : Model definition (`DualHeadDenseNet`), plus helper functions to load pretrained weights.
* **model\_epoch\_8.pth** : Trained checkpoint after epoch 8 (best chest AUC).
* **model\_epoch\_12.pth** : Trained checkpoint after epoch 12 (best fracture F1).
* **requirements.txt**  : List of required Python packages.
* **step.txt**      : Text file outlining the high-level steps to reproduce experiments end to end.
* **capstone.key**    : (If applicable) any private key or configuration for data decryption.

---

## Setup & Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/geeknoobie/Capstone_project.git
   cd Capstone_project
   ```

2. **Create a new virtual environment** (recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate          # macOS/Linux
   venv\Scripts\activate             # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   > **Note:**
   >
   > * Make sure you have PyTorch installed with CUDA support if you plan to train on GPU.
   > * If you are on macOS with M-series, set `torch.device("mps")` manually in notebooks.

4. **Download or Prepare Data**

   * Place your Chest X-ray dataset (e.g., NIH ChestX-ray14) and Bone Fracture dataset in a directory structure like:

     ```
     data/
     ├── chest_xray/
     │   ├── train/
     │   ├── val/
     │   └── test/
     └── bone_fracture/
         ├── train/
         ├── val/
         └── test/
     ```
   * Update file paths in `cap.ipynb` (and/or in `func.py`) so that `DATA_ROOT = "./data"` (or whichever path you choose).

---

## Usage

### 1. Data Preparation

1. **Preprocessing with CLAHE**

   * In `func.py`, you will find a `clahe_transform` applied to each training image.
   * Modify the transform if your images are in DICOM format: first convert to PNG/JPEG, then apply CLAHE.

2. **Label Mapping**

   * `chest_condition_map.json` maps indices to disease names.
   * If using a different header order, adjust keys in the JSON accordingly.

3. **Custom Dataset & Balanced Sampler**

   * `func.py` defines `MultiTaskDataset` that returns `(image, fracture_label, chest_labels_list)`.
   * A `BalancedSampler` ensures each batch has roughly equal positive/negative fracture examples.

---

### 2. Training & Evaluation

1. **Open the main notebook**

   ```bash
   jupyter notebook cap.ipynb
   ```
2. **Walk through the cells**

   * **Data Loaders**: Initializes train/validation/test DataLoaders with appropriate transforms.
   * **Model Instantiation**:

     ```python
     from mod import DualHeadDenseNet
     model = DualHeadDenseNet(num_chest_labels=14, pretrained=True)
     ```
   * **Loss Functions**:

     ```python
     from func import fracture_loss_fn, chest_loss_fn
     ```
   * **Optimizer & Scheduler**:

     ```python
     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
     ```
   * **Training Loop**: Runs for N epochs, logging:

     * Fracture head: accuracy, precision, recall, F1, AUC
     * Chest head: per-disease accuracy, AUC, mAP (mean average precision)
   * **Save Best Checkpoints**:

     ```python
     torch.save(model.state_dict(), "model_epoch_{epoch}.pth")
     ```
3. **Review `capstone_trials.ipynb`**

   * Contains hyperparameter sweep experiments (e.g., different learning rates, batch sizes, augmentations).
   * Use as a template to run your own trials.

---

### 3. Inference App

1. **Launch the Flask/Streamlit app**

   ```bash
   python app.py
   ```

   * If using Flask, it will be served at `http://127.0.0.1:5000/`.
   * If using Streamlit, use:

     ```bash
     streamlit run app.py
     ```

2. **Upload an X-ray image**

   * Choose either a bone X-ray or a frontal chest X-ray.
   * The server will automatically detect image dimensions:

     * If it looks like a bone (e.g., wrist, hand), it runs the fracture head → outputs: “Fracture” or “No Fracture” with confidence score.
     * If it looks like a chest X-ray, it runs the chest head → outputs a list of detected conditions, each with a confidence percentage.

3. **View Results**

   * Example output:

     ```
     Fracture: 0.93 (93%)  
     Chest Conditions:  
       – Pneumonia: 0.12 (12%)  
       – Atelectasis: 0.56 (56%)  
       – Effusion: 0.40 (40%)  
       … (all 14 conditions, sorted by probability)
     ```
   * The UI also shows Grad-CAM heatmaps (if implemented) overlaying areas of highest model activation.

---

## File Descriptions

| File / Folder                                                                  | Description                                                                                                                                           |
| ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app.py`                                                                       | Inference server (Flask or Streamlit) for uploading an X-ray and receiving predictions.                                                               |
| `cap.ipynb`                                                                    | Main Jupyter notebook: end-to-end training, validation, and testing of dual-head model.                                                               |
| `capstone_trials.ipynb`                                                        | Experimentation notebook: hyperparameter sweeps, ablation studies, data augmentation trials.                                                          |
| `capy.ipynb`                                                                   | Early prototype notebook (data exploration, feature testing).                                                                                         |
| `chest_condition_map.json`                                                     | JSON mapping of class indices ↔ human-readable chest condition names (e.g., `{ "0": "Atelectasis", … }`).                                             |
| `func.py`                                                                      | Utility functions:                                                                                                                                    |
| • Dataset class (`MultiTaskDataset`)                                           |                                                                                                                                                       |
| • CLAHE and augmentation transforms                                            |                                                                                                                                                       |
| • Customized Loss functions (`fracture_loss_fn`, `chest_loss_fn`)              |                                                                                                                                                       |
| • Balanced batch sampler (`BalancedSampler`)                                   |                                                                                                                                                       |
| • Evaluation utilities (AUC, F1, per-label metrics).                           |                                                                                                                                                       |
| `mod.py`                                                                       | Model definition:                                                                                                                                     |
| • `DualHeadDenseNet`: Shared DenseNet encoder + two separate classifier heads. |                                                                                                                                                       |
| • `load_pretrained_weights()` helper.                                          |                                                                                                                                                       |
| `model_epoch_8.pth`                                                            | Checkpoint saved after epoch 8 (highest chest classification validation AUC).                                                                         |
| `model_epoch_12.pth`                                                           | Checkpoint saved after epoch 12 (highest fracture detection validation F1).                                                                           |
| `requirements.txt`                                                             | List of Python dependencies (e.g., `torch`, `torchvision`, `numpy`, `opencv-python`, `pillow`, `scikit-learn`, `matplotlib`, `streamlit` or `flask`). |
| `step.txt`                                                                     | Plain-text instructions outlining the exact pipeline stages (data download → preprocessing → training → inference).                                   |
| `Archive.zip`                                                                  | (Optional) Backup of raw or intermediate datasets, older model versions, or original data splits.                                                     |
| `capstone.key`                                                                 | (Optional) Private key or credential used to decrypt a proprietary dataset (ensure not to commit to public repos).                                    |
| `.gitignore`                                                                   | Git ignore patterns for Python caches, IDE config, large model files, etc.                                                                            |

---

## Model Checkpoints

* **`model_epoch_8.pth`**

  * Achieved best validation AUC on the chest disease head (≈ 0.75 – 0.80 average across all 14 conditions).
  * Binary fracture head was still improving—saved for comparison.

* **`model_epoch_12.pth`**

  * Achieved best binary fracture detection F1 (≈ 0.84) on validation set.
  * Chest head maintained competitive AUC (≈ 0.74), with slight overfitting starting.

> **Tip:**
> If you want to fine-tune from an earlier checkpoint, load either of these weights in your training script:
>
> ```python
> model = DualHeadDenseNet(num_chest_labels=14, pretrained=False)
> model.load_state_dict(torch.load("model_epoch_12.pth"))
> ```

---

## Requirements

Below is a minimal example of `requirements.txt`. Your local setup may vary if using GPU or M-series Mac:

```
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
opencv-python>=4.5.5.64
Pillow>=8.4.0
scikit-learn>=1.0.2
matplotlib>=3.5.0
tqdm>=4.62.0
pandas>=1.3.5
```

If using **Flask** for the inference app:

```
Flask>=2.0.3
Werkzeug>=2.0.3
```

If using **Streamlit** instead:

```
streamlit>=1.12.0
```

---

## Acknowledgments & References

* **NIH ChestX-ray14** dataset (Shenzhen Hospital) was used for multi-label chest classification.
* **MURA (Musculoskeletal Radiographs)** or **Bone Fracture Dataset** was used for binary fracture detection.
* Adapted DenseNet architecture from:

  > Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). Densely Connected Convolutional Networks (DenseNet). *CVPR*.
* CLAHE preprocessing reference:

  > Pizer, S. M., Amburn, E. P., Austin, J. D., et al. (1987). Adaptive Histogram Equalization and Its Variations. *Computer Vision, Graphics, and Image Processing, 39*(3), 355–368.

---

**Feel free to raise an issue if you encounter errors, or submit a pull request to improve functionality.**
Enjoy exploring dual-task medical image classification!
