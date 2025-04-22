# importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm.notebook import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import os
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchcam.methods import GradCAM
from torchvision.transforms.functional import to_pil_image
from PIL import Image
# function for image processing
def load_and_preprocess_image(image_or_path, target_size=(224, 224), save_flag=False):
    """
    Load and preprocess an X-ray image, with optional saving support.
    Accepts both file paths (str) and raw NumPy arrays.

    Args:
        image_or_path (str or np.ndarray): Path to image or grayscale NumPy array.
        target_size (tuple): Desired size for the processed image (default: 224x224)
        save_flag (bool): If True, saves the image to a processed folder; else returns it.

    Returns:
        np.ndarray or None: Processed image array (if save_flag is False)
    """
    # Handle image input as array or path
    if isinstance(image_or_path, np.ndarray):
        image = image_or_path
    else:
        image = cv2.imread(str(image_or_path), cv2.IMREAD_GRAYSCALE)

    # Validate image loading
    if image is None:
        print(f"Error: Failed to load image: {image_or_path}")
        return None

    try:
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

        # Resize with aspect ratio preservation
        current_height, current_width = image.shape
        scale = min(target_size[0] / current_width, target_size[1] / current_height)
        new_width = int(current_width * scale)
        new_height = int(current_height * scale)
        resized = cv2.resize(image, (new_width, new_height))

        # Pad to final size
        final_image = np.zeros(target_size, dtype=np.float32)
        y_offset = (target_size[0] - new_height) // 2
        x_offset = (target_size[1] - new_width) // 2
        final_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

        # Normalize pixel values to [0, 1]
        final_image = final_image / 255.0

        # Save if needed
        if save_flag and not isinstance(image_or_path, np.ndarray):
            original_folder = os.path.dirname(image_or_path)
            processed_folder = original_folder + '_processed'
            os.makedirs(processed_folder, exist_ok=True)

            base_name = os.path.basename(image_or_path)
            file_name = os.path.splitext(base_name)[0] + '.npy'
            save_path = os.path.join(processed_folder, file_name)
            np.save(save_path, final_image)
            return None

        return final_image

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return None
# function for dataset creation
def create_dataset_df(final_chest_df, chest_processed, fracture_processed, not_fracture_processed, raw = False ):
    """
    Creates a unified DataFrame containing file paths, labels, and dataset types
    for both chest X-ray and fracture datasets. It also performs single-hot
    encoding for chest condition labels.

    Args:
        final_chest_df (pd.DataFrame): DataFrame containing chest X-ray image
                                        information, including 'Image Index' and
                                        'Finding Labels'. This DataFrame is assumed
                                        to be the processed version of chest labels.
        chest_processed (str): Path to the directory containing processed chest
                               X-ray image data (presumably in .npy format).
        fracture_processed (str): Path to the directory containing processed
                                  fracture image data (.npy format).
        not_fracture_processed (str): Path to the directory containing processed
                                      non-fracture image data (.npy format).

    Returns:
        tuple: A tuple containing:
            - combined_df (pd.DataFrame): A DataFrame with columns like 'file_path',
              'fracture_label' (for fracture data), 'dataset_type',
              'original_filename' (for chest data), 'chest_labels' (single-hot
              encoded), and 'Finding Labels' (original chest label).
            - condition_to_idx (dict): A dictionary mapping unique chest
              conditions (from 'Finding Labels') to integer indices.
    """

    # --- Processing Fracture Dataset ---
    # here we are creating a list of tuples for fracture images. Each tuple contains:
    # 1. The full path to the processed fracture image file.
    # 2. The label for fracture (1 indicating a fracture).
    # 3. The type of dataset.
    valid_exts = ('.png', '.jpg', '.jpeg', '.JPG','.npy')

    fracture_files = [(os.path.join(fracture_processed, f), 1, 'fracture')
                      for f in os.listdir(fracture_processed) if f.endswith(valid_exts)]
    not_fracture_files = [(os.path.join(not_fracture_processed, f), 0, 'not fracture')
                          for f in os.listdir(not_fracture_processed) if f.endswith(valid_exts)]
    # here we are combining both the lists of fracture and non-fracture file information and creating a DataFrame. The DataFrame has columns 'file_path', 'fracture_label' and 'dataset_type'.
    fracture_df = pd.DataFrame(fracture_files + not_fracture_files,
                               columns=['file_path', 'fracture_label', 'dataset_type'])

    # --- Processing Chest X-ray Dataset ---
    # here we are creating a separate input DataFrame 'final_chest_df' which contains the necessary chest X-ray labels in the relevant column.
    chest_labels_df = final_chest_df

    # here we are Identifying all unique chest condition labels present.
    all_conditions = set(chest_labels_df['Finding Labels'].unique())

    # now we are creating a dictionary that maps each unique chest condition to a unique integer index this is useful as , machine learning task need numerical representation of the labels for functioning. we are also sorting the conditions to ensure consistent indexing.
    condition_to_idx = {condition: idx for idx, condition
                        in enumerate(sorted(all_conditions))}

    # now we are going to define a function to create a single-hot encoded label for a given chest condition string.which means that for each image, only the index corresponding to its specific condition will have a value of 1, while all other indices will be 0.
    def create_single_hot(label_str):
        # Initializing a list of zeros with a length equal to the total number of unique conditions.
        single_hot = [0] * len(condition_to_idx)
        # Setting the element at the index corresponding to the input 'label_str' to 1.
        single_hot[condition_to_idx[label_str]] = 1
        return single_hot

    # now we call this function and apply it to a column named 'Finding Labels' of the chest labels DataFrame. This will result in a newly created  column named 'chest_labels' containing the single-hot encoded representation of each chest condition.
    chest_labels_df['chest_labels'] = chest_labels_df['Finding Labels'].apply(create_single_hot)

    # Here we are creating a list of tuples for chest X-ray images. Each tuple contains:
    # 1. The full path to the processed chest X-ray image file.
    # 2. The original filename of the chest X-ray image.
    # 3. The type of dataset.
    chest_files = [
        (
            os.path.join(chest_processed, f if raw else f.replace('.png', '.npy')),
            f,
            'chest'
        )
        for f in chest_labels_df['Image Index']
    ]

    # now we will create a dataFrame from the list of chest file information, with columns 'file_path', 'original_filename', and 'dataset_type'.
    chest_df = pd.DataFrame(chest_files,
                            columns=['file_path', 'original_filename', 'dataset_type'])

    # here we are merging both the 'chest_df' with the 'chest_labels_df' to add the 'chest_labels' based on the 'original_filename' from 'chest_df' and the 'Image Index' from 'chest_labels_df'. A left merge is used to ensure that
    # all entries in 'chest_df' are kept. this leaves us our final chest_df with all proper labels and necessary data
    chest_df = chest_df.merge(chest_labels_df[['Image Index', 'chest_labels', 'Finding Labels']],
                                left_on='original_filename',
                                right_on='Image Index',
                                how='left')

    # --- Combining Datasets ---
    # finally we concatenate the 'fracture_df' and 'chest_df' to create a single unified DataFrame. while setting 'ignore_index=True' which resets the index of the resulting DataFrame.
    combined_df = pd.concat([fracture_df, chest_df], ignore_index=True)

    # Return the combined DataFrame and the mapped dictionary of chest conditions.
    return combined_df, condition_to_idx
# function to create augmented data
def augment_array(arr, aug_type):
    if aug_type == 'rotate':
        return cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
    elif aug_type == 'flip_h':
        return cv2.flip(arr, 1)
    elif aug_type == 'flip_v':
        return cv2.flip(arr, 0)
    elif aug_type == 'contrast':
        return np.clip(arr * 1.5, 0, 255).astype(np.uint8)
    elif aug_type == 'blur':
        return cv2.GaussianBlur(arr, (5, 5), 0)
    else:
        return arr
# Function to split dataset in train validation and test sets
def split_datasets(dataset_df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    This function will split the combined dataset (fracture and chest X-ray data) into training, validation, and test sets,
    maintaining class balance, especially for 'No Finding' vs. 'Findings' in chest X-ray data.

    Args:
        dataset_df (pd.DataFrame): Combined DataFrame containing both fracture and chest X-ray datasets.
        train_size (float): Proportion of the dataset to include in the training split (default: 0.7).
        val_size (float): Proportion of the dataset to include in the validation split (default: 0.15).
        test_size (float): Proportion of the dataset to include in the test split (default: 0.15).
        random_state (int): Seed used by the random number generator for reproducible splits (default: 42).

    Returns:
        tuple: A tuple containing three pandas DataFrames: (train_data, val_data, test_data).
               Each DataFrame represents the training, validation, and test splits, respectively.
    """

    # --- Separate Fracture and Chest Datasets ---
    # We're separating the combined DataFrame into fracture and chest X-ray DataFrames based on 'dataset_type'.
    # This helps us prevent class imbalance, especially since the chest X-ray data is much larger than the fracture data.
    fracture_data = dataset_df[dataset_df['dataset_type'].isin(['fracture', 'not fracture'])]
    chest_data = dataset_df[dataset_df['dataset_type'] == 'chest']

    # --- Splitting Fracture Dataset ---
    # 1. we will start by splitting the fracture data. We'll use a temporary set here which we'll then divide into validation and test sets.
    # 2. we also are using the stratify parameter to make sure we keep a balanced representation of fracture/not fracture classes.
    fracture_train, fracture_temp = train_test_split(
        fracture_data,
        train_size=train_size,
        stratify=fracture_data['dataset_type'],  # Stratifying by fracture/not fracture
        random_state=random_state
    )

    # Step 2: Splitting the temporary fracture set into validation and test sets.
    # 3. now we will calculate the relative validation size and split the temporary set to form the validation and test sets.
    relative_val_size = val_size / (val_size + test_size)
    fracture_val, fracture_test = train_test_split(
        fracture_temp,
        train_size=relative_val_size,
        stratify=fracture_temp['dataset_type'],  # Stratifying by fracture/not fracture
        random_state=random_state
    )

    # --- Splitting Chest X-ray Dataset while Maintaining Balance ---
    # 1. now we are going to split the chest X-ray data, but we'll do it a bit differently to ensure balance across the various classes.
    # 2. we are going to split it on the basis of No Finding vs. other findings to maintain a good mix in each set as the no findings data have a very high percentage in there which can make the selection based and un balanced .
    no_finding_data = chest_data[chest_data['Finding Labels'] == 'No Finding']
    findings_data = chest_data[chest_data['Finding Labels'] != 'No Finding']

    # 3. splitting the No Finding data into training and temporary data
    no_finding_train, no_finding_temp = train_test_split(
        no_finding_data,
        train_size=train_size,
        random_state=random_state
    )

    # 4. splitting the temporary data into validation and test data
    no_finding_val, no_finding_test = train_test_split(
        no_finding_temp,
        train_size=relative_val_size,
        random_state=random_state
    )

    # 5. splitting the Finding data into training and temporary data
    findings_train, findings_temp = train_test_split(
        findings_data,
        train_size=train_size,
        random_state=random_state
    )
    # 6. splitting the temporary data into validation and test data
    findings_val, findings_test = train_test_split(
        findings_temp,
        train_size=relative_val_size,
        random_state=random_state
    )

    # 7. now we merge both the data to get our combined data for the chest data
    chest_train = pd.concat([no_finding_train, findings_train])
    chest_val = pd.concat([no_finding_val, findings_val])
    chest_test = pd.concat([no_finding_test, findings_test])

    # --- Combine Splits ---
    # 1.now we will put all the data together and combine the fracture and chest splits to obtain our final training, validation and test sets.
    train_data = pd.concat([fracture_train, chest_train])
    val_data = pd.concat([fracture_val, chest_val])
    test_data = pd.concat([fracture_test, chest_test])

    # --- Print Final Dataset Sizes ---
    # 1. as a last step we will print the dataset sizes to check Just to double-check, let's print the final sizes of our training, validation, and test sets.
    print("\nAfter splitting:")
    print(f"Train set size: {len(train_data)} (Fracture: {len(fracture_train)}, Chest: {len(chest_train)})")
    print(f"Validation set size: {len(val_data)} (Fracture: {len(fracture_val)}, Chest: {len(chest_val)})")
    print(f"Test set size: {len(test_data)} (Fracture: {len(fracture_test)}, Chest: {len(chest_test)})")

    # --- Return Split Datasets ---
    # 1. finally, we'll return the training, validation, and test DataFrames.
    return train_data, val_data, test_data
# function for creating batches for model
def create_batch_generator(train_data, condition_to_idx, batch_size=32, no_finding_weight=0.5, sz = 224):
    """Creates balanced batches with weighted sampling for 'No Finding'."""

    # Separating data
    # 1. here we are separating the training data into fracture, no-finding chest, and other-findings chest data.
    fracture_data = train_data[train_data['dataset_type'].isin(['fracture', 'not fracture'])]
    chest_data = train_data[train_data['dataset_type'] == 'chest']
    no_finding_data = chest_data[chest_data['Finding Labels'] == 'No Finding']
    findings_data = chest_data[chest_data['Finding Labels'] != 'No Finding']

    # Calculating batch sizes
    # 1. here we are calculating the batch sizes for each dataset type to ensure balance.
    fracture_batch_size = batch_size // 2 # Half of the batch for fracture data.
    no_finding_size = int(fracture_batch_size * no_finding_weight) # Weighted size for no-finding data.
    findings_size = fracture_batch_size - no_finding_size # Remaining size for other-findings data.

    # Calculating number of complete batches
    # 1. We determine the number of complete batches that can be created from the smallest dataset.
    num_batches = min(
        len(fracture_data) // fracture_batch_size,
        len(no_finding_data) // no_finding_size,
        len(findings_data) // findings_size
    )

    # Create random indices
    # 1. here we are creating random permutations of indices for each dataset to shuffle the data.
    fracture_indices = np.random.permutation(len(fracture_data))
    no_finding_indices = np.random.permutation(len(no_finding_data))
    findings_indices = np.random.permutation(len(findings_data))

    for batch_idx in tqdm(range(num_batches), desc='Generating Batches'):
        # Get indices for this batch
        # 1. We select the indices for the current batch from the shuffled indices.
        fracture_batch_idx = fracture_indices[batch_idx * fracture_batch_size:(batch_idx + 1) * fracture_batch_size]
        no_finding_batch_idx = no_finding_indices[batch_idx * no_finding_size:(batch_idx + 1) * no_finding_size]
        findings_batch_idx = findings_indices[batch_idx * findings_size:(batch_idx + 1) * findings_size]

        # Get the data
        # 1. here we are retrieving the actual data rows from the DataFrames using the selected indices.
        fracture_batch = fracture_data.iloc[fracture_batch_idx]
        no_finding_batch = no_finding_data.iloc[no_finding_batch_idx]
        findings_batch = findings_data.iloc[findings_batch_idx]

        # Combine all data
        # 1. here we are concatenating the fracture, no-finding, and other-findings data to form the batch.
        batch = pd.concat([fracture_batch, no_finding_batch, findings_batch])
        batch_images = []
        batch_labels = []

        for _, row in batch.iterrows():
            # Load the original image
            # 1. here we are loading the image data from the file path and reshape it.
            img = np.load(row['file_path'])
            img = img.reshape(1, sz, sz)

            batch_images.append(img)

            # Use original working label format
            # 1. here we are creating the label vector based on the dataset type.
            if row['dataset_type'] in ['fracture', 'not fracture']:
                # 2. For fracture data, the label is [fracture_label, 0, 0, ...].
                label = np.array([float(row['fracture_label'])] + [0.0] * len(condition_to_idx))
            else:
                # 2. For chest data, the label is [0, chest_label_vector].
                chest_label_vector = row['chest_labels']
                label = np.array([0.0] + chest_label_vector)
            batch_labels.append(label)

        # Stack the images and labels into numpy arrays.
        batch_images = np.stack(batch_images)
        batch_labels = np.stack(batch_labels)

        # Yield the batch.
        yield batch_images, batch_labels
# function to train our model
def train_model(model, batch_generator, learning_rate=0.001, optimizer=None, scheduler=None,device='mps'):
    model.train()
    model.to(device)  # Move model to the correct device

    # Simple loss functions (no Focal Loss)
    fracture_criterion = nn.BCEWithLogitsLoss()
    chest_criterion = nn.CrossEntropyLoss()

    fracture_losses = []
    chest_losses = []

    for batch_images, batch_labels in tqdm(batch_generator, desc="Training"):
        batch_images = torch.tensor(batch_images, dtype=torch.float32, device=device)
        batch_labels = torch.tensor(batch_labels, dtype=torch.float32, device=device)

        optimizer.zero_grad()

        # Forward pass
        fracture_pred, chest_pred = model(batch_images)

        # Labels
        fracture_labels = batch_labels[:, 0].unsqueeze(1)  # First column for fracture
        chest_labels = batch_labels[:, 1:]  # Rest for chest conditions

        # Loss
        fracture_loss = fracture_criterion(fracture_pred, fracture_labels)
        chest_labels_indices = torch.argmax(chest_labels, dim=1)
        chest_loss = chest_criterion(chest_pred, chest_labels_indices)

        # Weighted total loss
        total_loss = 0.3 * fracture_loss + 0.7 * chest_loss
        total_loss.backward()
        optimizer.step()

        fracture_losses.append(fracture_loss.item())
        chest_losses.append(chest_loss.item())

    avg_fracture_loss = sum(fracture_losses) / len(fracture_losses)
    avg_chest_loss = sum(chest_losses) / len(chest_losses)

    print(f"Average Fracture Loss: {avg_fracture_loss:.4f}")
    print(f"Average Chest Loss: {avg_chest_loss:.4f}")

    return avg_chest_loss
# function to validate our model
def validate_model(model, val_generator, device):
    model.eval()
    all_fracture_preds = []
    all_fracture_labels = []
    all_chest_probs = []
    all_chest_labels = []

    with torch.no_grad():
        for batch_images, batch_labels in val_generator:
            batch_images = torch.tensor(batch_images, dtype=torch.float32, device=device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32, device=device)

            fracture_pred, chest_logits = model(batch_images)

            # Fracture
            fracture_labels = batch_labels[:, 0].unsqueeze(1)
            all_fracture_preds.extend(fracture_pred.cpu().numpy())
            all_fracture_labels.extend(fracture_labels.cpu().numpy())

            # Chest
            chest_probs = torch.softmax(chest_logits, dim=1)
            chest_labels = torch.argmax(batch_labels[:, 1:], dim=1)

            all_chest_probs.extend(chest_probs.cpu().numpy())
            all_chest_labels.extend(chest_labels.cpu().numpy())

    # Convert everything to numpy
    fracture_preds = np.array(all_fracture_preds)
    fracture_labels = np.array(all_fracture_labels)
    chest_probs = np.array(all_chest_probs)
    chest_labels = np.array(all_chest_labels)

    # === Fracture Metrics ===
    fracture_pred_binary = (fracture_preds > 0.5).astype(int)
    fracture_accuracy = accuracy_score(fracture_labels, fracture_pred_binary)
    fracture_precision = precision_score(fracture_labels, fracture_pred_binary, zero_division=0)
    fracture_recall = recall_score(fracture_labels, fracture_pred_binary, zero_division=0)
    fracture_f1 = f1_score(fracture_labels, fracture_pred_binary, zero_division=0)
    fracture_auc = roc_auc_score(fracture_labels, fracture_preds)
    fracture_ap = average_precision_score(fracture_labels, fracture_preds)

    # === Chest Metrics ===
    chest_pred_labels = np.argmax(chest_probs, axis=1)
    chest_accuracy = accuracy_score(chest_labels, chest_pred_labels)
    chest_precision = precision_score(chest_labels, chest_pred_labels, average='weighted', zero_division=0)
    chest_recall = recall_score(chest_labels, chest_pred_labels, average='weighted', zero_division=0)
    chest_f1 = f1_score(chest_labels, chest_pred_labels, average='weighted', zero_division=0)

    # ✅ Print summary
    print("\n============ VALIDATION METRICS ============")
    print("\n----- FRACTURE DETECTION -----")
    print(f"Accuracy:  {fracture_accuracy:.4f}")
    print(f"Precision: {fracture_precision:.4f}")
    print(f"Recall:    {fracture_recall:.4f}")
    print(f"F1 Score:  {fracture_f1:.4f}")
    print(f"AUC-ROC:   {fracture_auc:.4f}")
    print(f"Avg Prec:  {fracture_ap:.4f}")

    print("\n----- CHEST CONDITIONS -----")
    print(f"Accuracy:  {chest_accuracy:.4f}")
    print(f"Precision: {chest_precision:.4f}")
    print(f"Recall:    {chest_recall:.4f}")
    print(f"F1 Score:  {chest_f1:.4f}")

    return {
        'fracture_accuracy': fracture_accuracy,
        'fracture_precision': fracture_precision,
        'fracture_recall': fracture_recall,
        'fracture_f1': fracture_f1,
        'fracture_auc': fracture_auc,
        'chest_accuracy': chest_accuracy,
        'chest_precision': chest_precision,
        'chest_recall': chest_recall,
        'chest_f1': chest_f1
    }
# function to test our model
def test_model(model, test_generator, device="mps"):
    """
    Evaluate model for binary fracture and single-label chest classification.
    """
    model.eval()
    all_fracture_preds = []
    all_fracture_labels = []
    all_chest_probs = []
    all_chest_labels = []

    with torch.no_grad():
        for batch_images, batch_labels in test_generator:
            batch_images = torch.tensor(batch_images, dtype=torch.float32, device=device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32, device=device)

            fracture_pred, chest_logits = model(batch_images)

            # Fracture labels
            fracture_labels = batch_labels[:, 0].unsqueeze(1)
            all_fracture_preds.extend(fracture_pred.cpu().numpy())
            all_fracture_labels.extend(fracture_labels.cpu().numpy())

            # Chest prediction
            chest_probs = torch.softmax(chest_logits, dim=1)
            chest_labels = torch.argmax(batch_labels[:, 1:], dim=1)

            all_chest_probs.extend(chest_probs.cpu().numpy())
            all_chest_labels.extend(chest_labels.cpu().numpy())

    # Convert to numpy
    fracture_preds = np.array(all_fracture_preds)
    fracture_labels = np.array(all_fracture_labels)
    chest_probs = np.array(all_chest_probs)
    chest_labels = np.array(all_chest_labels)

    # === Fracture Metrics ===
    fracture_pred_binary = (fracture_preds > 0.5).astype(int)
    fracture_accuracy = accuracy_score(fracture_labels, fracture_pred_binary)
    fracture_precision = precision_score(fracture_labels, fracture_pred_binary, zero_division=0)
    fracture_recall = recall_score(fracture_labels, fracture_pred_binary, zero_division=0)
    fracture_f1 = f1_score(fracture_labels, fracture_pred_binary, zero_division=0)
    fracture_auc = roc_auc_score(fracture_labels, fracture_preds)
    fracture_ap = average_precision_score(fracture_labels, fracture_preds)

    # === Chest Metrics ===
    chest_pred_labels = np.argmax(chest_probs, axis=1)
    chest_accuracy = accuracy_score(chest_labels, chest_pred_labels)
    chest_precision = precision_score(chest_labels, chest_pred_labels, average='weighted', zero_division=0)
    chest_recall = recall_score(chest_labels, chest_pred_labels, average='weighted', zero_division=0)
    chest_f1 = f1_score(chest_labels, chest_pred_labels, average='weighted', zero_division=0)

    # === Print results ===
    print("\n============ TESTING METRICS ============")
    print("\n----- FRACTURE DETECTION -----")
    print(f"Accuracy:  {fracture_accuracy:.4f}")
    print(f"Precision: {fracture_precision:.4f}")
    print(f"Recall:    {fracture_recall:.4f}")
    print(f"F1 Score:  {fracture_f1:.4f}")
    print(f"AUC-ROC:   {fracture_auc:.4f}")
    print(f"Avg Prec:  {fracture_ap:.4f}")

    print("\n----- CHEST CONDITIONS -----")
    print(f"Accuracy:  {chest_accuracy:.4f}")
    print(f"Precision: {chest_precision:.4f}")
    print(f"Recall:    {chest_recall:.4f}")
    print(f"F1 Score:  {chest_f1:.4f}")

    return {
        'fracture_accuracy': fracture_accuracy,
        'fracture_precision': fracture_precision,
        'fracture_recall': fracture_recall,
        'fracture_f1': fracture_f1,
        'fracture_auc': fracture_auc,
        'chest_accuracy': chest_accuracy,
        'chest_precision': chest_precision,
        'chest_recall': chest_recall,
        'chest_f1': chest_f1
    }
# function to Generate a Grad-CAM heatmap overlay on the input image.
def generate_gradcam(model, image_tensor, head="chest", class_idx=None, target_layer_name="blocks.2.layers.12.conv2", device="cpu"):
    model.eval()

    # Hook storage
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    def forward_hook(module, input, output):
        activations.append(output.detach())

    # Register hooks on chosen layer
    target_layer = dict(model.named_modules())[target_layer_name]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # Clone and ensure grad
    image_tensor = image_tensor.detach().clone().requires_grad_(True).to(device)

    # Forward
    fracture_pred, chest_pred = model(image_tensor)

    # Choose head
    if head == "fracture":
        output = fracture_pred
        class_idx = 0  # sigmoid binary
    elif head == "chest":
        output = chest_pred
        if class_idx is None:
            class_idx = chest_pred.argmax().item()
    else:
        raise ValueError("head must be 'chest' or 'fracture'")

    # Backward on target class
    model.zero_grad()
    target = output[0, class_idx]
    target.backward()

    # Grad-CAM calculations
    grads = gradients[0]
    acts = activations[0]
    pooled_grads = torch.mean(grads, dim=[0, 2, 3])

    for i in range(acts.shape[1]):
        acts[0, i] *= pooled_grads[i]

    heatmap = torch.mean(acts, dim=1).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap + 1e-8)  # avoid divide by zero

    # Resize heatmap and overlay
    heatmap = cv2.resize(heatmap, (image_tensor.shape[3], image_tensor.shape[2]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_INFERNO)

    # Convert image tensor to RGB
    img = image_tensor.detach().cpu().numpy()

    # Ensure it's [C, H, W]
    if img.ndim == 4:
        img = img[0]  # remove batch

    # Now handle shape [1, H, W] or [C, H, W]
    if img.shape[0] == 1:
        img = np.repeat(img, 3, axis=0)  # grayscale → RGB

    img = np.transpose(img, (1, 2, 0))  # [C, H, W] → [H, W, C]
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    return overlay



