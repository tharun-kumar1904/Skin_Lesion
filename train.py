import os
import json
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import psutil
import shutil
import logging
from utils import SkinLesionDataset, fit_and_encode_labels
from models.cnn_transformer import EfficientNetModel
from models.attention_model import MobileNetModel
from models.diversity_model import DiversityModel
from models.vit_model import ViTModel

# Logging setup with file handling and fallback
log_file = 'train_output.txt'
alternate_log_file = 'train_output_alt.txt'
if os.path.exists(log_file):
    try:
        with open(log_file, 'a') as f:
            pass
    except PermissionError:
        log_file = alternate_log_file
        logging.warning(f"Permission denied for {log_file}, using {alternate_log_file}")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)])

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def compute_class_weights(df, label_col='label_encoded'):
    class_counts = df[label_col].value_counts().sort_index()
    n_samples = len(df)
    n_classes = len(class_counts)
    if n_samples == 0 or n_classes == 0:
        logging.error("No valid samples or classes found for weight computation.")
        raise ValueError("Dataset is empty or lacks valid classes.")
    weights = n_samples / (n_classes * class_counts)
    return torch.tensor(weights.values, dtype=torch.float32)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, input, target):
        log_prob = torch.nn.functional.log_softmax(input, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(-1).to(torch.long)).squeeze(-1)
        if self.weight is not None:
            nll_loss = nll_loss * self.weight[target]
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def validate_model(model, val_loader, device, epoch, loss=None):
    model.eval()
    correct = total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device).long()
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = correct / total
    if loss is not None:
        print(f"Epoch {epoch}: Loss: {loss:.4f}")
        print(f"Validation Accuracy: {acc*100:.2f}%")
        logging.info(f"Epoch {epoch}: Loss: {loss:.4f}")
        logging.info(f"Validation Accuracy: {acc*100:.2f}%")
    return acc

def train_epoch(model, train_loader, criterion, optimizer, device, scaler, accumulation_steps=64, source_csv=None):
    model.train()
    optimizer.zero_grad()
    total_loss = 0
    logging.info(f"Starting training epoch with {len(train_loader)} batches")
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    for i, (x, y) in enumerate(train_loader):
        logging.info(f"Processing batch {i+1}/{len(train_loader)} from {source_csv}")
        x, y = x.to(device), y.to(device).long()
        try:
            with autocast(device_type=device_type):
                pred = model(x)
                loss = criterion(pred, y) / accumulation_steps
            scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss += loss.item() * accumulation_steps
            logging.info(f"Batch {i+1} loss: {loss.item() * accumulation_steps:.4f}")
            if i == 0:
                print(f"First batch processed successfully at {time.strftime('%H:%M:%S')}")
        except Exception as e:
            logging.error(f"Error in batch {i+1}: {e}")
            raise
    logging.info(f"Epoch completed with average loss: {total_loss / len(train_loader):.4f}")
    return total_loss / len(train_loader)

def main():
    try:
        logging.info(f"Current directory: {os.getcwd()}")
        logging.info("Using updated version with checkpointing and 39 classes")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        logging.info(f"Initial memory: {get_memory_usage():.2f} MB")

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        # Updated CSV paths for pre-split datasets
        train_csv_path = "F:/C-download/skin_lesion_project_Major/data/train_dataset.csv"
        val_csv_path = "F:/C-download/skin_lesion_project_Major/data/val_dataset.csv"
        test_csv_path = "F:/C-download/skin_lesion_project_Major/data/test_dataset.csv"

        for csv_path in [train_csv_path, val_csv_path, test_csv_path]:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            logging.info(f"Found CSV file: {csv_path}")

        # Load and combine data for initial label encoding
        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)
        test_df = pd.read_csv(test_csv_path)
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        logging.info(f"Loaded CSVs. Combined Shape: {df.shape}")

        # Fix path checking with consistent separators and subdirectory
        base_dir = "F:/C-download/skin_lesion_project_Major/Skin-Lesions-Classification/Dataset"
        def check_path(row, split_dir):
            subdir = os.path.basename(split_dir).replace('_dataset.csv', '')
            full_path = os.path.join(base_dir, subdir, row['image'].replace('\\', '/'))
            return os.path.exists(full_path)

        # Assign original CSV source to each row for path validation
        df['source_csv'] = [train_csv_path] * len(train_df) + [val_csv_path] * len(val_df) + [test_csv_path] * len(test_df)
        invalid_paths = df[~df.apply(lambda row: check_path(row, row['source_csv']), axis=1)]
        if not invalid_paths.empty:
            logging.warning(f"Invalid paths found: {invalid_paths['image'].tolist()[:5]}")
            df = df[df.apply(lambda row: check_path(row, row['source_csv']), axis=1)]

        if df.empty:
            raise ValueError("No valid data remains after filtering invalid paths. Check CSV paths and filesystem.")

        le, label2idx, idx2label, num_classes = fit_and_encode_labels(df)
        if num_classes != 39:
            logging.warning(f"Expected 39 classes, found {num_classes}")
        df['label_encoded'] = le.transform(df['label'])
        logging.info(f"Found {num_classes} classes: {list(idx2label.values())}")

        with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
            json.dump(idx2label, f, indent=2)
        label_fullnames = {
            "Akne": "Akne", "Atopic Dermatitis": "Atopic Dermatitis",
            "Basal Cell Carcinoma": "Basal Cell Carcinoma", "Benign Keratosis": "Benign Keratosis",
            "Bullous": "Bullous", "Chickenpox": "Chickenpox", "Cowpox": "Cowpox",
            "Dermatofibroma": "Dermatofibroma", "Eczema": "Eczema",
            "Exanthems and Drug Eruptions": "Exanthems and Drug Eruptions",
            "Hailey-Hailey Disease": "Hailey-Hailey Disease", "Hair loss Alopecia": "Hair loss Alopecia",
            "HFMD": "HFMD", "Impetigo": "Impetigo", "Larva Migrans": "Larva Migrans",
            "Leprosy Borderline": "Leprosy Borderline", "Leprosy Lepromatous": "Leprosy Lepromatous",
            "Leprosy Tuberculoid": "Leprosy Tuberculoid", "Lichen Planus": "Lichen Planus",
            "Light Diseases and disorders of pigmentation": "Light Diseases and disorders of pigmentation",
            "Lupus": "Lupus", "Measles": "Measles", "Melanocytic Nevi": "Melanocytic Nevi",
            "Melanoma": "Melanoma", "Molluscum Contagiosum": "Molluscum Contagiosum",
            "Monkeypox": "Monkeypox", "Nail Fungus": "Nail Fungus", "Pigment": "Pigment",
            "Pityriasis Rosea": "Pityriasis Rosea", "Poison Ivy": "Poison Ivy",
            "Porokeratosis Actinic": "Porokeratosis Actinic", "Psoriasis": "Psoriasis",
            "Scabies Lyme Disease": "Scabies Lyme Disease", "Seborheic Keratosis": "Seborheic Keratosis",
            "Systemic Disease": "Systemic Disease", "Tinea Ringworm": "Tinea Ringworm",
            "Tungiasis": "Tungiasis", "Urticaria Hives": "Urticaria Hives",
            "Vasculitis Photos": "Vasculitis Photos"
        }
        with open(os.path.join(output_dir, "label_fullnames.json"), "w") as f:
            json.dump(label_fullnames, f, indent=2)

        original_class_counts = df['label_encoded'].value_counts().sort_index()
        logging.info(f"Original class distribution:\n{original_class_counts}")
        df_balanced = df
        class_dist = original_class_counts
        logging.info(f"Balanced class distribution:\n{class_dist}")
        with open(os.path.join(output_dir, "class_distribution.txt"), "w") as f:
            f.write(str(class_dist))

        # Split data into train, val, test DataFrames
        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)
        test_df = pd.read_csv(test_csv_path)

        # Apply the same path fix and filtering to individual splits
        train_df = train_df[train_df.apply(lambda row: check_path(row, train_csv_path), axis=1)]
        val_df = val_df[val_df.apply(lambda row: check_path(row, val_csv_path), axis=1)]
        test_df = test_df[test_df.apply(lambda row: check_path(row, test_csv_path), axis=1)]

        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError("One or more data splits are empty after filtering. Check CSV paths and filesystem.")

        train_df['label_encoded'] = le.transform(train_df['label'])
        val_df['label_encoded'] = le.transform(val_df['label'])
        test_df['label_encoded'] = le.transform(test_df['label'])

        logging.info(f"Train dataset size: {len(train_df)} images")
        logging.info(f"Val dataset size: {len(val_df)} images")
        logging.info(f"Test dataset size: {len(test_df)} images")

        class_weights = compute_class_weights(train_df).to(device)

        train_loader = DataLoader(
            SkinLesionDataset(train_df.assign(source_csv=train_csv_path), augment=True),
            batch_size=1,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        val_loader = DataLoader(
            SkinLesionDataset(val_df.assign(source_csv=val_csv_path)),
            batch_size=1,
            num_workers=0,
            pin_memory=True
        )
        test_loader = DataLoader(
            SkinLesionDataset(test_df.assign(source_csv=test_csv_path)),
            batch_size=1,
            num_workers=0,
            pin_memory=True
        )

        # Initialize models with explicit logging and error handling
        logging.info("Initializing EfficientNetModel")
        effnet = EfficientNetModel(num_classes).to(device)
        logging.info("Initializing MobileNetModel")
        mobilenet = MobileNetModel(num_classes).to(device)
        logging.info("Initializing ViTModel")
        try:
            vit = ViTModel(num_classes).to(device)
            logging.info("ViTModel initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize ViTModel: {e}")
            raise

        effnet_path = os.path.join(output_dir, "effnet_pretrained.pth")
        if os.path.exists(effnet_path):
            state_dict = torch.load(effnet_path, weights_only=True)
            try:
                effnet.load_state_dict(state_dict)
                logging.info(f"Loaded checkpoint for effnet")
            except RuntimeError as e:
                logging.warning(f"Failed to load effnet checkpoint due to mismatch: {e}. Initializing from scratch.")
        mobilenet_path = os.path.join(output_dir, "mobilenet_pretrained.pth")
        if os.path.exists(mobilenet_path):
            state_dict = torch.load(mobilenet_path, weights_only=True)
            try:
                mobilenet.load_state_dict(state_dict)
                logging.info(f"Loaded checkpoint for mobilenet")
            except RuntimeError as e:
                logging.warning(f"Failed to load mobilenet checkpoint due to mismatch: {e}. Initializing from scratch.")
        vit_path = os.path.join(output_dir, "vit_pretrained.pth")
        if os.path.exists(vit_path):
            state_dict = torch.load(vit_path, weights_only=True)
            try:
                vit.load_state_dict(state_dict)
                logging.info(f"Loaded checkpoint for vit")
            except RuntimeError as e:
                logging.warning(f"Failed to load vit checkpoint due to mismatch: {e}. Initializing from scratch.")

        ensemble = DiversityModel([effnet, mobilenet, vit], weights=[0.33, 0.33, 0.33]).to(device)
        ensemble_path = os.path.join(output_dir, "best_model.pth")

        criterion = LabelSmoothingCrossEntropy(smoothing=0.1, weight=class_weights)
        optimizer = optim.Adam(ensemble.parameters(), lr=0.0001)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
        scaler = torch.amp.GradScaler('cuda')

        num_epochs = 30
        best_acc = 0
        for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
            train_loss = train_epoch(ensemble, train_loader, criterion, optimizer, device, scaler, accumulation_steps=64, source_csv=train_csv_path)
            acc = validate_model(ensemble, val_loader, device, epoch + 1, train_loss)
            scheduler.step(acc)
            if acc > best_acc:
                best_acc = acc
                torch.save(ensemble.state_dict(), ensemble_path)
                torch.save(effnet.state_dict(), os.path.join(output_dir, "effnet_pretrained.pth"))
                torch.save(mobilenet.state_dict(), os.path.join(output_dir, "mobilenet_pretrained.pth"))
                torch.save(vit.state_dict(), os.path.join(output_dir, "vit_pretrained.pth"))
                logging.info(f"Saved best model checkpoint at epoch {epoch + 1} with accuracy {best_acc*100:.2f}%")

        final_acc = validate_model(ensemble, test_loader, device, num_epochs, None)
        logging.info(f"Final Accuracy: {final_acc*100:.2f}%")
        torch.save(ensemble.state_dict(), os.path.join(output_dir, "div_model.pth"))
        torch.save(effnet.state_dict(), os.path.join(output_dir, "effnet_pretrained.pth"))
        torch.save(mobilenet.state_dict(), os.path.join(output_dir, "mobilenet_pretrained.pth"))
        torch.save(vit.state_dict(), os.path.join(output_dir, "vit_pretrained.pth"))

        # Add count of images trained from each directory
        train_images = len(train_loader.dataset)
        val_images = len(val_loader.dataset)
        test_images = len(test_loader.dataset)
        print(f"Number of images trained from Train: {train_images}")
        print(f"Number of images validated from Val: {val_images}")
        print(f"Number of images tested from Test: {test_images}")
        logging.info(f"Number of images trained from Train: {train_images}")
        logging.info(f"Number of images validated from Val: {val_images}")
        logging.info(f"Number of images tested from Test: {test_images}")

    except Exception as e:
        logging.error(f"Error in training: {e}")
        raise

if __name__ == "__main__":
    main()