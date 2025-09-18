import os
import torch
from PIL import Image
from torchvision import transforms
import logging

# Basic logging setup for utils
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SkinLesionDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, augment=False):
        self.dataframe = dataframe
        self.augment = augment
        self.base_dir = "F:/C-download/skin_lesion_project_Major/Skin-Lesions-Classification/Dataset"
        
        # Define transforms
        self.base_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = self.base_transforms

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image']
        label = self.dataframe.iloc[idx]['label_encoded']
        source_csv = self.dataframe.iloc[idx].get('source_csv', None)
        
        if source_csv is None:
            raise ValueError("Source CSV not available. Ensure 'source_csv' column is present.")
        
        subdir = os.path.basename(source_csv).replace('_dataset.csv', '')
        full_path = os.path.join(self.base_dir, subdir, img_path.replace('\\', '/'))
        logging.info(f"Attempting to load image: {full_path}")
        
        if not os.path.exists(full_path):
            logging.error(f"Image not found at {full_path}")
            raise FileNotFoundError(f"Image not found: {full_path}")
        
        try:
            image = Image.open(full_path).convert('RGB')
            image = self.transform(image)
            return image, label
        except Exception as e:
            logging.error(f"Error loading image {full_path}: {e}")
            raise

def fit_and_encode_labels(df):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels = df['label'].values
    le.fit(labels)
    label_encoded = le.transform(labels)
    df['label_encoded'] = label_encoded
    label2idx = {label: idx for idx, label in enumerate(le.classes_)}
    idx2label = {idx: label for idx, label in enumerate(le.classes_)}
    return le, label2idx, idx2label, len(le.classes_)