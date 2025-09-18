import pandas as pd
import torch
from imblearn.over_sampling import RandomOverSampler
from utils import SkinLesionDataset, fit_and_encode_labels
from models.cnn_transformer import EfficientNetModel
from models.attention_model import MobileNetModel
from models.diversity_model import DiversityModel

# Test imblearn
ros = RandomOverSampler(random_state=42)
X = pd.DataFrame({'image': ['a.jpg', 'b.jpg']})  # Changed from image_path to image
y = pd.Series([0, 1])
X_res, y_res = ros.fit_resample(X, y)
print("✅ imblearn test passed")

# Test utils with 39 classes
df = pd.DataFrame({
    'image': ['Train/Akne/test.jpg'],  # Adjusted to match new CSV structure
    'label': ['Akne']  # Example label from 39 types
})
le, l2i, i2l, nc = fit_and_encode_labels(df)
print(f"✅ utils test passed. Label mapping: {i2l}, Number of classes: {nc}")

# Test models with 39 classes
num_classes = 39  # Adjusted to match 39 classes
effnet = EfficientNetModel(num_classes)
mobilenet = MobileNetModel(num_classes)
ensemble = DiversityModel([effnet, mobilenet])
x = torch.randn(1, 3, 224, 224)
out = ensemble(x)
print(f"✅ Model test passed. Output shape: {out.shape}, Expected classes: {num_classes}")