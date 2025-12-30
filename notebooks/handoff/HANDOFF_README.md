
NIH CHEST X-RAY DATA PIPELINE
=============================

Overview
--------
This pipeline loads, preprocesses, and validates the NIH Chest X-ray dataset
for multi-label classification of 15 thoracic diseases.

Dataset Structure
-----------------
- Train: 69,219 images
- Validation: 17,305 images
- Test: 25,596 images
- Total: 112,120 images

Classes (15):
-------------
Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema,
Fibrosis, Hernia, Infiltration, Mass, No Finding, Nodule,
Pleural_Thickening, Pneumonia, Pneumothorax

Pipeline Modules
----------------
1. `data_loader.py` - Load train/val/test splits
2. `image_preprocessor.py` - Image loading and preprocessing
3. `data_validator.py` - Data leakage checks and statistics

Usage Example
-------------
```python
# Import modules
from modules.data_loader import load_nih_splits
from modules.image_preprocessor import load_and_preprocess_image
from modules.data_validator import check_data_leakage

# Load data
train_df, val_df, test_df = load_nih_splits()

# Check for data leakage
check_data_leakage(train_df, val_df)

# Preprocess an image
img = load_and_preprocess_image(train_df.iloc[0]['full_path'])
    