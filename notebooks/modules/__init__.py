
"""
NIH Chest X-ray Data Pipeline Module
====================================

This module provides a complete pipeline for loading, preprocessing,
and validating the NIH Chest X-ray dataset.

Usage:
------
>>> from modules.data_loader import load_nih_splits
>>> from modules.image_preprocessor import load_and_preprocess_image
>>> from modules.data_validator import check_data_leakage

>>> # Load data
>>> train_df, val_df, test_df = load_nih_splits()
>>>
>>> # Check for data leakage
>>> check_data_leakage(train_df, val_df)
>>>
>>> # Preprocess an image
>>> img = load_and_preprocess_image(train_df.iloc[0]['full_path'])
"""

__version__ = "1.0.0"
__author__ = "NIH Image Processing Team"

# Export main functions
from .data_loader import load_nih_splits, LABEL_COLUMNS
from .image_preprocessor import load_and_preprocess_image, create_tf_dataset
from .data_validator import check_data_leakage, get_data_statistics

__all__ = [
    'load_nih_splits',
    'LABEL_COLUMNS',
    'load_and_preprocess_image',
    'create_tf_dataset',
    'check_data_leakage',
    'get_data_statistics'
]
