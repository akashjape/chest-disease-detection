
import pandas as pd
from pathlib import Path

def load_nih_splits(data_path="../data"):
    """
    Load NIH Chest X-ray train/val/test splits.

    Parameters:
    -----------
    data_path : str or Path
        Path to the main data directory

    Returns:
    --------
    train_df, val_df, test_df : pandas DataFrames
        DataFrames containing image paths and labels
    """
    processed_path = Path(data_path) / "processed/metadata"

    train_df = pd.read_csv(processed_path / "nih_train_data.csv")
    val_df = pd.read_csv(processed_path / "nih_val_data.csv")
    test_df = pd.read_csv(processed_path / "nih_test_data.csv")

    return train_df, val_df, test_df

LABEL_COLUMNS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
    'Infiltration', 'Mass', 'No Finding', 'Nodule',
    'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]
