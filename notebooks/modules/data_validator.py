
import pandas as pd

def check_data_leakage(df1, df2, col='full_path'):
    """
    Check if any image appears in both df1 and df2.

    Parameters:
    -----------
    df1, df2 : pandas DataFrames
        DataFrames to compare
    col : str
        Column name to check for duplicates

    Returns:
    --------
    bool : True if no leakage, False if leakage detected
    """
    set1 = set(df1[col].tolist())
    set2 = set(df2[col].tolist())
    overlap = set1.intersection(set2)

    if len(overlap) > 0:
        print(f" WARNING: {len(overlap)} overlapping images found!")
        return False
    else:
        print("No data leakage detected.")
        return True

def get_data_statistics(df, split_name):
    """
    Generate statistics for a dataset split.

    Parameters:
    -----------
    df : pandas DataFrame
        Dataset to analyze
    split_name : str
        Name of the split (for reporting)

    Returns:
    --------
    dict : Dictionary containing statistics
    """
    from collections import Counter

    stats = {
        'split_name': split_name,
        'total_samples': len(df),
        'samples_per_class': {},
        'multi_label_distribution': dict(Counter(df['num_labels'])),
        'avg_labels_per_image': df['num_labels'].mean()
    }

    # Count per class
    for col in LABEL_COLUMNS:
        stats['samples_per_class'][col] = int(df[col].sum())

    return stats
