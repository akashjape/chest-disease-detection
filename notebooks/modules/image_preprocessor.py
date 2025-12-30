
import cv2
import numpy as np

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image for model input.

    Parameters:
    -----------
    image_path : str
        Path to the image file
    target_size : tuple
        (height, width) to resize the image

    Returns:
    --------
    img : numpy array
        Preprocessed image normalized to [0, 1]
    """
    try:
        # Load image
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"Could not load image at {image_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to target size
        img = cv2.resize(img, target_size)

        # Normalize to [0, 1] range
        img = img.astype(np.float32) / 255.0

        return img

    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

def create_tf_dataset(df, image_dir="", target_size=(224, 224),
                      batch_size=32, shuffle=True, augment=False):
    """
    Create a TensorFlow dataset from a DataFrame.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with 'full_path' and label columns
    target_size : tuple
        Target image size
    batch_size : int
        Batch size
    shuffle : bool
        Whether to shuffle the dataset
    augment : bool
        Whether to apply data augmentation

    Returns:
    --------
    dataset : tf.data.Dataset
        TensorFlow dataset
    """
    import tensorflow as tf

    # Data augmentation
    if augment:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )
    else:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    # Create dataset
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir,  # This would need adjustment for NIH dataset structure
        labels=df[LABEL_COLUMNS].values,
        batch_size=batch_size,
        image_size=target_size,
        shuffle=shuffle
    )

    return dataset
