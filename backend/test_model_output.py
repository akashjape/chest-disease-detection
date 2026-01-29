import torch
import numpy as np
import cv2
from PIL import Image
import io

# Load model like backend does
MODEL_PATH = "../models/best_chest_model_8320.pth"
DEVICE = torch.device("cpu")

# Load models
import torchvision.models as models
from torchvision import transforms

# Build EfficientNet-B4 and load checkpoint
efficient = models.efficientnet_b4(weights=None)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

# Get num_classes from checkpoint
import re
num_classes = 15
try:
    inferred = None
    for k, v in state_dict.items():
        m = re.search(r"classifier\.(\d+)\.weight", k)
        if m and hasattr(v, 'ndim') and v.ndim == 2:
            idx = int(m.group(1))
            final_v = v
            inferred = int(final_v.shape[0])
    if inferred is not None:
        num_classes = int(inferred)
except:
    pass

print(f"Num classes: {num_classes}")
print(f"Checkpoint keys sample: {list(state_dict.keys())[:5]}")

# Rebuild classifier
in_features = 1792
model = efficient
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.4),
    torch.nn.Linear(in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, num_classes),
)

model.load_state_dict(state_dict)
model.eval()
model.to(DEVICE)

print("✓ Model loaded")

# Create a test input (all black - should be similar to "No Finding")
test_input = torch.zeros(1, 3, 224, 224)

with torch.no_grad():
    output = model(test_input)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output values: {output[0][:5]}...")
    
    # Apply sigmoid
    probs = torch.sigmoid(output)
    print(f"\nProbs shape: {probs.shape}")
    print(f"Probs values: {probs[0][:5]}...")
    print(f"Max prob: {probs[0].max():.3f}, Min prob: {probs[0].min():.3f}")
    
    # Get top 5
    top_vals, top_idx = torch.topk(probs[0], 5)
    print(f"\nTop 5 predictions:")
    LABEL_MAPPING = {
        0: "Atelectasis",
        1: "Cardiomegaly",
        2: "Consolidation",
        3: "Edema",
        4: "Effusion",
        5: "Emphysema",
        6: "Fibrosis",
        7: "Hernia",
        8: "Infiltration",
        9: "Mass",
        10: "No Finding",
        11: "Nodule",
        12: "Pleural_Thickening",
        13: "Pneumonia",
        14: "Pneumothorax",
    }
    for idx, val in zip(top_idx, top_vals):
        idx_int = idx.item()
        print(f"  {idx_int}: {LABEL_MAPPING[idx_int]}: {val:.3f}")
