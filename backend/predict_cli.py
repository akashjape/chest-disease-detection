import os
import argparse
import json
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# Labels used during training (14 labels from training notebook)
LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

IMAGE_SIZE = 224


def build_model(num_classes, device):
    # Build EfficientNet-B4 and replace classifier to match training
    try:
        from torchvision.models import efficientnet_b4
    except Exception:
        # older torchvision namespace fallback
        efficientnet_b4 = getattr(__import__('torchvision.models', fromlist=['efficientnet_b4']), 'efficientnet_b4')

    model = efficientnet_b4(weights=None)
    # training notebook used model.classifier[1].in_features
    try:
        in_features = model.classifier[1].in_features
    except Exception:
        in_features = getattr(model.classifier, 'in_features', 1792)

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

    return model.to(device)


def load_checkpoint(model, path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    ck = torch.load(path, map_location=device)
    state_dict = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck

    # Try strict load first
    try:
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint into model (strict load) from {path}")
        return model
    except Exception as e:
        print(f"Strict load failed: {e}")
        print("Attempting partial/compatible load (features-only + compatible classifier weights)")

    # Partial load: only copy matching keys with same shape
    model_dict = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k in model_dict and v.size() == model_dict[k].size():
            filtered[k] = v

    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    print(f"Partially loaded {len(filtered)} / {len(model_dict)} parameter tensors from checkpoint")
    return model


def preprocess_image(path):
    img = Image.open(path).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0)  # batch dim


def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return probs


def pretty_output(probs, labels, topk=5, thresholds=None):
    result = {labels[i]: float(probs[i]) for i in range(min(len(labels), len(probs)))}
    sorted_idx = probs.argsort()[::-1]
    top = []
    for i in sorted_idx[:topk]:
        label = labels[i]
        prob = float(probs[i])
        thr = None
        detected = None
        if thresholds and label in thresholds:
            thr = float(thresholds[label])
            detected = prob >= thr
        top.append({"label": label, "probability": prob, "threshold": thr, "detected": detected})
    return result, top


def main():
    p = argparse.ArgumentParser(description="Predict chest X-ray diseases using EfficientNet-B4 checkpoint")
    p.add_argument('--image', '-i', required=True, help='Path to chest X-ray image')
    p.add_argument('--model-path', '-m', default=os.path.join('..', 'models', 'best_chest_model_8320.pth'), help='Path to checkpoint file')
    p.add_argument('--thresholds', '-t', help='Optional JSON file with per-label thresholds')
    p.add_argument('--device', '-d', default='cpu', help='Device: cpu or cuda')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')

    # Load thresholds if provided
    thresholds = None
    if args.thresholds:
        with open(args.thresholds, 'r') as fh:
            thresholds = json.load(fh)

    # Build model to match training
    num_classes = len(LABELS)
    model = build_model(num_classes, device)

    # Load checkpoint (robust handling)
    model = load_checkpoint(model, args.model_path, device)

    # Preprocess and predict
    img_tensor = preprocess_image(args.image)
    probs = predict(model, img_tensor, device)

    predictions, top = pretty_output(probs, LABELS, topk=5, thresholds=thresholds)

    out = {
        'success': True,
        'predictions': predictions,
        'top_predictions': top,
        'model_path': args.model_path,
    }

    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
