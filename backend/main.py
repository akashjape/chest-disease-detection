import os
import io
import json
import numpy as np
import cv2
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# Lazy load PyTorch to avoid dynamo overhead
def load_torch():
    import torch
    return torch

def load_model_lib():
    import torchvision.models as models
    from torchvision import transforms
    return models, transforms

# Will be initialized on startup
torch = None
models = None
transforms = None

# Initialize FastAPI app
app = FastAPI(title="Chest X-Ray Disease Detection API", version="1.0.0")

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "../models/best_chest_model_8320.pth")
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.40  # Only show predictions above this threshold

# Label mapping (14 classes - matches model output)
# Note: "No Finding" is detected based on confidence levels, not a separate class
# Note: "No Finding" is detected based on confidence levels, not a separate class
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
    10: "Nodule",
    11: "Pleural_Thickening",
    12: "Pneumonia",
    13: "Pneumothorax",
}

NO_FINDING_THRESHOLD = 0.35  # If max prediction < this, classify as "No Finding"

NO_FINDING_THRESHOLD = 0.35  # If max prediction < this, classify as "No Finding"

# Global model variable
model = None
torch = None
models = None
transforms = None
DEVICE = None


def initialize_torch():
    """Initialize PyTorch on first use"""
    global torch, models, transforms, DEVICE
    if torch is None:
        torch = load_torch()
        models, transforms = load_model_lib()
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {DEVICE}")


def load_model():
    """Load the pretrained PyTorch model"""
    global model
    initialize_torch()
    
    try:
        # Build EfficientNet-B4 architecture to match training notebook
        # Use weights=None to avoid downloading imagenet weights at runtime
        try:
            # torchvision >= 0.13 provides efficientnet_b4
            efficient = models.efficientnet_b4
        except Exception:
            # Fallback name if models namespace differs
            efficient = getattr(models, 'efficientnet_b4')

        model = efficient(weights=None)

        # Determine in_features for classifier
        try:
            in_features = model.classifier[1].in_features
        except Exception:
            in_features = model.classifier.in_features if hasattr(model.classifier, 'in_features') else 1792

        # Default num_classes; will try to infer from checkpoint when available
        num_classes = 15
        
        # Load the saved weights
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            # Extract state_dict
            state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

            # Try to infer num_classes from checkpoint final linear weight shape.
            # Some checkpoints store a multi-layer classifier with keys like
            # 'classifier.1.weight' and 'classifier.4.weight'. We should pick
            # the classifier weight with the highest numeric index (likely the
            # final linear) rather than the first matching 2D weight.
            try:
                inferred = None
                final_key = None
                import re
                cls_keys = []
                for k, v in state_dict.items():
                    m = re.search(r"classifier\.(\d+)\.weight", k)
                    if m and hasattr(v, 'ndim') and v.ndim == 2:
                        idx = int(m.group(1))
                        cls_keys.append((idx, k, v))

                if cls_keys:
                    # pick the entry with the largest index (assumed final linear)
                    cls_keys.sort(key=lambda x: x[0])
                    idx, final_key, final_v = cls_keys[-1]
                    inferred = int(final_v.shape[0])

                if inferred is not None:
                    num_classes = int(inferred)
                    print(f"[INFO] Inferred num_classes={num_classes} from checkpoint key '{final_key}'")
            except Exception:
                pass

            # Rebuild classifier using inferred num_classes
            model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.4),
                torch.nn.Linear(in_features, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(512, num_classes),
            )

            # Now attempt to load state dict
            try:
                model.load_state_dict(state_dict)
                print("[OK] Model loaded successfully from {}".format(MODEL_PATH))
            except Exception as e:
                # Loading failed — print error and some debug info
                try:
                    ck_keys = list(state_dict.keys())
                except Exception:
                    ck_keys = []
                print("[ERROR] Failed to load checkpoint into EfficientNet-B4: {}".format(str(e)))
                print("[DEBUG] Checkpoint keys: {}".format(ck_keys[-10:]))
                model = None
                return
        else:
            print("[WARNING] Model path not found: {}".format(MODEL_PATH))
        
        model = model.to(DEVICE)
        model.eval()
    except Exception as e:
        print("[ERROR] Error loading model: {}".format(str(e)))
        raise


def preprocess_image(image_bytes):
    """Preprocess uploaded image for model prediction"""
    initialize_torch()
    
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if grayscale or RGBA
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Apply transforms
        image_tensor = transform(image)
        return image_tensor.unsqueeze(0)  # Add batch dimension
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")


def validate_chest_xray(image_bytes):
    """Smart heuristic to distinguish chest X-rays from documents/photos.
    Focuses on detecting TEXT patterns (documents) vs MEDICAL PATTERNS (X-rays).
    Returns True if likely a chest X-ray, False otherwise.
    """
    try:
        # Load via PIL then convert to cv2-friendly array
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # 1) Color saturation check (X-rays are mostly grayscale)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        mean_sat = float(np.mean(sat))

        # 2) Edge density check - documents have VERY HIGH edge density from text
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.sum(edges > 0)) / (edges.shape[0] * edges.shape[1])

        # 3) Text detection: look for LINES and CONNECTED TEXT regions
        # Apply morphological operations to detect connected text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count small, narrow regions (characteristic of text)
        text_like_regions = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50 and area < 500:  # Text character range
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / (h + 1)
                # Text has varied aspect ratios; check if it forms lines
                if 0.1 < aspect_ratio < 10 and w > 5 and h > 5:
                    text_like_regions += 1

        # 4) Histogram analysis for text detection
        # Documents show sharp peaks from text, X-rays show smooth distribution
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist.flatten() / (hist.sum() + 1e-5)
        
        # Count significant peaks
        peaks = 0
        for i in range(10, 246):  # Skip extremes
            if (hist_normalized[i] > hist_normalized[i-1] and 
                hist_normalized[i] > hist_normalized[i+1] and 
                hist_normalized[i] > 0.015):
                peaks += 1

        # === REJECTION LOGIC ===
        # Reject ONLY if strong evidence of document/text:
        is_document = False
        
        # Very high saturation = likely color photo/document
        if mean_sat > 0.20:
            is_document = True
            
        # Very high edge density + text regions = likely document
        if edge_density > 0.15 and text_like_regions > 30:
            is_document = True
            
        # Many histogram peaks + high text regions = likely document text
        if peaks > 10 and text_like_regions > 25:
            is_document = True
            
        # Document-specific: extreme edge density with text patterns
        if edge_density > 0.20 and peaks > 8:
            is_document = True

        is_xray = not is_document

        # Debug output
        print(f"[VALIDATION] sat={mean_sat:.3f} | edges={edge_density:.3f} | text_regions={text_like_regions} | peaks={peaks} | => {'✓ XRAY' if is_xray else '✗ NOT XRAY'}")

        return bool(is_xray)
    except Exception as e:
        print(f"[VALIDATION ERROR] {e}")
        # If validation fails, be lenient (let it through for user to see results)
        return True


def predict(image_tensor):
    """Run prediction on the image"""
    initialize_torch()
    
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(DEVICE)
            logits = model(image_tensor)
            probabilities = torch.sigmoid(logits)  # Use sigmoid for multi-label classification
            
        return probabilities.cpu().numpy()[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Initialize PyTorch and load model on application startup"""
    print("Initializing backend...")
    initialize_torch()
    print("Loading model...")
    load_model()
    print("Backend ready!")


@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": "Chest X-Ray Disease Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "labels": "/labels (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": model is not None
    }


@app.get("/labels")
async def get_labels():
    """Get available disease labels"""
    return {
        "labels": LABEL_MAPPING,
        "total_classes": len(LABEL_MAPPING)
    }


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Predict chest diseases from X-ray image
    
    - **file**: X-ray image file (JPG, PNG, etc.)
    
    Returns:
    - **predictions**: Dictionary with disease probabilities
    - **top_predictions**: Top 5 most likely diseases
    """
    try:
        # Validate file type
        allowed_formats = {"image/jpeg", "image/png", "image/jpg"}
        if file.content_type not in allowed_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed: {allowed_formats}"
            )
        
        # Read file
        image_bytes = await file.read()

        # Validate that uploaded image is likely a chest X-ray
        is_xray = validate_chest_xray(image_bytes)
        if not is_xray:
            raise HTTPException(status_code=400, detail="Uploaded image does not appear to be a frontal chest X-ray. Please upload a chest radiograph (PA/AP).")

        # Check if model is loaded
        if model is None:
            # Generate mock predictions for testing UI
            import hashlib
            image_hash = int(hashlib.md5(image_bytes).hexdigest(), 16)
            np.random.seed(image_hash % (2**31))
            
            # Create realistic mock predictions sized to label mapping
            n_labels = len(LABEL_MAPPING)
            probabilities = np.random.beta(2, 5, size=n_labels)  # Beta distribution for realistic variation
            probabilities = probabilities * 0.7  # Scale down to avoid all 100%
            
            predictions = {
                LABEL_MAPPING[idx]: float(prob)
                for idx, prob in enumerate(probabilities)
            }
            
            # Get top 5 predictions with confidence filtering
            max_prob = np.max(probabilities)
            if max_prob < NO_FINDING_THRESHOLD:
                # Normal X-ray
                top_predictions = [
                    {
                        "disease": "No Finding",
                        "probability": 1.0 - max_prob
                    }
                ]
            else:
                top_indices = np.argsort(probabilities)[::-1]
                top_predictions = [
                    {
                        "disease": LABEL_MAPPING[idx],
                        "probability": float(probabilities[idx])
                    }
                    for idx in top_indices
                    if probabilities[idx] >= CONFIDENCE_THRESHOLD
                ][:5]  # Limit to top 5
            
            return {
                "success": True,
                "predictions": predictions,
                "top_predictions": top_predictions,
                "device": str(DEVICE),
                "model_status": "DEMO MODE - Model file not found. Using mock predictions for UI testing.",
                "is_mock": True
            }
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        
        # Run prediction
        probabilities = predict(image_tensor)
        
        # Check if this is a "No Finding" case (all predictions weak)
        max_prob = np.max(probabilities)
        
        # Format results
        predictions = {
            LABEL_MAPPING[idx]: float(prob)
            for idx, prob in enumerate(probabilities)
        }
        
        # Get top 5 predictions with confidence filtering
        if max_prob < NO_FINDING_THRESHOLD:
            # Normal X-ray with no pathology
            top_predictions = [
                {
                    "disease": "No Finding",
                    "probability": 1.0 - max_prob  # Confidence of being normal
                }
            ]
        else:
            top_indices = np.argsort(probabilities)[::-1]
            top_predictions = [
                {
                    "disease": LABEL_MAPPING[idx],
                    "probability": float(probabilities[idx])
                }
                for idx in top_indices
                if probabilities[idx] >= CONFIDENCE_THRESHOLD
            ][:5]  # Limit to top 5
        
        return {
            "success": True,
            "predictions": predictions,
            "top_predictions": top_predictions,
            "device": str(DEVICE),
            "model_status": "Live predictions from trained model",
            "is_mock": False
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Batch prediction for multiple X-ray images
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for file in files:
        try:
            image_bytes = await file.read()

            # Validate image before processing
            if not validate_chest_xray(image_bytes):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Uploaded file does not appear to be a chest X-ray"
                })
                continue

            image_tensor = preprocess_image(image_bytes)
            probabilities = predict(image_tensor)
            
            predictions = {
                LABEL_MAPPING[idx]: float(prob)
                for idx, prob in enumerate(probabilities)
            }
            
            results.append({
                "filename": file.filename,
                "success": True,
                "predictions": predictions
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "batch_results": results,
        "total_files": len(files),
        "successful": sum(1 for r in results if r.get("success", False))
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
