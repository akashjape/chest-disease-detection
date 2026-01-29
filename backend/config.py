import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
API_TITLE = "Chest X-Ray Disease Detection"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Detect multiple chest diseases from X-ray images using deep learning"

# Model Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_chest_model_8320.pth")
DEVICE = os.getenv("DEVICE", "cuda")
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5

# API Configuration
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 10485760))  # 10MB
ALLOWED_FORMATS = {"image/jpeg", "image/png", "image/jpg"}

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "['http://localhost:3000']")
