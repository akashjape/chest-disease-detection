# Backend Deployment Guide

## Local Development

1. **Setup Environment**

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Run API**

```bash
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

3. **Test API**
   Visit: `http://localhost:8000/docs` for interactive API documentation

## Deployment on Render

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Add backend API"
git push origin main
```

### Step 2: Create Render Service

1. Go to https://dashboard.render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: chest-xray-api
   - **Environment**: Python 3
   - **Region**: Choose closest to users
   - **Branch**: main
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`

### Step 3: Set Environment Variables

In Render dashboard:

- **MODEL_PATH**: `models/best_chest_model_8320.pth`
- **DEVICE**: `cuda` (if available)
- **CORS_ORIGINS**: `["https://your-frontend.vercel.app"]`

### Step 4: Deploy Models

- Upload your model file to Render or use a storage service (AWS S3, etc.)
- Update MODEL_PATH accordingly

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /labels` - Get available disease labels
- `POST /predict` - Predict from single image
- `POST /predict-batch` - Batch prediction

## Environment Variables

Create `.env` file in backend folder:

```
MODEL_PATH=../models/best_chest_model_8320.pth
DEVICE=cuda
CORS_ORIGINS=["http://localhost:3000", "https://your-domain.vercel.app"]
MAX_UPLOAD_SIZE=10485760
```

## Important Notes

1. Model size might be large - consider using AWS S3 or similar for storage
2. GPU memory might be limited on free Render tier - may need paid plan
3. Update CORS_ORIGINS with your actual Vercel domain
4. Cold start might be slow due to model loading
