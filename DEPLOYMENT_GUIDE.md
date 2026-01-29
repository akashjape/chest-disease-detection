# Complete Deployment Guide - Chest X-Ray Disease Detection

This guide covers deployment on **Render** (Backend) and **Vercel** (Frontend).

## Project Structure

```
chest-xray-deployment/
├── backend/                    # FastAPI backend (Render)
│   ├── main.py                # FastAPI application
│   ├── config.py              # Configuration
│   ├── requirements.txt        # Python dependencies
│   ├── Procfile               # Render/Heroku config
│   ├── render.yaml            # Render-specific config
│   └── DEPLOYMENT.md          # Backend deployment guide
│
├── frontend/                   # React frontend (Vercel)
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── App.js            # Main app
│   │   └── index.js          # Entry point
│   ├── package.json
│   ├── vercel.json           # Vercel config
│   └── DEPLOYMENT.md         # Frontend deployment guide
│
├── models/
│   └── best_chest_model_8320.pth  # Trained model
│
└── README.md
```

## Quick Start - Local Development

### 1. Setup Backend

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
python main.py
```

Backend runs on: `http://localhost:8000`
API Docs: `http://localhost:8000/docs`

### 2. Setup Frontend

```bash
cd frontend
npm install
npm start
```

Frontend runs on: `http://localhost:3000`

## Deployment Steps

### Phase 1: Prepare Repository

```bash
# Initialize git if not already done
git init
git add .
git commit -m "Initial commit: Chest X-ray prediction system"
git remote add origin https://github.com/YOUR_USERNAME/chest-xray-prediction.git
git push -u origin main
```

### Phase 2: Deploy Backend to Render

1. **Create Render Account**: https://render.com

2. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select `chest-xray-prediction` repo

3. **Configure Service**:
   - **Name**: `chest-xray-api`
   - **Environment**: Python 3
   - **Region**: Pick closest to users
   - **Branch**: main
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`

4. **Set Environment Variables**:

   ```
   MODEL_PATH=models/best_chest_model_8320.pth
   DEVICE=cuda  (or cpu for free tier)
   CORS_ORIGINS=["https://your-vercel-app.vercel.app"]
   ```

5. **Deploy**: Click "Create Web Service"

**Note**: Model file should be in the repository. If too large (>100MB), use:

- AWS S3
- GitHub LFS
- Or upload to Render via SFTP

### Phase 3: Deploy Frontend to Vercel

1. **Create Vercel Account**: https://vercel.com

2. **Import Project**:
   - Click "Add New..." → "Project"
   - Import your GitHub repository

3. **Configure**:
   - **Framework**: Create React App
   - **Root Directory**: `./frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `build`

4. **Environment Variables**:

   ```
   REACT_APP_API_URL=https://your-render-api-name.onrender.com
   ```

5. **Deploy**: Click "Deploy"

## Architecture

```
┌─────────────────────────────────────────────────┐
│            User Browser (Client)                │
│         (Hosted on Vercel CDN)                  │
└──────────────────┬──────────────────────────────┘
                   │
                   │ HTTPS Requests
                   │ (Image Upload & Predictions)
                   │
┌──────────────────▼──────────────────────────────┐
│         FastAPI Backend (Render)                │
│  - Load PyTorch Model                           │
│  - Process X-ray Images                         │
│  - Run Predictions                              │
│  - Return Disease Probabilities                 │
└──────────────────┬──────────────────────────────┘
                   │
                   │ File System Access
                   │
┌──────────────────▼──────────────────────────────┐
│          Trained PyTorch Model                  │
│     (best_chest_model_8320.pth)                 │
└─────────────────────────────────────────────────┘
```

## API Endpoints

| Method | Endpoint         | Description              |
| ------ | ---------------- | ------------------------ |
| GET    | `/`              | API info                 |
| GET    | `/health`        | Health check             |
| GET    | `/labels`        | Available disease labels |
| POST   | `/predict`       | Single image prediction  |
| POST   | `/predict-batch` | Batch predictions        |

### Example Usage

```bash
# Health Check
curl https://your-api.onrender.com/health

# Get Labels
curl https://your-api.onrender.com/labels

# Single Prediction
curl -X POST \
  -F "file=@xray.jpg" \
  https://your-api.onrender.com/predict

# Response
{
  "success": true,
  "predictions": {
    "Atelectasis": 0.12,
    "No Finding": 0.85,
    ...
  },
  "top_predictions": [
    {
      "disease": "No Finding",
      "probability": 0.85
    }
  ]
}
```

## Detected Diseases

The model detects 15 chest conditions:

1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Effusion
6. Emphysema
7. Fibrosis
8. Hernia
9. Infiltration
10. Mass
11. No Finding
12. Nodule
13. Pleural_Thickening
14. Pneumonia
15. Pneumothorax

## Cost Estimates

### Render (Backend)

- **Free Tier**: Limited resources, auto-sleep after 15 min inactivity
- **Starter**: $7/month - Good for development
- **Standard**: $25+/month - Recommended for production

### Vercel (Frontend)

- **Free Tier**: Unlimited deployments, generous free tier
- **Pro**: $20/month if needed

### Total Monthly Cost

- **Minimum**: Free (with limitations)
- **Recommended**: ~$25-35/month

## Troubleshooting

### Backend Issues

1. **Model Not Found**
   - Ensure model file is in repository
   - Check `MODEL_PATH` env variable
   - For large files, use S3 or GitHub LFS

2. **GPU Not Available**
   - GPU requires paid Render plan
   - Set `DEVICE=cpu` in env vars
   - Predictions will be slower

3. **Slow Startup**
   - Model loading takes time on cold start
   - Consider using cache warming
   - Upgrade to paid plan for persistent instances

### Frontend Issues

1. **API Connection Error**
   - Verify backend is running
   - Check `REACT_APP_API_URL` in Vercel env vars
   - Ensure CORS is configured in backend

2. **Build Fails on Vercel**
   - Check Node.js version: 18+ recommended
   - Clear cache and redeploy
   - Check for environment variable issues

3. **Large Image Upload**
   - Maximum 10MB currently
   - Compress images before upload
   - Update limit in `backend/main.py` if needed

## Performance Optimization

### Backend

```python
# Use asyncio for concurrent requests
# Cache model in memory
# Use GPU if available
# Implement request queuing for high load
```

### Frontend

```javascript
// Code splitting
// Image compression before upload
// Implement request throttling
// Add service worker for offline support
```

## Security Considerations

1. **Input Validation**: Backend validates image types and sizes
2. **CORS**: Configure allowed origins
3. **Rate Limiting**: Add to prevent abuse
4. **Model Protection**: Use environment variables for sensitive paths
5. **HTTPS**: Always enabled on both platforms

## Production Checklist

- [ ] Test locally with both frontend and backend
- [ ] Push to GitHub
- [ ] Deploy backend to Render
- [ ] Test backend API independently
- [ ] Deploy frontend to Vercel
- [ ] Update API URL in frontend env vars
- [ ] Test full workflow on production
- [ ] Monitor error logs
- [ ] Set up automated backups
- [ ] Configure CI/CD for automatic deployments
- [ ] Add disclaimer about AI limitations

## Support & Resources

- FastAPI Docs: https://fastapi.tiangolo.com
- React Docs: https://react.dev
- Render Docs: https://render.com/docs
- Vercel Docs: https://vercel.com/docs
- PyTorch: https://pytorch.org

## Next Steps

1. **Monitor Performance**: Use Render/Vercel dashboards
2. **Gather Feedback**: Collect user feedback
3. **Improve Model**: Retrain with more data
4. **Add Features**:
   - User authentication
   - Result history
   - Batch processing
   - Export reports
5. **Scale**: Upgrade plans as traffic increases

---

**Last Updated**: January 2026
**Status**: Ready for Production
