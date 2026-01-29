# 🩺 Chest Disease Detection using Deep Learning

A full-stack AI-powered application for detecting multiple chest diseases from X-ray images using PyTorch deep learning. The system features a React frontend (deployed on Vercel) and FastAPI backend (deployed on Render).

## ✨ Features

- 🚀 **Full-Stack Application** - Complete frontend & backend ready for production
- 🔬 **Multi-Disease Detection** - Identifies 15 different chest conditions
- 📱 **Responsive UI** - Works seamlessly on desktop, tablet, and mobile
- 🎯 **Real-time Predictions** - Get results in seconds
- 📊 **Probability Scores** - View confidence for each disease
- ⚡ **Fast API** - Optimized for quick predictions
- 🔒 **Production Ready** - CORS, error handling, validation
- 📚 **Well Documented** - Complete guides and examples
- 🐳 **Docker Support** - Easy containerized deployment
- ☁️ **Cloud Ready** - Pre-configured for Render & Vercel

## 📊 Detected Diseases (15 Classes)

| #   | Disease       | #   | Disease            |
| --- | ------------- | --- | ------------------ |
| 1   | Atelectasis   | 9   | Infiltration       |
| 2   | Cardiomegaly  | 10  | Mass               |
| 3   | Consolidation | 11  | No Finding         |
| 4   | Edema         | 12  | Nodule             |
| 5   | Effusion      | 13  | Pleural_Thickening |
| 6   | Emphysema     | 14  | Pneumonia          |
| 7   | Fibrosis      | 15  | Pneumothorax       |
| 8   | Hernia        | -   | -                  |

## 🚀 Quick Start

### Local Development (Windows)

```bash
setup.bat
cd backend && python main.py      # Terminal 1
cd frontend && npm start           # Terminal 2
```

### Local Development (macOS/Linux)

```bash
bash setup.sh
cd backend && python main.py       # Terminal 1
cd frontend && npm start           # Terminal 2
```

Visit: **API** (http://localhost:8000/docs) | **App** (http://localhost:3000)

## 📁 Project Structure

```
├── backend/              # FastAPI Backend (Render)
├── frontend/             # React Frontend (Vercel)
├── models/               # Trained PyTorch models
├── QUICK_START.md        # Quick start guide
├── DEPLOYMENT_GUIDE.md   # Full deployment guide
├── TESTING_GUIDE.md      # Testing guide
└── DEPLOYMENT_CHECKLIST.md
```

## 🛠 Tech Stack

| Component  | Technology                          |
| ---------- | ----------------------------------- |
| Backend    | FastAPI, PyTorch, Uvicorn           |
| Frontend   | React 18, Tailwind CSS, Axios       |
| Deployment | Render (backend), Vercel (frontend) |
| Container  | Docker, Docker Compose              |

## 📖 Documentation

- **[Quick Start](QUICK_START.md)** - Get running in 5 minutes
- **[Full Deployment Guide](DEPLOYMENT_GUIDE.md)** - Complete setup
- **[Testing Guide](TESTING_GUIDE.md)** - How to test
- **[Deployment Checklist](DEPLOYMENT_CHECKLIST.md)** - Pre-launch

## 🌐 API Endpoints

| Method | Endpoint         | Description        |
| ------ | ---------------- | ------------------ |
| GET    | `/health`        | Health check       |
| GET    | `/labels`        | Available diseases |
| POST   | `/predict`       | Single prediction  |
| POST   | `/predict-batch` | Batch predictions  |

## 🚀 Deploy Now

1. **Push to GitHub**: `git add . && git commit -m "Deploy" && git push`
2. **Deploy Backend**: Create Render Web Service
3. **Deploy Frontend**: Import to Vercel

See [Deployment Guide](DEPLOYMENT_GUIDE.md) for step-by-step instructions.

## 💰 Cost: ~$25/month (optional - free tier available)

## 👥 Team Members

- Abhay
- Akash
- Vipul
- Prathmesh

## ⚠️ Disclaimer

This is an **educational tool only**. Always consult healthcare professionals for proper diagnosis.

---

**Status**: ✅ Production Ready | **Updated**: January 2026 | 🩺 [Get Started](QUICK_START.md)
