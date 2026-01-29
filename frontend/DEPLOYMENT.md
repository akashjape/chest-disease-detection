# Frontend Deployment Guide

## Local Development

1. **Setup Environment**

```bash
cd frontend
npm install
```

2. **Create .env.local**

```
REACT_APP_API_URL=http://localhost:8000
```

3. **Run Development Server**

```bash
npm start
```

Visit: `http://localhost:3000`

## Deployment on Vercel

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Add frontend UI"
git push origin main
```

### Step 2: Deploy to Vercel

1. Go to https://vercel.com
2. Click "New Project"
3. Import your GitHub repository
4. Configure:
   - **Framework**: Create React App
   - **Root Directory**: `./frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `build`

### Step 3: Environment Variables

In Vercel project settings:

```
REACT_APP_API_URL=https://your-render-api.onrender.com
```

### Step 4: Redeploy

After setting env vars, redeploy the project for changes to take effect

## Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── LoadingSpinner.js
│   │   └── Results.js
│   ├── App.js
│   ├── App.css
│   ├── index.js
│   └── index.css
├── package.json
├── tailwind.config.js
├── postcss.config.js
└── .env.local
```

## Key Features

- ✅ Drag & drop image upload
- ✅ Real-time image preview
- ✅ Disease probability predictions
- ✅ Top 5 predictions highlighted
- ✅ Full prediction list with confidence scores
- ✅ Responsive design (mobile & desktop)
- ✅ Loading states and error handling
- ✅ Medical disclaimer

## API Integration

The frontend communicates with the backend API:

```javascript
const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

// POST /predict - Send image for analysis
// Returns: predictions object with disease probabilities
```

## Build & Deployment Checklist

- [ ] Update `REACT_APP_API_URL` in `.env.production`
- [ ] Test locally with backend running
- [ ] Test production build: `npm run build`
- [ ] Push to GitHub
- [ ] Deploy to Vercel
- [ ] Configure environment variables in Vercel
- [ ] Test deployed version with backend API
- [ ] Monitor performance on Vercel dashboard

## Troubleshooting

1. **API Connection Error**
   - Check backend is running and accessible
   - Verify `REACT_APP_API_URL` is correct
   - Check CORS configuration in backend

2. **Build Fails**
   - Clear `node_modules`: `rm -rf node_modules && npm install`
   - Check Node.js version: `node -v` (should be 14+)

3. **Slow Load Times**
   - Enable caching in Vercel settings
   - Optimize images before upload
   - Consider CDN for static assets
