#!/bin/bash

# Chest X-Ray Prediction System - Setup Script

echo "🩺 Chest X-Ray Disease Detection Setup"
echo "======================================"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.9 or higher."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Setup Backend
echo ""
echo "Setting up Backend..."
cd backend

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

source venv/bin/activate
pip install -r requirements.txt
echo "✓ Backend dependencies installed"

cd ..

# Setup Frontend
echo ""
echo "Setting up Frontend..."
cd frontend

if ! command -v node &> /dev/null; then
    echo "⚠️  Node.js not found. Please install Node.js 14+ from https://nodejs.org"
else
    npm install
    echo "✓ Frontend dependencies installed"
fi

cd ..

echo ""
echo "======================================"
echo "✅ Setup Complete!"
echo ""
echo "To run locally:"
echo ""
echo "Terminal 1 - Backend:"
echo "  cd backend"
echo "  source venv/bin/activate  # or venv\\Scripts\\activate on Windows"
echo "  python main.py"
echo ""
echo "Terminal 2 - Frontend:"
echo "  cd frontend"
echo "  npm start"
echo ""
echo "API Docs: http://localhost:8000/docs"
echo "Frontend: http://localhost:3000"
echo ""
echo "For deployment, see DEPLOYMENT_GUIDE.md"
