@echo off
REM Chest X-Ray Prediction System - Setup Script for Windows

echo.
echo Chest X-Ray Disease Detection Setup
echo ====================================
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.9 or higher.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✓ %PYTHON_VERSION% found
echo.

REM Setup Backend
echo Setting up Backend...
cd backend

if not exist "venv" (
    python -m venv venv
    echo ✓ Virtual environment created
)

call venv\Scripts\activate.bat
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install backend dependencies
    pause
    exit /b 1
)
echo ✓ Backend dependencies installed

cd ..

REM Setup Frontend
echo.
echo Setting up Frontend...
cd frontend

node --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Node.js not found. Please install Node.js 14+ from https://nodejs.org
    echo After installing Node.js, run: npm install
) else (
    call npm install
    if errorlevel 1 (
        echo ❌ Failed to install frontend dependencies
        pause
        exit /b 1
    )
    echo ✓ Frontend dependencies installed
)

cd ..

echo.
echo ====================================
echo ✅ Setup Complete!
echo ====================================
echo.
echo To run locally:
echo.
echo Terminal 1 - Backend:
echo   cd backend
echo   venv\Scripts\activate.bat
echo   python main.py
echo.
echo Terminal 2 - Frontend:
echo   cd frontend
echo   npm start
echo.
echo API Docs: http://localhost:8000/docs
echo Frontend: http://localhost:3000
echo.
echo For deployment, see DEPLOYMENT_GUIDE.md
echo.
pause
