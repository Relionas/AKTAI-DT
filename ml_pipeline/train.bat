@echo off
REM Quick start script for training ML pipeline
REM Team: AKTAI

echo ========================================
echo FORTEBANK VIOLATION DETECTION
echo ML Training Pipeline
echo Team: AKTAI
echo ========================================
echo.

cd /d "C:\Users\vanac\OneDrive\Desktop\FINAL_PROJECT\ml_pipeline"

echo Activating virtual environment...
call ..\.venv\Scripts\activate.bat

echo.
echo Installing required packages...
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics timm albumentations scikit-learn pandas openpyxl pyyaml tqdm

echo.
echo ========================================
echo Starting training pipeline...
echo ========================================
python run_training_pipeline.py

pause
