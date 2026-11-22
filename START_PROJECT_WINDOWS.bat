@echo off
setlocal
cd /d "%~dp0"
title Cat vs Dog - Setup, Train and Run

echo ================================================
echo      CAT VS DOG IMAGE CLASSIFIER
 echo ================================================
echo.

if not exist app.py (
  echo ERROR: app.py was not found.
  echo Please run this file from inside the Cat_Dog_Classifier folder.
  pause
  exit /b 1
)

if not exist .venv\Scripts\python.exe (
  echo [1/5] Creating Python 3.10 virtual environment...
  py -3.10 -m venv .venv
  if errorlevel 1 goto :failed
) else (
  echo [1/5] Python virtual environment already exists.
)

call .venv\Scripts\activate.bat
if errorlevel 1 goto :failed

echo.
echo [2/5] Installing/checking required packages...
python -m pip install --upgrade pip
if errorlevel 1 goto :failed
python -m pip install -r requirements.txt
if errorlevel 1 goto :failed

if not exist data\train\cats\*.jpg (
  echo.
  echo [3/5] Downloading and preparing the dataset.
  echo This download is about 787 MB. Keep the internet connected.
  python download_data.py
  if errorlevel 1 goto :failed
) else (
  echo.
  echo [3/5] Prepared training images already exist. Skipping dataset download.
)

if not exist models\cat_dog_model.keras (
  echo.
  echo [4/5] Training the CNN / MobileNetV2 model.
  echo This can take a while, especially on a CPU. Do not close this window.
  python train.py --batch-size 16
  if errorlevel 1 goto :failed
) else (
  echo.
  echo [4/5] Trained model already exists. Skipping training.
)

if not exist models\cat_dog_model.keras (
  echo ERROR: The model file was not created.
  goto :failed
)
if not exist models\class_names.json (
  echo ERROR: models\class_names.json is missing.
  goto :failed
)

echo.
echo [5/5] Starting the Cat vs Dog web app...
python -m streamlit run app.py
if errorlevel 1 goto :failed
pause
exit /b 0

:failed
echo.
echo ================================================
echo A step failed. Please send a screenshot of the
 echo error lines shown above this message.
echo ================================================
pause
exit /b 1
