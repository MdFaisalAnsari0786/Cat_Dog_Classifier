@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo ERROR: Virtual environment is missing.
  echo First double-click setup_windows.bat.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
if errorlevel 1 goto :failed

echo.
echo STEP 1/3: Downloading and preparing the Cats vs Dogs dataset...
python download_data.py
if errorlevel 1 goto :failed

echo.
echo STEP 2/3: Training the CNN / MobileNetV2 model...
python train.py
if errorlevel 1 goto :failed

if not exist models\cat_dog_model.keras (
  echo ERROR: Training ended but the model file was not created.
  pause
  exit /b 1
)

echo.
echo STEP 3/3: Starting the Streamlit app...
streamlit run app.py
pause
exit /b 0
:failed
echo.
echo ERROR: A step failed. Read the last error shown above and share it for help.
pause
exit /b 1
