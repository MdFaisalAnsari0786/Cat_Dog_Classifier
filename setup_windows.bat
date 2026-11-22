@echo off
setlocal
cd /d "%~dp0"
echo Creating the virtual environment with Python 3.10...
py -3.10 -m venv .venv
if errorlevel 1 (
  echo ERROR: Python 3.10 was not found. Check py -0 in Command Prompt.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto :failed
pip install -r requirements.txt
if errorlevel 1 goto :failed
echo.
echo Setup complete. Next, run train_and_run_windows.bat.
pause
exit /b 0
:failed
echo.
echo ERROR: Package installation failed. Read the error shown above.
pause
exit /b 1
