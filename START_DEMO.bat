@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel% equ 0 (
    set "PYTHON=py -3"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python 3 was not found. Install Python 3.9 or newer and enable "Add Python to PATH".
        pause
        exit /b 1
    )
    set "PYTHON=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating a local Python environment...
    %PYTHON% -m venv .venv
    if errorlevel 1 (
        echo Could not create the environment. Check that Python 3.9 or newer is installed.
        pause
        exit /b 1
    )
)

echo Checking demo packages...
".venv\Scripts\python.exe" -c "import cv2, ultralytics" >nul 2>nul
if errorlevel 1 (
    echo Installing demo packages. This may take several minutes...
    ".venv\Scripts\python.exe" -m pip install -r safety_demo_requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed. Check the error above and try again.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" human_webcam_demo.py --source 0 --weights yolov8n.pt --imgsz 480
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
