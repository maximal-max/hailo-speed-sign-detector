@echo off
chcp 65001 >nul
title YOLO Training + Hailo Export Umgebung (.venv)
setlocal enableextensions
REM ==========================================================================
REM   Umgebung: .venv
REM   Python:   3.11.x
REM   Ziel:     Reproduzierbares YOLO Training + Hailo Export
REM ==========================================================================
REM Nutzt das aktuelle Verzeichnis, in dem die .bat liegt
SET VENV_PATH=.venv

echo ================================================
echo Erstelle optimierte AI Trainingsumgebung in: %CD%
echo ================================================

REM Python vorhanden?
where python >nul 2>&1 || (
    echo FEHLER: Python nicht gefunden! Bitte installieren und zum PATH hinzufuegen.
    exit /b 1
)

REM Python-Version pruefen (3.11 empfohlen)
python --version 2>&1 | findstr /R "3\.11\." >nul || (
    echo WARNUNG: Python 3.11 wird empfohlen. Gefundene Version:
    python --version
    echo Druecke eine Taste zum Fortfahren oder schliesse das Fenster zum Abbrechen.
    pause >nul
)

REM venv erstellen oder aktualisieren
IF EXIST "%VENV_PATH%" (
    echo Umgebung existiert bereits - Aktualisiere Pakete...
) ELSE (
    echo Erstelle neue venv ^(.venv^)...
    python -m venv "%VENV_PATH%"
    if errorlevel 1 (
        echo FEHLER: venv konnte nicht erstellt werden.
        exit /b 1
    )
)

REM Aktivierung der Umgebung
call "%VENV_PATH%\Scripts\activate.bat"
if errorlevel 1 (
    echo FEHLER: venv konnte nicht aktiviert werden.
    exit /b 1
)

echo.
echo Upgrade pip, setuptools und wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 ( echo FEHLER bei pip/setuptools/wheel Upgrade & exit /b 1 )

echo.
echo ================================
echo 1) PyTorch ^(CUDA 12.1 Build^)
echo ================================
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 ( echo FEHLER bei PyTorch-Installation & exit /b 1 )

echo.
echo ================================
echo 2) Core Scientific Stack
echo ================================
pip install numpy==1.26.4 pandas==2.2.3 scipy==1.14.1 matplotlib==3.9.2 polars==1.34.0 pyyaml tqdm
if errorlevel 1 ( echo FEHLER bei Core Scientific Stack & exit /b 1 )

echo.
echo ================================
echo 3) YOLO ^& Vision Toolkit
echo ================================
REM opencv-python (mit GUI-Support) benoetigt fuer PC_application.py (imshow, namedWindow)
REM NICHT opencv-python-headless verwenden - das unterstuetzt keine Fenster!
pip install ultralytics opencv-python==4.10.0.84 "albumentations>=2.0.8" lapx==0.5.11.post1 seaborn==0.13.2 pillow==11.0.0
if errorlevel 1 ( echo FEHLER bei YOLO/Vision Toolkit & exit /b 1 )

echo.
echo ================================
echo 4) Training + Monitoring Tools
echo ================================
pip install lightning torchmetrics tensorboard wandb rich psutil coloredlogs
if errorlevel 1 ( echo FEHLER bei Training/Monitoring Tools & exit /b 1 )

echo.
echo ================================
echo 5) Hailo Export / ONNX Stack
echo ================================
pip install onnx==1.20.0 onnxruntime-gpu==1.20.1 onnxsim==0.4.36
if errorlevel 1 ( echo FEHLER bei ONNX Stack & exit /b 1 )

pip install onnx-graphsurgeon --index-url https://pypi.ngc.nvidia.com
if errorlevel 1 ( echo FEHLER bei onnx-graphsurgeon ^(NVIDIA NGC Index^) & exit /b 1 )

echo.
echo ================================
echo 6) Tracking/Inference/Capture
echo ================================
REM mss: Screen-Capture fuer PC_application.py (SOURCE="screen")
pip install supervision==0.23.0 norfair==2.3.0 mss
if errorlevel 1 ( echo FEHLER bei Tracking/Inference/Capture & exit /b 1 )

echo.
echo =================================================
echo Teste die Umgebung...
echo =================================================
python -c "import torch; print(f'Torch:         {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
python -c "import ultralytics; print(f'Ultralytics:   {ultralytics.__version__}')"
python -c "import numpy; print(f'Numpy:         {numpy.__version__} (Soll <2.0!)')"
python -c "import onnx; print(f'ONNX:          {onnx.__version__}')"
python -c "import onnxruntime; print(f'onnxruntime:   {onnxruntime.__version__}')"
python -c "import cv2; print(f'OpenCV:        {cv2.__version__} | GUI: {hasattr(cv2, \"namedWindow\")}')"
python -c "import mss; print(f'mss:           OK')"

echo.
echo ================================================
echo Alle Checks bestanden. Umgebung ist einsatzbereit.
echo Nutze 'deactivate' zum Beenden der Umgebung.
echo ================================================
pause