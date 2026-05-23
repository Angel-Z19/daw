@echo off
REM Lanzador DAW — Windows
REM Uso: doble clic en run_gui.bat  o  ejecutar desde CMD

SET DIR=%~dp0
SET VENV=%DIR%.venv

IF NOT EXIST "%VENV%\Scripts\python.exe" (
    echo Entorno virtual no encontrado. Creando...
    python -m venv "%VENV%"
    "%VENV%\Scripts\pip" install --quiet PyQt6 sounddevice numpy
    echo Dependencias instaladas.
)

cd /d "%DIR%"
"%VENV%\Scripts\python.exe" gui.py %*
pause
