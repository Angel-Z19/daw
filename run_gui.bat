@echo off
cd /d "%~dp0"
echo --- Iniciando entorno DAW con Python 3.11 ---
call .venv\Scripts\activate
python gui.py
pause