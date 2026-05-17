@echo off
title BDL CHATBOT Server
echo.
echo ============================================
echo   BDL CHATBOT - Starting Local Server...
echo ============================================
echo.

:: Check if Ollama is running
ollama list >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama does not seem to be running.
    echo           Please start Ollama first: ollama serve
    echo.
    pause
    exit /b 1
)

:: Start the chatbot server
cd /d "%~dp0"
python "C:\Users\admin\Downloads\BDL ChatBot-main\scripts\frontend.py" --port 1234

pause
