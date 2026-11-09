@echo off
echo ========================================================
echo   Training Student Performance Machine Learning Models
echo ========================================================
echo.

IF EXIST ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    python src\train.py
) ELSE (
    python src\train.py
)

pause
