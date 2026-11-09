@echo off
echo ========================================================
echo   Launching EduPredict - Student Performance AI Web App
echo ========================================================
echo.

IF EXIST ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    python -m streamlit run app.py
) ELSE (
    python -m streamlit run app.py
)

pause
