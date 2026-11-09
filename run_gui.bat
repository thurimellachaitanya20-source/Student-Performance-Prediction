@echo off
echo ========================================================
echo   Launching EduPredict - Desktop Native GUI Suite
echo ========================================================
echo.

IF EXIST ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    python gui_app.py
) ELSE (
    python gui_app.py
)

pause
