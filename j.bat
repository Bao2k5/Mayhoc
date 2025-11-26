@echo off
REM Run from project root. Creates same behavior for cmd users.
cd /d "%~dp0"
REM Activate venv for cmd if available
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)
python -m streamlit run app.py --server.port 8502 --server.headless true
exit /b %ERRORLEVEL%
