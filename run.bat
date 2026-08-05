@echo off
echo =========================================================
echo  Starting Translator Rephraser API (FastAPI)
echo =========================================================
echo.

call venv\Scripts\activate.bat

REM Uvicorn is the ASGI server standard
echo Starting server on http://0.0.0.0:5005 ...
uvicorn app:app --host 0.0.0.0 --port 5005 --workers 4

pause
