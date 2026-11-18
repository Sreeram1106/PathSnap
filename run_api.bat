@echo off
REM LAPTOP 1: start the PathSnap API on all network interfaces, port 8000.
REM Optional: set PATHSNAP_EVAL_PARTITION=evaluator_test   (folder name under data\raw)
cd /d %~dp0
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
echo Laptop 1 IP addresses (give one of these to Laptop 2):
ipconfig | findstr /i "IPv4"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
