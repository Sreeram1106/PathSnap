@echo off
REM LAPTOP 2: start the evaluator UI. Usage:  run_ui.bat http://<laptop1-ip>:8000
cd /d %~dp0
if not "%~1"=="" set PATHSNAP_API_URL=%~1
python -m streamlit run ui/app.py --server.port 8501
