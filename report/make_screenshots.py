"""Local integration test that also captures the report screenshots:
starts the API and the UI, sends real requests, screenshots the UI with
headless Edge and renders the real API console output as an image.

    python report/make_screenshots.py [trip_id]
"""
import html
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_report import screenshot, SHOTS  # noqa: E402

PY = sys.executable
TRIP = sys.argv[1] if len(sys.argv) > 1 else "00000065"


def wait(url, t=90):
    for _ in range(t):
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def main():
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PATHSNAP_API_URL="http://127.0.0.1:8000")
    api_log = open(ROOT / "logs" / "api_console_capture.log", "w", encoding="utf-8")
    api = subprocess.Popen([PY, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"],
                           cwd=ROOT, stdout=api_log, stderr=subprocess.STDOUT, env=env)
    ui = subprocess.Popen([PY, "-m", "streamlit", "run", "ui/app.py", "--server.port", "8501",
                           "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                          cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    try:
        assert wait("http://127.0.0.1:8000/health"), "API did not start"
        print(requests.get("http://127.0.0.1:8000/trips").json())
        r = requests.post("http://127.0.0.1:8000/predict", json={"trip_id": TRIP}).json()
        print("metrics", r["metrics"])
        assert wait("http://127.0.0.1:8501"), "UI did not start"
        screenshot(f"http://localhost:8501/?trip={TRIP}&run=1", str(SHOTS / "ui_screenshot.png"),
                   w=1500, h=1720, wait_ms=8000, wait_text="All metrics returned by the API", full_page=True)
        time.sleep(2)
    finally:
        ui.terminate(); api.terminate()
        time.sleep(1)
        api_log.close()
    lines = (ROOT / "logs" / "api_console_capture.log").read_text(encoding="utf-8").splitlines()[-28:]
    body = "\n".join(html.escape(l) for l in lines)
    page = ROOT / "report" / "screenshots" / "api_console.html"
    page.write_text(f"""<html><body style="margin:0;background:#0c0c0c">
<div style="background:#1f1f1f;color:#ddd;font:13px Segoe UI;padding:6px 10px">Command Prompt - uvicorn api.main:app --host 0.0.0.0 --port 8000  (Laptop 1)</div>
<pre style="color:#cccccc;font:13px Consolas,monospace;padding:10px;white-space:pre-wrap;margin:0">{body}</pre>
</body></html>""", encoding="utf-8")
    screenshot(page.resolve().as_uri(), str(SHOTS / "api_console.png"), w=1500, h=430, wait_ms=1000)
    print("screenshots written to", SHOTS)


if __name__ == "__main__":
    main()
