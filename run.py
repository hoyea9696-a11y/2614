import subprocess
import webbrowser
import time
import os
import sys
import urllib.request
import socket

PORTS = [8000, 8001, 8002, 8003]  # 主外殼 + 三個子系統，四個都要就緒才算真的可用


def wait_for_all_servers(ports, timeout=60):
    """等『所有』埠口都回應 HTTP 200 才回 True。以前只等 8000（主外殼），結果 8000
    先起來、瀏覽器就打開了，但子頁面（8001/8002/8003）還沒好，使用者一進去就看到
    內容區「拒絕連線」。改成四個埠都通了才開瀏覽器，開啟時就不會再撞到未就緒的子頁面。"""
    start_time = time.time()
    print(f"Waiting for all services {ports} to initialize...")
    pending = set(ports)
    while pending and time.time() - start_time < timeout:
        for port in list(pending):
            try:
                with urllib.request.urlopen(f"http://localhost:{port}", timeout=3) as response:
                    if response.getcode() == 200:
                        pending.discard(port)
            except Exception:
                pass
        if pending:
            time.sleep(1)
    return not pending

def launch():
    print("Starting AI Detector Offline Service...")
    
    # Path to the virtual environment python
    current_dir = os.getcwd()
    venv_python = os.path.join(current_dir, "venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        venv_python = sys.executable

    # Start Backend in a separate process
    cmd = [venv_python, "run_multi_port.py"]
    backend_process = subprocess.Popen(cmd, cwd=current_dir)
    
    # Server health check：四個埠都就緒才開瀏覽器，避免一進去就看到子頁面「拒絕連線」
    url = "http://localhost:8000"
    if wait_for_all_servers(PORTS, timeout=90):
        print(f"All services are ready. Opening interface...")
        webbrowser.open(url)
    else:
        print("Error: Some services failed to start within the timeout period.")
        print("仍會開啟主頁面，未就緒的子頁面稍後重新整理即可。")
        webbrowser.open(url)

    try:
        backend_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down services...")
        backend_process.terminate()

if __name__ == "__main__":
    launch()
