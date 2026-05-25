import subprocess
import webbrowser
import time
import os
import sys
import urllib.request
import socket

def wait_for_server(url, timeout=30):
    """Wait for the server to be responsive."""
    start_time = time.time()
    print(f"Waiting for services to initialize at {url}...")
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url) as response:
                if response.getcode() == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False

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
    
    # Server health check
    url = "http://localhost:8000"
    if wait_for_server(url, timeout=60):
        print(f"Server is ready. Opening interface...")
        webbrowser.open(url)
    else:
        print("Error: Server failed to start within the timeout period.")

    try:
        backend_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down services...")
        backend_process.terminate()

if __name__ == "__main__":
    launch()
