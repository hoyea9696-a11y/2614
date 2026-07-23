import subprocess
import time
import sys
import os

def _listening_pids(port):
    """回傳目前正在 LISTENING 這個埠口的 PID 清單，沒有就回空清單。"""
    try:
        output = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode(errors="ignore")
    except subprocess.CalledProcessError:
        return []
    pids = set()
    for line in output.splitlines():
        if "LISTENING" in line:
            pids.add(line.strip().split()[-1])
    return list(pids)

def kill_port_process(port, wait_timeout=10):
    """強行殺掉佔用特定埠口的進程，並且真的等到埠口釋放才回傳。

    taskkill 送出去不代表程序馬上死透、socket 馬上釋放——舊版本砍完立刻讓
    uvicorn 綁定，若釋放得比預期慢就會撞上 WinError 10048（埠口仍被佔用），
    導致這個埠口的新伺服器反而啟動失敗，比原本殘留的舊行程還麻煩。"""
    if sys.platform != "win32":
        return
    pids = _listening_pids(port)
    if not pids:
        return
    for pid in pids:
        subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
        print(f"已清理埠口 {port} (PID: {pid})")

    deadline = time.time() + wait_timeout
    while time.time() < deadline:
        if not _listening_pids(port):
            return
        time.sleep(0.5)
    print(f"警告：埠口 {port} 在 {wait_timeout} 秒後仍被佔用，接下來啟動伺服器可能會失敗")

def start_server(port):
    kill_port_process(port)
    print(f"正在啟動埠口 {port} 的伺服器...")
    
    venv_python = os.path.join(os.getcwd(), "venv", "Scripts", "python.exe")
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable
    
    # 移除 --reload 以確保連線穩定，並統一使用 127.0.0.1
    cmd = [python_exe, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"]
    server_dir = os.path.join(os.getcwd(), "server")
    return subprocess.Popen(cmd, cwd=server_dir, stdout=None, stderr=None)

if __name__ == "__main__":
    # 核心修正：先啟動子服務 (8001-8003)，最後才啟動主頁面 (8000)
    ports = [8001, 8002, 8003, 8000]
    processes = []

    try:
        print("\n" + "="*50)
        print("啟動多埠混合動力伺服器中 (子系統優先)...")
        print("="*50)
        
        for p in ports:
            proc = start_server(p)
            processes.append(proc)
            time.sleep(2.0) # 給予 2 秒穩定時間
        
        print("\n" + "="*50)
        print("所有頁面已獨立啟動！")
        print("-> 主外殼: http://localhost:8000")
        print("-> 資料集: Port 8001")
        print("-> 標記區: Port 8002")
        print("-> 訓練區: Port 8003")
        print("="*50)
        print("按下 Ctrl+C 可關閉所有伺服器")
        
        # 保持腳本運行並監控進程
        while True:
            for i, proc in enumerate(processes):
                if proc.poll() is not None:
                    print(f"警告: Port {ports[i]} 的伺服器已停止。")
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n正在關閉所有伺服器...")
        for proc in processes:
            proc.terminate()
        print("已成功關閉。")
