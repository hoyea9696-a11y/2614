import subprocess
import time
import sys
import os

def kill_port_process(port):
    """強行殺掉佔用特定埠口的進程"""
    try:
        if sys.platform == "win32":
            output = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode()
            for line in output.splitlines():
                if "LISTENING" in line:
                    pid = line.strip().split()[-1]
                    subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
                    print(f"已清理埠口 {port} (PID: {pid})")
    except:
        pass

LOG_DIR = os.path.join(os.getcwd(), "logs")

def start_server(port):
    kill_port_process(port)
    print(f"正在啟動埠口 {port} 的伺服器...")

    venv_python = os.path.join(os.getcwd(), "venv", "Scripts", "python.exe")
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable

    # 移除 --reload 以確保連線穩定，並統一使用 127.0.0.1
    cmd = [python_exe, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"]
    server_dir = os.path.join(os.getcwd(), "server")

    # DETACHED_PROCESS：子行程完全不掛在這個 console 上，關掉終端機視窗
    # （右上 X）Windows 只會送 CTRL_CLOSE_EVENT 給還掛在這個 console 的行程，
    # 不會動到已經 detach 的子行程——訓練跑到一半、或使用者卡在訓練/存模型
    # 確認畫面時，才不會因為手滑關了終端機視窗就整個斷線。Ctrl+C 不受影響，
    # 仍是靠下面 except KeyboardInterrupt 裡的 proc.terminate() 主動關閉。
    #
    # 子行程沒有 console 可以繼承 stdout/stderr/stdin，改寫進 logs/port_<port>.log，
    # stdin 也要明確給 DEVNULL——留給預設值（繼承目前 console 的 handle）會讓
    # Windows 的 CreateProcess 卡住，因為 DETACHED_PROCESS 沒有 console 可以
    # duplicate 這個 handle 過去。
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"port_{port}.log")
    log_file = open(log_path, "a", encoding="utf-8")
    creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(
        cmd, cwd=server_dir, stdin=subprocess.DEVNULL,
        stdout=log_file, stderr=log_file, creationflags=creationflags
    )

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
        print(f"log 檔位置: {LOG_DIR}")
        print("按下 Ctrl+C 可關閉所有伺服器（直接關掉此視窗不會中斷伺服器，")
        print("伺服器會繼續在背景執行，下次啟動時會自動清掉舊行程）")
        
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
