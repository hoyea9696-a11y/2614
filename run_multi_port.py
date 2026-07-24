import subprocess
import time
import sys
import os
import socket

PORTS = [8001, 8002, 8003, 8000]  # 子系統先起，主頁面（8000）最後
LOG_DIR = os.path.join(os.getcwd(), "logs")


def _pids_on_port(port):
    """回傳目前在這個埠口 LISTENING 的所有 PID（可能有殘留的舊行程不只一個）。"""
    pids = set()
    try:
        output = subprocess.check_output(
            f"netstat -ano | findstr :{port}", shell=True
        ).decode(errors="ignore")
    except Exception:
        # findstr 沒找到任何行會回非 0（CalledProcessError），代表埠口沒人用
        return pids
    for line in output.splitlines():
        parts = line.split()
        # 只認「本機這個埠口 + LISTENING」，避免把連到別處 :port 的連線也算進來
        if "LISTENING" in line and len(parts) >= 5 and parts[1].endswith(f":{port}"):
            pids.add(parts[-1])
    return pids


def _can_bind(port):
    """試綁一下確認 OS 真的把埠口放掉了——taskkill 之後埠口未必立刻可用
    （行程 teardown／socket TIME_WAIT 都會讓 uvicorn bind 失敗）。綁完立刻關，
    未曾 accept 過連線的 listening socket 在 Windows 上會即時釋放，不留 TIME_WAIT。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def free_port(port, timeout=15):
    """殺光佔用埠口的行程，並且真的等到埠口可以重新 bind 才回傳。
    這是修掉「重啟時撞 WinError 10048 導致 server 起不來」的關鍵：以前殺完馬上
    bind，OS 還沒把埠口放掉就失敗。"""
    for pid in _pids_on_port(port):
        subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
        print(f"已清理埠口 {port} (PID: {pid})")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pids_on_port(port) and _can_bind(port):
            return True
        time.sleep(0.3)
    # 超時：盡力而為，回報埠口是否至少沒人 LISTENING 了
    return not _pids_on_port(port)


def wait_until_listening(port, proc, timeout=30):
    """啟動後確認伺服器真的起來了（開始 LISTENING）。行程中途自己死掉（例如
    bind 失敗、import 出錯）就提早回 False，不用傻等到 timeout。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False  # 行程已結束，肯定沒起來
        if _pids_on_port(port):
            return True
        time.sleep(0.3)
    return False


def _spawn(port):
    """實際 Popen 一個 uvicorn 子行程（不含清埠／等待）。"""
    venv_python = os.path.join(os.getcwd(), "venv", "Scripts", "python.exe")
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable

    # 移除 --reload 以確保連線穩定，並統一使用 127.0.0.1
    cmd = [python_exe, "-m", "uvicorn", "main:app", "--host", "127.0.0.1",
           "--port", str(port), "--log-level", "warning"]
    server_dir = os.path.join(os.getcwd(), "server")

    # CREATE_NO_WINDOW：幫子行程開一個「隱藏的新 console」，不是掛在這個
    # 終端機的 console 上，關掉終端機視窗（右上 X）Windows 只會對還掛在原本
    # console 的行程送 CTRL_CLOSE_EVENT，不會動到這個子行程——訓練跑到一半、
    # 或使用者卡在訓練/存模型確認畫面時，才不會因為手滑關了終端機視窗就整個
    # 斷線。Ctrl+C 不受影響，仍是靠下面 except KeyboardInterrupt 裡的
    # proc.terminate() 主動關閉。
    #
    # 一開始改用 DETACHED_PROCESS（完全沒有 console），結果 uvicorn 內部自己
    # 又會 spawn 出其他 python 子行程，這些孫行程沒指定 console flag，
    # Windows 預設規則是「父行程沒 console 時就自動開一個新的」，變成每個
    # port 都跳出一個 CMD 視窗。CREATE_NO_WINDOW 讓子行程有 console（只是
    # 不顯示視窗），孫行程會直接繼承這個隱藏 console，就不會再跳視窗。
    #
    # 子行程沒有原本的 console 可以繼承 stdout/stderr/stdin，改寫進
    # logs/port_<port>.log，stdin 也要明確給 DEVNULL——留給預設值會讓
    # Windows 的 CreateProcess 卡住。
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"port_{port}.log")
    log_file = open(log_path, "a", encoding="utf-8")
    creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(
        cmd, cwd=server_dir, stdin=subprocess.DEVNULL,
        stdout=log_file, stderr=log_file, creationflags=creationflags
    )


def start_server(port, attempts=2):
    """清埠 → 啟動 → 確認真的 LISTENING；起不來就重試（預設共兩次）。
    回傳的行程若還是死的，交給主監控迴圈回報，不會再無限重試卡住。"""
    for attempt in range(1, attempts + 1):
        free_port(port)
        print(f"正在啟動埠口 {port} 的伺服器...")
        proc = _spawn(port)
        if wait_until_listening(port, proc):
            return proc
        print(f"埠口 {port} 啟動失敗（第 {attempt}/{attempts} 次，詳見 logs/port_{port}.log）")
        try:
            proc.terminate()
        except Exception:
            pass
        time.sleep(1.0)
    print(f"埠口 {port} 連續啟動失敗，跳過。請檢查 logs/port_{port}.log")
    return proc  # 回傳最後一次（已死）的行程，讓監控知道它沒起來


if __name__ == "__main__":
    processes = []

    try:
        print("\n" + "=" * 50)
        print("啟動多埠混合動力伺服器中 (子系統優先)...")
        print("=" * 50)

        for p in PORTS:
            processes.append(start_server(p))

        alive = [PORTS[i] for i, pr in enumerate(processes) if pr.poll() is None]
        print("\n" + "=" * 50)
        if 8000 in alive:
            print("所有頁面已獨立啟動！")
        else:
            print("啟動完成（部分埠口未成功，見上方訊息）")
        print("-> 主外殼: http://localhost:8000")
        print("-> 資料集: Port 8001")
        print("-> 標記區: Port 8002")
        print("-> 訓練區: Port 8003")
        print("=" * 50)
        print(f"log 檔位置: {LOG_DIR}")
        print("按下 Ctrl+C 可關閉所有伺服器（直接關掉此視窗不會中斷伺服器，")
        print("伺服器會繼續在背景執行，下次啟動時會自動清掉舊行程）")

        # 監控 + 自我修復：某個埠口的伺服器死掉時，不要只印警告後放著不管——那會讓對應
        # 頁面一直卡在「拒絕連線」，直到使用者手動全部重開（這是先前反覆遇到的症狀）。
        # 這裡改成自動把死掉的子服務重新拉起（start_server 內部會先 free_port 清乾淨再綁），
        # 但連續重啟都失敗（例如程式本身壞掉、一起來就 crash）就不要無限 respawn 洗版，
        # 達到 MAX_RESTARTS 後對該埠口停手、只留警告，避免 crash-loop 把整台機器打爆。
        MAX_RESTARTS = 3
        restart_counts = {p: 0 for p in PORTS}
        while True:
            for i, proc in enumerate(processes):
                port = PORTS[i]
                if proc.poll() is None:
                    restart_counts[port] = 0  # 還活著：把連續失敗計數歸零
                    continue
                if restart_counts[port] >= MAX_RESTARTS:
                    continue  # 這個埠口已放棄自動重啟，交給使用者處理，不再洗版
                restart_counts[port] += 1
                print(f"警告: Port {port} 的伺服器已停止，自動重啟中"
                      f"（第 {restart_counts[port]}/{MAX_RESTARTS} 次，詳見 logs/port_{port}.log）...")
                processes[i] = start_server(port)
                if processes[i].poll() is None:
                    restart_counts[port] = 0
                    print(f"Port {port} 已自動重啟成功。")
                else:
                    print(f"Port {port} 自動重啟失敗。")
            # 只有「全部都死、且每個埠口都用完重啟次數」才真的收工；否則持續監控。
            all_gave_up = all(
                proc.poll() is not None and restart_counts[PORTS[i]] >= MAX_RESTARTS
                for i, proc in enumerate(processes)
            )
            if all_gave_up:
                print("所有伺服器皆已停止且無法自動重啟，啟動器結束。")
                break
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n正在關閉所有伺服器...")
        for proc in processes:
            try:
                proc.terminate()
            except Exception:
                pass
        print("已成功關閉。")
