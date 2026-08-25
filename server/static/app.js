// State Management
let singleFiles = [];
let batchFiles = [];
let currentIndex = { single: 0, batch: 0 };
let currentPage = 0;
let ITEMS_PER_PAGE = 20;
let trainChart = null;
let trainInterval = null;
let isPaused = false;
// ------------------------------
// 全域狀態與通訊
// ------------------------------
let activeDataset = JSON.parse(localStorage.getItem('ai_detector_active_ds') || 'null');
// 僅供「一般模式」聊天使用：只有本次頁面收到使用者點選資料集的事件才會有值。
// 不讀 localStorage，避免一般模式把上次工作區誤當成本次訓練的選擇。
let generalChatDataset = null;

// 監聽來自 iframe 的消息 (跨 Port 通訊)
window.addEventListener('message', (event) => {
    const data = event.data;
    console.log("Shell Received Message:", data);

    if (data.type === 'DATASET_SELECTED') {
        activeDataset = data.dataset;
        localStorage.setItem('ai_detector_active_ds', JSON.stringify(activeDataset));
        generalChatDataset = data.dataset;
        
        // 轉發給標記模組 iframe (Port 8002)
        const labelingIframe = document.getElementById('iframe-labeling');
        if (labelingIframe && labelingIframe.contentWindow) {
            labelingIframe.contentWindow.postMessage({
                type: 'DATASET_SELECTED',
                dataset: activeDataset
            }, '*');
        }

        // 「目前資料集資訊」應跟著使用者剛點選的卡片更新，而不是停在上一次訓練
        // 寫入 workspace_state.json 的資料集。
        const roboflowPanel = document.getElementById('roboflow-status-panel');
        if (roboflowPanel && roboflowPanel.style.display !== 'none') {
            refreshRoboflowStatus();
        }
        
        showAlert(`已切換工作區: ${activeDataset.name}`, "success");
    }
    
    if (data.type === 'REQUEST_DS') {
        // 子頁面請求當前資料集 (例如重新整理時)
        if (activeDataset) {
            event.source.postMessage({
                type: 'DATASET_SELECTED',
                dataset: activeDataset
            }, '*');
        }
    }
    
    if (data.type === 'SWITCH_PAGE') {
        switchPage(event.data.pageId);
    }
});

// 頁面加載時初始化
document.addEventListener('DOMContentLoaded', () => {
    updateSystemMonitor();
    setInterval(updateSystemMonitor, 5000);
    setInterval(checkAutoLabelStatus, 3000); // 每一秒檢查一次自動標記狀態
    
    // 恢復上次瀏覽的頁面
    const lastPage = localStorage.getItem('ai_detector_active_page') || 'datasets';
    switchPage(lastPage);

    initManagerAgentSession();
    initChatWindowResize();
    initChatImageDrop();
    resumeTrainingWatchIfRunning();
});

// Agent 做推論吃的是「圖片的絕對路徑」（skill_infer 的 image_path），不是檔案本身。
// 這裡把上傳／拖放的圖片存到 data/uploads/，再把回傳的絕對路徑填進輸入框，使用者
// 就不用自己去檔案總管複製路徑。
async function handleAgentImagePick(files) {
    if (!files || !files.length) return;
    const file = files[0];
    if (!file.type.startsWith('image/')) {
        appendMessage('只支援圖片檔案（jpg / png / bmp / webp）', 'error');
        return;
    }

    const loadingId = appendMessage(`⏳ 正在上傳 ${file.name}…`, 'ai', true);
    try {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch('/api/agent/upload', { method: 'POST', body: formData });
        const data = await res.json();
        removeMessage(loadingId);

        if (data.status !== 'success') {
            appendMessage(data.message || '上傳失敗', 'error');
            return;
        }

        const input = document.getElementById('ai-chat-input');
        const existing = input.value.trim();
        // 沒打字時直接補一句可以送出的指令，打過字就只把路徑接在後面
        input.value = existing
            ? `${existing} ${data.path}`
            : `用這張圖片做推論驗證：${data.path}`;
        input.focus();
        showAlert(`已上傳 ${data.filename}，路徑已填入輸入框`, 'success');
    } catch (e) {
        removeMessage(loadingId);
        appendMessage('上傳失敗: ' + e.message, 'error');
    } finally {
        const picker = document.getElementById('ai-chat-file-input');
        if (picker) picker.value = '';  // 清掉才能重複選同一個檔案
    }
}

// 整個聊天視窗都可以當拖放區（不用精準拖到輸入框那一小條）。dragover 一定要
// preventDefault，否則瀏覽器會用預設行為「直接開啟圖片」取代我們的處理。
function initChatImageDrop() {
    const zone = document.getElementById('ai-chat-window');
    const input = document.getElementById('ai-chat-input');
    if (!zone || !input) return;

    const originalBorder = input.style.border;
    const highlight = on => {
        input.style.border = on ? '1px dashed #10b981' : originalBorder;
    };

    zone.addEventListener('dragover', e => { e.preventDefault(); highlight(true); });
    zone.addEventListener('dragleave', e => {
        // 只有真正離開整個視窗才取消高亮，避免在子元素之間移動時一直閃
        if (!zone.contains(e.relatedTarget)) highlight(false);
    });
    zone.addEventListener('drop', e => {
        e.preventDefault();
        highlight(false);
        const files = e.dataTransfer && e.dataTransfer.files;
        if (files && files.length) handleAgentImagePick(files);
    });
}

// LLM 聊天視窗固定在右下角，縮放把手放在左上角，往左上拖就是放大。
// 只改 inline style、不寫入任何儲存，重新整理頁面自然回到 HTML 預設的 350x500。
function initChatWindowResize() {
    const chatWindow = document.getElementById('ai-chat-window');
    const handle = document.getElementById('ai-chat-resize-handle');
    if (!chatWindow || !handle) return;

    handle.addEventListener('pointerdown', (e) => {
        e.preventDefault();
        handle.setPointerCapture(e.pointerId);
        const startX = e.clientX, startY = e.clientY;
        const startW = chatWindow.offsetWidth, startH = chatWindow.offsetHeight;

        const onMove = (ev) => {
            const w = Math.min(Math.max(startW + (startX - ev.clientX), 320), window.innerWidth - 48);
            const h = Math.min(Math.max(startH + (startY - ev.clientY), 420), window.innerHeight - 120);
            chatWindow.style.width = `${w}px`;
            chatWindow.style.height = `${h}px`;
        };
        const onUp = () => {
            handle.removeEventListener('pointermove', onMove);
            handle.removeEventListener('pointerup', onUp);
        };
        handle.addEventListener('pointermove', onMove);
        handle.addEventListener('pointerup', onUp);
    });
}

// 使用者在 confirm() 彈窗跳出來、還沒按下去之前關掉終端機／重新整理頁面，
// SSE 連線會直接斷掉、彈窗也隨之消失，但 SqliteSaver 其實還留著暫停在
// interrupt() 的 checkpoint（見 server/multi_agent.py 的 train_confirm_gate／
// save_offer_gate）。頁面重新載入時打 /api/agent/pending 用同一個 thread_id
// 查一次，卡著的話直接重新問一次、接著送 /api/agent/resume 繼續跑，不用
// 讓使用者自己發現「怎麼訓練確認就這樣消失了」。
async function checkPendingAgentInterrupt() {
    const threadId = sessionStorage.getItem(MANAGER_AGENT_THREAD_KEY);
    if (!threadId) return;

    const modelTypeEl = document.getElementById('manager-agent-engine-select');
    const modelType = modelTypeEl ? modelTypeEl.value : 'gemini';

    try {
        const res = await fetch(`/api/agent/pending?thread_id=${encodeURIComponent(threadId)}&model_type=${encodeURIComponent(modelType)}`);
        const data = await res.json();
        if (data.pending) {
            const pendingPayload = {
                ...(data.payload || {}),
                interrupt_id: data.interrupt_id
            };
            await finishOrResumeAgentTurn(
                { traceRefs: null, finalReply: null, sawError: false, interruptPayload: pendingPayload },
                threadId,
                modelType
            );
        }
    } catch (e) {
        console.error('檢查未完成的 Agent 確認狀態失敗:', e);
    }
}

// Deep Agent 模式的同一件事，查的是自己的 thread_id 與 /api/agent/deep/pending。
// 兩個模式各自有可能卡著一個等待確認的中斷，所以進頁面時兩邊都要查一次。
async function checkPendingDeepAgentInterrupt() {
    const threadId = sessionStorage.getItem(DEEP_AGENT_THREAD_KEY);
    if (!threadId) return;

    const modelTypeEl = document.getElementById('manager-agent-engine-select');
    const modelType = modelTypeEl ? modelTypeEl.value : 'gemini';

    try {
        const res = await fetch(`/api/agent/deep/pending?thread_id=${encodeURIComponent(threadId)}&model_type=${encodeURIComponent(modelType)}`);
        const data = await res.json();
        if (data.pending) {
            const pendingPayload = {
                ...(data.payload || {}),
                interrupt_id: data.interrupt_id
            };
            await finishOrResumeAgentTurn(
                { traceRefs: null, finalReply: null, sawError: false, interruptPayload: pendingPayload },
                threadId,
                modelType,
                '/api/agent/deep/resume'
            );
        }
    } catch (e) {
        console.error('檢查未完成的 Deep Agent 確認狀態失敗:', e);
    }
}

let lastAutoLabelStatus = "idle";
async function checkAutoLabelStatus() {
    try {
        const res = await fetch('/api/ai/auto_label_status');
        const data = await res.json();
        
        if (data.status === "finished" && lastAutoLabelStatus === "running") {
            document.getElementById('auto-label-panel').style.display = 'none';
            showAlert(`✅ 自動標記完成！已成功為「${data.dataset}」標記了 ${data.labeled_count} 張圖片。`, "success");
            
            const labelingIframe = document.getElementById('iframe-labeling');
            if (labelingIframe && labelingIframe.contentWindow) {
                if (activeDataset && activeDataset.name === data.dataset) {
                    labelingIframe.contentWindow.postMessage({ type: 'DATASET_SELECTED', dataset: activeDataset }, '*');
                }
            }
        } else if (data.status === "error" && lastAutoLabelStatus === "running") {
            document.getElementById('auto-label-panel').style.display = 'none';
            showAlert(`❌ 自動標記失敗: ${data.error}`, "error");
        } else if (data.status === "running") {
            // 更新懸浮面板進度
            const panel = document.getElementById('auto-label-panel');
            panel.style.display = 'block';
            
            const processed = data.processed || 0;
            const total = data.total || 0;
            const labeled = data.labeled_count || 0;
            const percent = total > 0 ? Math.round((processed / total) * 100) : 0;
            
            document.getElementById('auto-label-bar').style.width = percent + '%';
            document.getElementById('auto-label-percent').innerText = percent + '%';
            document.getElementById('auto-label-stats').innerText = `掃描進度: ${processed} / ${total}`;
            document.getElementById('auto-label-count').innerText = `已標記: ${labeled} 張`;
        }
        
        lastAutoLabelStatus = data.status;
    } catch (e) {
        console.error("Auto label status poll error:", e);
    }
}

function switchPage(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(p => p.classList.remove('active'));
    const pageEl = document.getElementById('page-' + pageId);
    if (pageEl) {
        pageEl.classList.add('active');
        setTimeout(() => { pageEl.style.opacity = '1'; }, 10);
    }
    document.querySelectorAll('.nav-item').forEach(item => {
        if (item.getAttribute('onclick').includes(pageId)) item.classList.add('active');
    });
    if (pageId === 'train') loadTrainDatasets();
    
    // 記憶當前頁面
    localStorage.setItem('ai_detector_active_page', pageId);
}

function showAlert(message, type = 'info') {
    const alertBox = document.getElementById('alert-box');
    alertBox.innerText = message;
    alertBox.style.display = 'block';
    alertBox.style.background = (type === 'error') ? '#ff4757' : '#111';
    setTimeout(() => { alertBox.style.display = 'none'; }, 5000);
}

function openLightbox(src) {
    if (!src) return;
    const overlay = document.getElementById('lightbox-overlay');
    document.getElementById('lightbox-img').src = src;
    overlay.style.display = 'flex';
}
function closeLightbox() {
    document.getElementById('lightbox-overlay').style.display = 'none';
}
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeLightbox(); });

// ------------------------------
// 其他功能保持簡潔
// ------------------------------
async function updateSystemMonitor() {
    try {
        const res = await fetch('/api/system/info');
        const data = await res.json();
        
        // 更新 CPU & RAM
        const cpuEl = document.getElementById('sys-cpu');
        cpuEl.innerText = data.cpu || "-";
        cpuEl.title = data.cpu || "";
        document.getElementById('sys-ram').innerText = data.ram || "-";
        
        // 更新模式與 GPU 資訊
        const modeEl = document.getElementById('sys-mode');
        const gpuEl = document.getElementById('sys-gpu');
        
        if (data.cuda) {
            modeEl.innerText = "GPU ACCEL";
            modeEl.style.background = "#10b981"; // 綠色代表效能
            gpuEl.innerText = data.gpu || "Generic GPU";
            gpuEl.style.color = "#34d399";
        } else {
            modeEl.innerText = "CPU ONLY";
            modeEl.style.background = "#475569"; // 灰色代表基礎模式
            gpuEl.innerText = "No NVIDIA GPU detected";
            gpuEl.style.color = "#64748b";
        }
    } catch (e) {
        console.error("Monitor error:", e);
    }
}

// ------------------------------
// 訓練邏輯
// ------------------------------

function initChart() {
    const ctx = document.getElementById('trainChart').getContext('2d');
    if (trainChart) trainChart.destroy();

    trainChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [], // 輪次標籤
            datasets: [
                {
                    label: 'mAP50 (%)',
                    data: [],
                    borderColor: '#3b82f6',
                    borderWidth: 3,
                    fill: false,
                    pointRadius: 4,
                    yAxisID: 'y'
                },
                {
                    label: 'Loss',
                    data: [],
                    borderColor: '#ef4444',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 2,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { 
                    type: 'linear', 
                    position: 'left', 
                    min: 0, 
                    max: 100, 
                    title: { display: true, text: 'mAP (%)' } 
                },
                y1: { 
                    type: 'linear', 
                    position: 'right', 
                    min: 0, 
                    grid: { drawOnChartArea: false }, 
                    title: { display: true, text: 'Loss' } 
                }
            },
            plugins: {
                legend: { display: true, position: 'top' }
            }
        }
    });
}

async function loadTrainDatasets() {
    try {
        const res = await fetch('/api/datasets/list', { cache: 'no-store' });
        const data = await res.json();
        const select = document.getElementById('train-dataset-select');
        const current = select.value;
        select.innerHTML = '<option value="">-- 請選擇已匯入的資料集 --</option>';
        if (data.status === 'success') {
            data.datasets.forEach(ds => {
                const opt = document.createElement('option');
                opt.value = ds.name;
                opt.textContent = `${ds.name} (${ds.imgCount || 0} 張圖片)`;
                select.appendChild(opt);
            });
            if (current) select.value = current;
        }
    } catch (e) { console.error('Load datasets error:', e); }
}

async function startTraining() {
    if (trainInterval) return;

    const startBtn = document.getElementById('start-train-btn');
    const datasetName = document.getElementById('train-dataset-select').value;

    if (!datasetName) {
        showAlert("請先選擇要訓練的資料集", "error");
        return;
    }

    startBtn.disabled = true;
    startBtn.innerText = "啟動中...";

    const formData = new FormData();
    formData.append('dataset_name', datasetName);
    formData.append('model_name', document.getElementById('train-model-name').value);
    formData.append('variant', document.getElementById('train-model-variant').value);
    formData.append('epochs', document.getElementById('train-epochs').value);
    formData.append('patience', document.getElementById('train-patience').value);
    formData.append('batch', document.getElementById('train-batch').value);
    formData.append('imgsz', document.getElementById('train-imgsz').value);

    try {
        const res = await fetch('/api/train/start', { method: 'POST', body: formData });
        const result = await res.json();

        if (result.status === "error") {
            showAlert(result.message, "error");
            startBtn.disabled = false;
            startBtn.innerText = "啟動訓練";
            return;
        }

        startTrainingWatch();
    } catch (e) {
        showAlert("啟動訓練失敗: " + e.message, "error");
        startBtn.disabled = false;
        startBtn.innerText = "啟動訓練";
    }
}

// 進度監看抽成獨立函式：不管訓練是從表單按鈕、聊天視窗關鍵字還是 Manager Agent
// 啟動的，訓練中心都掛同一套進度圖與終止按鈕，不會再出現「訓練在跑但頁面沒動作、
// 也沒地方可以終止」的狀況。
function startTrainingWatch() {
    if (trainInterval) return; // 已經在監看，不重複掛
    document.getElementById('training-progress-card').style.display = 'block';
    initChart();

    trainInterval = setInterval(async () => {
        if (isPaused) return;
        try {
            const r = await fetch('/api/train/status');
            const d = await r.json();

            const progressBar = document.getElementById('train-progress-bar');
            const statusText = document.getElementById('train-status-text');
            const startBtn = document.getElementById('start-train-btn');

            if (d.status === "training") {
                // 優先用後端算好的細粒度 percent（epoch 內就會動）；舊資料沒有才退回用 epoch 比例
                const p = (d.percent != null) ? Math.round(d.percent) : Math.round((d.current_epoch / d.total_epochs) * 100);
                progressBar.style.width = `${p}%`;
                statusText.innerText = `訓練進行中: 第 ${d.current_epoch} / ${d.total_epochs} 輪 (${p}%)`;

                if (d.metrics && d.metrics.length > 0) {
                    trainChart.data.labels = d.metrics.map(m => m.epoch);
                    trainChart.data.datasets[0].data = d.metrics.map(m => m.map);
                    trainChart.data.datasets[1].data = d.metrics.map(m => m.loss);
                    trainChart.update();
                }
            } else if (d.status === "completed" || d.status === "error" || d.status === "stopped") {
                clearInterval(trainInterval);
                trainInterval = null;
                startBtn.disabled = false;
                startBtn.innerText = "啟動訓練";
                statusText.innerText = d.status === "completed" ? "訓練已完成"
                    : (d.status === "stopped" ? "❌ 訓練已終止" : "訓練發生錯誤");
                if (d.status === "completed") {
                    finishTraining("訓練成功完成", "success");
                    loadModels();  // 剛部署的模型立刻出現在驗證／診斷下拉，不用手動重整
                }
            }
        } catch (e) { console.error(e); }
    }, 2000);
}

// 頁面載入（或聊天觸發訓練後）發現後端已有訓練在跑，就把監看掛上去。
async function resumeTrainingWatchIfRunning() {
    try {
        const r = await fetch('/api/train/status');
        const d = await r.json();
        if (d.status === 'training') {
            const startBtn = document.getElementById('start-train-btn');
            if (startBtn) { startBtn.disabled = true; startBtn.innerText = "訓練進行中..."; }
            startTrainingWatch();
        }
    } catch (e) { /* 後端還沒起來就算了，之後的觸發點會再試 */ }
}

function finishTraining(msg, type) {
    showAlert(msg, type);
    // 顯示評價
    document.getElementById('train-evaluation').style.display = 'block';
}

function terminateTraining() {
    if (confirm("確定終止？")) {
        fetch('/api/train/stop', { method: 'POST' });
        clearInterval(trainInterval);
        trainInterval = null;
        document.getElementById('start-train-btn').disabled = false;
        document.getElementById('train-status-text').innerText = "❌ 訓練已終止";
    }
}

function togglePause() {
    isPaused = !isPaused;
    document.getElementById('pause-train-btn').innerText = isPaused ? "恢復" : "暫停";
}

// ------------------------------
// 驗證與同步邏輯 [RESTORED]
// ------------------------------
function syncParams(el, selector) {
    const val = (el.type === 'checkbox') ? el.checked : el.value;
    document.querySelectorAll(selector).forEach(target => {
        if (target !== el) {
            if (target.type === 'checkbox') target.checked = val;
            else target.value = val;
            
            // 更新數值顯示
            const display = target.parentElement.querySelector('span[class$="-val"]');
            if (display) display.innerText = val;
        }
    });
    
    // 如果是當前操作的元素，也要更新它的顯示
    const currentDisplay = el.parentElement.querySelector('span[class$="-val"]');
    if (currentDisplay) currentDisplay.innerText = val;
}

function handleFilesUpload(input, type) {
    const files = Array.from(input.files);
    if (type === 'single') {
        singleFiles = files;
        if (files.length > 0) {
            const reader = new FileReader();
            reader.onload = e => document.getElementById('prediction-img-single').src = e.target.result;
            reader.readAsDataURL(files[0]);
        }
    } else {
        batchFiles = files.filter(f => f.name.match(/\.(jpg|jpeg|png)$/i));
        showAlert(`已選擇 ${batchFiles.length} 張影像`, "success");
    }
}

async function startVerification(type) {
    const model = document.querySelector('.model-selector').value;
    const imgsz = document.querySelector('.imgsz-selector').value;
    const conf = document.querySelector('.conf-range').value;
    const iou = document.querySelector('.iou-range').value;
    const augment = document.querySelector('.augment-toggle').checked;

    if (type === 'single' && singleFiles.length === 0) return showAlert("請先選擇影像", "error");

    const formData = new FormData();
    formData.append('model_name', model);
    formData.append('imgsz', imgsz);
    formData.append('conf', conf);
    formData.append('iou', iou);
    formData.append('augment', augment);
    
    if (type === 'single') formData.append('file', singleFiles[0]);

    showAlert("啟動檢測中...", "info");
    try {
        const res = await fetch('/api/verify/single', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.status === 'success') {
            document.getElementById('prediction-img-single').src = data.image_url + '?t=' + Date.now();
            showAlert("檢測完成", "success");
        } else {
            showAlert(data.message, "error");
        }
    } catch (e) { showAlert("連線失敗", "error"); }
}

async function startBatchVerification() {
    if (batchFiles.length === 0) return showAlert("請先選擇資料夾", "error");
    // ... 批量邏輯實作 ...
    showAlert("批量檢測功能執行中...", "info");
}

function toggleTooltip(id) {
    const box = document.getElementById('tooltip-' + id);
    box.style.display = (box.style.display === 'block') ? 'none' : 'block';
}

function clearFiles() {
    if (confirm("確定清空伺服器快取？")) {
        fetch('/api/clear', { method: 'POST' }).then(() => showAlert("快取已清除", "success"));
    }
}

// ------------------------------
// 初始化
// ------------------------------
window.onload = () => {
    switchPage('datasets');
    setInterval(updateSystemMonitor, 30000);
};

// ------------------------------
// AI Chat Assistant Logic
// ------------------------------
let chatMessages = [];
let roboflowReadiness = null;

function renderRoboflowStatus(data) {
    const content = document.getElementById('roboflow-status-content');
    if (!content) return;
    if (!data || data.status !== 'success') {
        content.textContent = '無法讀取 Roboflow 狀態。';
        return;
    }
    const rows = [
        ['資料集', data.dataset_name || '未選擇'],
        ['圖片數', data.image_count ?? '—'],
        ['健康度', data.health_score ?? '—'],
        ['標註覆蓋率', data.coverage === undefined || data.coverage === null ? '—' : `${data.coverage}%`],
        ['類別模式', data.single_class ? '單類別' : '多類別（支援明確名稱對應）'],
        ['最近 mAP50', data.last_map50 ?? '—'],
        ['Roboflow API Key', data.api_key_configured ? '已設定' : '未設定'],
        ['目前判斷', data.reason || '—'],
    ];
    let html = rows.map(([label, value]) =>
        `<div><span style="color:#94a3b8;">${escapeHtml(label)}：</span>${escapeHtml(value)}</div>`
    ).join('');
    if (data.latest_operation) {
        const op = data.latest_operation;
        html += `<div style="margin-top:8px; padding-top:8px; border-top:1px solid rgba(148,163,184,.18);">` +
            `<div style="color:#7dd3fc; font-weight:600;">最近一次正式合併</div>` +
            `<div>${escapeHtml(op.source || op.source_url || 'Roboflow')} ・ 新增 ${escapeHtml(op.added_images ?? 0)} 張</div>` +
            `<div>${escapeHtml(op.before_images ?? '—')} → ${escapeHtml(op.total_images_now ?? '—')} 張</div>` +
            `</div>`;
    }
    content.innerHTML = html;
}

async function refreshRoboflowStatus() {
    const content = document.getElementById('roboflow-status-content');
    if (content) content.textContent = '讀取中…';
    try {
        const params = activeDataset && activeDataset.name
            ? `?dataset_name=${encodeURIComponent(activeDataset.name)}`
            : '';
        const res = await fetch(`/api/agent/roboflow/readiness${params}`);
        const data = await res.json();
        roboflowReadiness = data;
        renderRoboflowStatus(data);
        return data;
    } catch (e) {
        roboflowReadiness = null;
        if (content) content.textContent = '讀取狀態失敗，請確認伺服器是否啟動。';
        return null;
    }
}

function agentConfirmationTitle(payload) {
    if (payload && payload.action_type === 'roboflow_proposal') return 'Agent 的 Roboflow 評估結果';
    if (payload && payload.action_type === 'model_save') return '要將模型加入模型庫嗎？';
    if (payload && payload.action_type === 'training') return '要開始模型訓練嗎？';
    if (payload && payload.model_name) return '要將模型加入模型庫嗎？';
    return '要開始模型訓練嗎？';
}

function renderRoboflowProposalMeta(proposal) {
    const candidate = proposal.candidate || {};
    const mappings = (proposal.class_mapping || []).map(item =>
        `${escapeHtml(item.source_name)} → ${escapeHtml(item.target_name)}`
    ).join('<br>');
    const risks = (proposal.risks || []).length
        ? (proposal.risks || []).map(item => `• ${escapeHtml(item)}`).join('<br>')
        : '未發現額外風險';
    return `
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:7px 12px;">
            <div><span style="color:#64748b;">目標資料集</span><br><b>${escapeHtml(proposal.dataset_name)}</b></div>
            <div><span style="color:#64748b;">比較候選</span><br><b>${escapeHtml(proposal.candidate_count || 0)} 個</b></div>
            <div><span style="color:#64748b;">建議來源</span><br><b>${escapeHtml(candidate.name || candidate.project)}</b></div>
            <div><span style="color:#64748b;">版本／授權</span><br><b>v${escapeHtml(candidate.version)} ・ ${escapeHtml(candidate.license || '未標示')}</b></div>
            <div><span style="color:#64748b;">可安全加入</span><br><b style="color:#4ade80;">${escapeHtml(proposal.usable_images)} 張</b></div>
            <div><span style="color:#64748b;">加入後預估</span><br><b>${escapeHtml(proposal.baseline_images)} → ${escapeHtml(proposal.estimated_total_images)} 張</b></div>
        </div>
        <div style="margin-top:10px; padding-top:9px; border-top:1px solid rgba(148,163,184,.18);">
            <div style="color:#7dd3fc; font-weight:600; margin-bottom:4px;">類別對應</div>
            ${mappings || '沒有可用的類別對應'}
        </div>
        <div style="margin-top:9px;">
            <div style="color:#fbbf24; font-weight:600; margin-bottom:4px;">跳過與風險</div>
            <div>無標註 ${escapeHtml(proposal.unlabeled_images || 0)} 張 ・ 無對應 ${escapeHtml(proposal.unmapped_images || 0)} 張 ・ 重複 ${escapeHtml(proposal.duplicate_images || 0)} 張</div>
            <div style="margin-top:4px;">${risks}</div>
        </div>
        ${candidate.url ? `<a href="${escapeHtml(candidate.url)}" target="_blank" rel="noopener" style="display:inline-block; margin-top:9px; color:#60a5fa;">查看 Roboflow 來源 ↗</a>` : ''}
    `;
}

function showAgentConfirmation(payload) {
    return new Promise(resolve => {
        const modal = document.getElementById('agent-confirm-modal');
        const title = document.getElementById('agent-confirm-title');
        const reason = document.getElementById('agent-confirm-reason');
        const meta = document.getElementById('agent-confirm-meta');
        const approve = document.getElementById('agent-confirm-approve');
        const cancel = document.getElementById('agent-confirm-cancel');
        title.textContent = agentConfirmationTitle(payload);
        reason.textContent = (payload && payload.reason) || '此操作需要你的確認。';
        const isTraining = Boolean(payload && payload.action_type === 'training');
        const isRoboflow = Boolean(payload && payload.action_type === 'roboflow_proposal');
        if (isTraining) {
            const p = payload.training_params || {};
            const operationId = payload.operation_id || '—';
            meta.style.display = 'block';
            meta.textContent =
                `操作編號：${operationId}\n` +
                `資料集：${p.dataset_name || '—'}\n` +
                `模型權重：${p.variant || '—'}；Epochs：${p.epochs ?? '—'}\n` +
                `圖片尺寸：${p.imgsz ?? '—'}；Batch：${p.batch ?? '—'}\n` +
                `實驗名稱：${p.model_name || '—'}\n` +
                '確認後，Model Agent 只能使用以上參數。';
        } else if (isRoboflow && payload.proposal) {
            meta.style.display = 'block';
            meta.innerHTML = renderRoboflowProposalMeta(payload.proposal);
        } else {
            meta.style.display = 'none';
            meta.innerHTML = '';
        }
        approve.textContent = isRoboflow ? '接受並合併' : (isTraining ? '開始訓練' : '繼續');
        modal.style.display = 'flex';
        const finish = approved => {
            modal.style.display = 'none';
            approve.onclick = null;
            cancel.onclick = null;
            resolve(approved);
        };
        approve.onclick = () => finish(true);
        cancel.onclick = () => finish(false);
    });
}

async function showRoboflowTestPrompt() {
    if (!roboflowReadiness) await refreshRoboflowStatus();
    await showAgentConfirmation({
        action_type: 'roboflow_proposal',
        dataset_name: (roboflowReadiness && roboflowReadiness.dataset_name) || '範例資料集',
        reason: '這是介面預覽，不會呼叫 Roboflow、下載資料或修改正式資料集。',
        proposal: {
            dataset_name: (roboflowReadiness && roboflowReadiness.dataset_name) || 'bear',
            candidate_count: 3,
            candidate: { name: 'Bear Detection', project: 'bear-demo', version: 3, license: 'CC BY 4.0', url: 'https://universe.roboflow.com/' },
            usable_images: 110,
            baseline_images: (roboflowReadiness && roboflowReadiness.image_count) || 234,
            estimated_total_images: ((roboflowReadiness && roboflowReadiness.image_count) || 234) + 110,
            class_mapping: [{ source_name: 'Bear', target_name: 'Bear' }],
            unlabeled_images: 4,
            unmapped_images: 8,
            duplicate_images: 12,
            risks: ['未對應類別將跳過：Polar-bear', '12 張重複圖片將跳過']
        }
    });
}

function onAiModelSelectChange() {
    // Manager Agent 底層要用哪個 LLM，用第二個下拉選單另外選——切到別的選項時就隱藏，
    // 避免使用者以為一般聊天視窗也能選 Manager Agent 的引擎
    const modelType = document.getElementById('ai-model-select').value;
    const engineBar = document.getElementById('manager-agent-engine-bar');
    const roboflowPanel = document.getElementById('roboflow-status-panel');
    // 推理引擎選單兩個 Agent 模式都要顯示（Deep Agent 底層同樣是這三個 LLM 之一）；
    // Roboflow 面板只有 Manager 模式有對應的後端狀態端點，維持只給它看。
    const isAgentMode = modelType === 'manager-agent' || modelType === 'deep-agent';
    engineBar.style.display = isAgentMode ? 'flex' : 'none';
    roboflowPanel.style.display = modelType === 'manager-agent' ? 'block' : 'none';
    if (modelType === 'manager-agent') refreshRoboflowStatus();
}

function toggleChat() {
    const chatWindow = document.getElementById('ai-chat-window');
    const btn = document.getElementById('ai-chat-btn');
    if (chatWindow.style.display === 'none') {
        chatWindow.style.display = 'flex';
        btn.style.transform = 'scale(0.9)';
    } else {
        chatWindow.style.display = 'none';
        btn.style.transform = 'scale(1)';
    }
}

async function sendChatMessage() {
    const input = document.getElementById('ai-chat-input');
    const text = input.value.trim();
    if (!text) return;
    
    const modelType = document.getElementById('ai-model-select').value;
    console.log(">>> [Browser] Sending chat request to platform:", modelType);
    appendMessage(text, 'user');
    input.value = '';
    
    // Add context if needed (e.g., current dataset info)
    const contextData = {
        activeDataset: generalChatDataset ? generalChatDataset.name : "None",
        currentTrainEpoch: trainChart && trainChart.data.labels.length > 0 ? trainChart.data.labels[trainChart.data.labels.length-1] : "Not training"
    };

    chatMessages.push({ role: 'user', content: text });
    persistChatHistory();

    const loadingId = appendMessage(`[${modelType.toUpperCase()}] 思考中...`, 'ai', true);

    // Manager Agent 走獨立的 /api/agent/chat（LangGraph + Tool，SSE 串流），跟原本的
    // /api/ai/chat（Gemini/Groq/Local 的一般對話，整包 JSON）分開，互不影響。Manager
    // Agent 底層要用哪個 LLM 是另一個下拉選單決定的（manager-agent-engine-select），
    // 跟一般聊天視窗的 modelType 是兩件事，不要混在一起。
    if (modelType === 'manager-agent') {
        await sendManagerAgentMessage(text, loadingId);
        return;
    }

    // Deep Agent 模式走第三條路徑 /api/agent/deep/chat（deepagents preset）。
    // 這個 return 一定要在下面 /api/ai/chat 的 fetch 之前——漏掉的話 model_type
    // 會被當成一般聊天的平台名稱送去 ai_service.py，那邊不認識就直接報錯。
    if (modelType === 'deep-agent') {
        await sendDeepAgentMessage(text, loadingId);
        return;
    }

    const requestBody = { messages: chatMessages, context_data: contextData, model_type: modelType };

    try {
        const res = await fetch('/api/ai/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });
        const data = await res.json();

        removeMessage(loadingId);

        if (data.status === 'success') {
            let reply = data.reply;

            // 檢查是否有健康度數據標籤 [HEALTH_DATA: {json}]
            // JSON 內含巢狀物件（class_distribution），要用貪婪匹配吃到最後一個大括號
            const healthMatch = reply.match(/\[HEALTH_DATA: ({.*})\]/);
            let healthJson = null;
            if (healthMatch) {
                try { healthJson = JSON.parse(healthMatch[1]); } catch (e) { console.warn('健康數據解析失敗:', e); }
            }
            if (healthJson) {
                reply = reply.replace(healthMatch[0], "").trimEnd(); // 移除標籤，保留文字
                appendMessage(reply, 'ai');
                appendHealthCard(healthJson);
            } else {
                appendMessage(reply, 'ai');
            }

            chatMessages.push({ role: 'assistant', content: reply });
            persistChatHistory();

            // 聊天關鍵字直接觸發的訓練：立刻把訓練中心的進度監看掛上，
            // 使用者切到訓練頁就能看到進度、也能按終止
            if (reply.includes('已直接啟動模型訓練')) resumeTrainingWatchIfRunning();
        } else {
            appendMessage("發生錯誤: " + data.message, 'error');
        }
    } catch (e) {
        removeMessage(loadingId);
        appendMessage("連線失敗", 'error');
    }
}

// Manager Agent 的對話記憶（LangGraph checkpointer）用 thread_id 對應到同一份
// state。thread_id／對話畫面／伺服器啟動識別碼都存在 sessionStorage：
//   - 重新整理頁面：sessionStorage 不會清，boot_id 也沒變 → 沿用同一個
//     thread_id、把之前的對話畫回來（見 initManagerAgentSession/renderChatHistory）。
//   - 重開終端機（重啟 server）：後端的 boot_id 會換一組新的，跟存著的不一樣
//     → 自動開一個新的 thread_id、清空對話，符合「重開終端就是新的一次」。
//   - 分頁關閉：sessionStorage 本來就會清，下次進站是全新對話。
// 也可以隨時按「開新對話」按鈕手動重置。
const MANAGER_AGENT_THREAD_KEY = 'manager_agent_thread_id';
const MANAGER_AGENT_BOOT_KEY = 'manager_agent_server_boot';
const MANAGER_AGENT_HISTORY_KEY = 'manager_agent_chat_history';

// Deep Agent 模式一定要用「自己的」thread_id，不能跟 Manager 模式共用一把：
// LangGraph 的 checkpointer 以 thread_id 為主鍵，Manager 模式存進去的是
// SupervisorState、Deep Agent 存的是 DeepAgentState，兩種 schema 輪流寫同一個
// thread 會在 resume 時讀到不匹配的欄位——而且症狀通常不是乾脆報錯，是靜默的
// 錯誤行為。後端的 checkpoint sqlite 檔也是分開的（deep_agent_checkpoints.sqlite）。
const DEEP_AGENT_THREAD_KEY = 'deep_agent_thread_id';

function getOrCreateThreadId() {
    let threadId = sessionStorage.getItem(MANAGER_AGENT_THREAD_KEY);
    if (!threadId) {
        threadId = crypto.randomUUID();
        sessionStorage.setItem(MANAGER_AGENT_THREAD_KEY, threadId);
    }
    return threadId;
}

function getOrCreateDeepThreadId() {
    let threadId = sessionStorage.getItem(DEEP_AGENT_THREAD_KEY);
    if (!threadId) {
        threadId = crypto.randomUUID();
        sessionStorage.setItem(DEEP_AGENT_THREAD_KEY, threadId);
    }
    return threadId;
}

// 每次對話有進展（送出訊息、收到回覆）就把 chatMessages 存進 sessionStorage，
// 這樣重新整理後 renderChatHistory 才能原封不動把對話泡泡畫回來。決策過程卡片
// （trace）跟健康度圖卡屬於過場用的即時資訊，不在還原範圍內。
function persistChatHistory() {
    try {
        sessionStorage.setItem(MANAGER_AGENT_HISTORY_KEY, JSON.stringify(chatMessages));
    } catch (e) {
        console.warn('儲存對話紀錄失敗:', e);
    }
}

function renderChatHistory() {
    const container = document.getElementById('ai-chat-history');
    if (!container) return;
    container.innerHTML = '';
    for (const m of chatMessages) {
        appendMessage(m.content, m.role === 'user' ? 'user' : 'ai');
    }
}

function resetManagerAgentThread() {
    sessionStorage.setItem(MANAGER_AGENT_THREAD_KEY, crypto.randomUUID());
    // 「開新對話」按鈕清的是整個聊天視窗（對話泡泡是三個模式共用的），所以兩個
    // Agent 模式的 thread_id 要一起換掉——只換一個會變成畫面清空了、但另一個模式
    // 的後端記憶還接著上一段對話。
    sessionStorage.setItem(DEEP_AGENT_THREAD_KEY, crypto.randomUUID());
    sessionStorage.removeItem(MANAGER_AGENT_HISTORY_KEY);
    chatMessages.length = 0;
    document.getElementById('ai-chat-history').innerHTML = '';
}

// 進頁面時初始化 Manager Agent 對話：先問後端 boot_id 判斷這是「重新整理」還是
// 「重開終端」，決定要還原舊對話還是開新的；還原的情況再順便檢查有沒有卡住的
// 訓練/存模型確認（interrupt）要接續。
async function initManagerAgentSession() {
    let currentBoot = null;
    try {
        const res = await fetch('/api/agent/boot');
        const data = await res.json();
        currentBoot = data.boot_id || null;
    } catch (e) {
        // 後端還沒起來或連不上：不要因為一次暫時性錯誤就把使用者的對話清掉，
        // 保守地沿用 sessionStorage 裡既有的內容還原。
        console.warn('取得伺服器啟動識別碼失敗，沿用既有對話:', e);
    }

    const storedBoot = sessionStorage.getItem(MANAGER_AGENT_BOOT_KEY);

    if (currentBoot && storedBoot && storedBoot !== currentBoot) {
        // boot_id 變了 = 重開過終端機（server 重啟）→ 開全新對話
        resetManagerAgentThread();
        sessionStorage.setItem(MANAGER_AGENT_BOOT_KEY, currentBoot);
        return;
    }

    if (currentBoot) sessionStorage.setItem(MANAGER_AGENT_BOOT_KEY, currentBoot);

    // 同一個 server（重新整理）或抓不到 boot_id：把上次的對話畫回來
    const saved = sessionStorage.getItem(MANAGER_AGENT_HISTORY_KEY);
    if (saved) {
        try {
            chatMessages = JSON.parse(saved);
            renderChatHistory();
        } catch (e) {
            console.warn('還原對話紀錄失敗:', e);
        }
    }

    checkPendingAgentInterrupt();
    checkPendingDeepAgentInterrupt();
}

// /api/agent/chat、/api/agent/resume 回應都是 SSE 串流（text/event-stream），逐個
// trace step 邊跑邊塞進決策過程卡片（createTraceCard/appendTraceStep），不用等整個
// Plan→Act→Observe→Decide 迴圈跑完才看到東西；中途失敗也能看到已經跑出來的步驟跟
// 錯誤原因。兩個端點的 SSE frame 格式一樣，解析邏輯抽成這個共用函式，resume 時
// 沿用同一張決策過程卡片（傳入既有的 traceRefs）繼續往下疊，不會另開一張新卡片。
const MANAGER_AGENT_STATUS_AFTER_NODE = {
    planner: 'Manager 正在評估計畫並安排下一步…',
    data_agent: 'Observer 正在檢視資料處理結果…',
    model_agent: 'Observer 正在檢視模型處理結果…',
    evaluation_agent: 'Observer 正在檢視評估結果…',
    observer: 'Manager 正在整合觀測結果並決策…'
};

function updateManagerAgentLoadingMessage(messageId, message) {
    if (!messageId || !message) return;
    const messageEl = document.getElementById(messageId);
    if (!messageEl) return;
    messageEl.textContent = `⏳ ${message}`;
    messageEl.style.opacity = '1';
}

function managerAgentStatusAfterStep(step) {
    if (step.node === 'manager' && step.decision) {
        const targetLabels = {
            data_agent: 'Data Agent 正在檢查資料集與標註…',
            model_agent: 'Model Agent 正在處理訓練或推論任務…',
            evaluation_agent: 'Evaluation Agent 正在評估模型結果…',
            finish: 'Manager 正在整理最終回覆…',
            ask_user: 'Manager 正在整理需要您確認的事項…'
        };
        return targetLabels[step.decision] || 'Manager 正在執行下一個工作階段…';
    }
    return MANAGER_AGENT_STATUS_AFTER_NODE[step.node] || '';
}

async function consumeAgentStream(res, traceRefs, loadingId = null) {
    let finalReply = null;
    let sawError = false;
    let interruptPayload = null;

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let idx;
        while ((idx = buffer.indexOf('\n\n')) >= 0) {
            const rawEvent = buffer.slice(0, idx);
            buffer = buffer.slice(idx + 2);
            if (!rawEvent.startsWith('data: ')) continue;

            const payload = JSON.parse(rawEvent.slice(6));
            if (payload.type === 'status') {
                updateManagerAgentLoadingMessage(loadingId, payload.message);
            } else if (payload.type === 'step') {
                updateManagerAgentLoadingMessage(loadingId, managerAgentStatusAfterStep(payload));
                if (!traceRefs) traceRefs = createTraceCard();
                appendTraceStep(traceRefs, payload);
            } else if (payload.type === 'final') {
                finalReply = payload.reply;
                if (traceRefs) finishTraceCard(traceRefs, payload.usage, payload.llm_calls);
                // 這回合 agent 可能把新訓練的模型存進了模型庫（save_offer_gate），重載一次
                // 下拉選單，剛存的模型不用手動重整頁面就會出現。loadModels 內含 null 防護，
                // 不在驗證中心頁面時會直接跳過。
                loadModels();
            } else if (payload.type === 'error') {
                sawError = true;
                if (!traceRefs) traceRefs = createTraceCard();
                appendTraceError(traceRefs, payload.message);
            } else if (payload.type === 'eval_charts') {
                appendEvalChartsCard(payload);
            } else if (payload.type === 'dataset_refresh') {
                await refreshDatasetViewsAfterAgentMerge(payload);
            } else if (payload.type === 'interrupt') {
                interruptPayload = {
                    ...(payload.payload || {}),
                    interrupt_id: payload.interrupt_id
                };
            }
        }
    }

    return { traceRefs, finalReply, sawError, interruptPayload };
}

async function refreshDatasetViewsAfterAgentMerge(payload) {
    // Roboflow has already committed files on the server when this event arrives.
    // Refresh both dataset cards and the training selector immediately.
    const datasetsIframe = document.getElementById('iframe-datasets');
    if (datasetsIframe && datasetsIframe.contentWindow) {
        datasetsIframe.contentWindow.postMessage({ type: 'REFRESH_DATASETS' }, '*');
    }
    await loadTrainDatasets();

    const datasetName = payload && payload.dataset_name;
    if (datasetName && activeDataset && activeDataset.name === datasetName) {
        const refreshed = {
            ...activeDataset,
            imgCount: payload.total_images_now ?? activeDataset.imgCount,
        };
        activeDataset = refreshed;
        localStorage.setItem('ai_detector_active_ds', JSON.stringify(refreshed));
        if (generalChatDataset && generalChatDataset.name === datasetName) {
            generalChatDataset = refreshed;
        }
    }
}

// 卡在 interrupt()（目前只有訓練前確認會觸發）時，用專案裡既有的瀏覽器原生
// confirm() 慣例（跟「終止訓練」「清空快取」同一套模式，不另外做卡片式按鈕 UI）
// 問使用者要不要繼續，答案立刻送去 /api/agent/resume 接著跑；用迴圈是為了以後
// 如果同一輪內有不只一個 interrupt 點，也能一樣接住，不用另外改這段邏輯。
// resumeUrl 有預設值是為了讓 Manager 模式的呼叫端一行都不用改；Deep Agent 模式
// 傳 '/api/agent/deep/resume' 進來，走的是同一套「彈確認視窗→送回答→接續串流」
// 流程，差別只在端點與後端恢復值的格式（後端 build_resume_value 負責轉換）。
async function resumeAgentInterruptsWithModal(result, threadId, modelType, resumeUrl = '/api/agent/resume') {
    let { traceRefs, finalReply, sawError, interruptPayload } = result;
    while (interruptPayload) {
        if (interruptPayload.dataset_name) await refreshRoboflowStatus();
        const approved = await showAgentConfirmation(interruptPayload);
        const res = await fetch(resumeUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                thread_id: threadId,
                model_type: modelType,
                approved,
                interrupt_id: interruptPayload.interrupt_id,
                operation_id: interruptPayload.operation_id || null
            })
        });
        if (!res.ok) {
            const errorData = await res.json().catch(() => ({}));
            throw new Error(errorData.detail || '確認已失效，請重新整理後再試。');
        }
        const next = await consumeAgentStream(res, traceRefs);
        traceRefs = next.traceRefs;
        finalReply = next.finalReply;
        sawError = next.sawError;
        interruptPayload = next.interruptPayload;
    }
    return { traceRefs, finalReply, sawError, interruptPayload };
}

async function finishOrResumeAgentTurn(result, threadId, modelType, resumeUrl = '/api/agent/resume') {
    result = await resumeAgentInterruptsWithModal(result, threadId, modelType, resumeUrl);
    let { traceRefs, finalReply, sawError, interruptPayload } = result;

    // 空字串也要當成「沒有回覆」：直接 appendMessage('') 會畫出一顆空氣泡，使用者
    // 看不出是跑完了還是靜默失敗。Deep Agent 後端已經改成回 null + error 事件，這裡
    // 再擋一層是為了 Manager 模式（共用這個函式）也一併受益。
    if (finalReply !== null && String(finalReply).trim()) {
        let reply = finalReply;
        const healthMatch = reply.match(/\[HEALTH_DATA: ({.*?})\]/);
        if (healthMatch) {
            const healthJson = JSON.parse(healthMatch[1]);
            reply = reply.replace(healthMatch[0], "");
            appendMessage(reply, 'ai');
            appendHealthCard(healthJson);
        } else {
            appendMessage(reply, 'ai');
        }
        chatMessages.push({ role: 'assistant', content: reply });
        persistChatHistory();
        // 有回覆但過程中出過錯：回覆可能不完整，甚至是模型在沒有結果的情況下編出來的。
        // 只把警告留在決策過程卡片裡不夠——那要展開才看得到，使用者會直接相信上面那段話。
        if (sawError) {
            appendMessage("⚠️ 這一輪有步驟失敗，上面的回覆可能不完整或不準確，請展開決策過程確認。", 'error');
        }
    } else if (sawError) {
        appendMessage("執行中發生錯誤，詳情請展開上方決策過程", 'error');
    } else {
        appendMessage("這一輪沒有產生任何回覆，展開上方決策過程可以看到實際執行到哪一步。", 'error');
    }
}

// ───────────────────────────────────────────────────────────
// 訓練是同步卡在 model_agent 的 tool 裡跑的（見 agent_tools.train_yolo_model），
// 那段期間 /api/agent/chat 的 SSE 串流整個被卡住、推不出任何 step，LLM 視窗會
// 「看起來當掉」。但 trainer 每個 epoch 都即時更新 shared_trainer.progress，而
// /api/train/status 是另一條執行緒在服務、不受卡住的 SSE 影響。所以對話回合期間
// 獨立輪詢這個端點，把進度即時畫成卡片塞進 LLM 視窗，使用者才看得到訓練跑到哪、
// 要不要按終止（終止鈕沿用網頁表單那顆的 /api/train/stop，對同一個 shared_trainer 生效）。
let agentTrainWatchInterval = null;
let agentTrainCardRefs = null;

function startAgentTrainingWatch() {
    if (agentTrainWatchInterval) return;
    agentTrainCardRefs = null;
    agentTrainWatchInterval = setInterval(async () => {
        try {
            const r = await fetch('/api/train/status');
            const d = await r.json();
            if (d.status === 'training') {
                renderAgentTrainingCard(d);
            } else if (agentTrainCardRefs) {
                finishAgentTrainingCard(d);
                agentTrainCardRefs = null;
            }
        } catch (e) { /* 輪詢失敗忽略，下一輪再試 */ }
    }, 2000);
}

function stopAgentTrainingWatch() {
    if (agentTrainWatchInterval) {
        clearInterval(agentTrainWatchInterval);
        agentTrainWatchInterval = null;
    }
    // 卡片保留在最後一次畫面（完成／終止／錯誤）不清掉，只放掉 refs，下一輪訓練會開新卡片
    agentTrainCardRefs = null;
}

function renderAgentTrainingCard(d) {
    const container = document.getElementById('ai-chat-history');
    if (!agentTrainCardRefs) {
        const wrapper = document.createElement('div');
        wrapper.style.cssText = "align-self: flex-start; width: 92%; margin-top: 6px; background: #0f172a; border: 1px solid rgba(34,197,94,0.4); border-radius: 10px; padding: 12px 14px; font-size: 0.78rem; color: #cbd5e1;";
        wrapper.innerHTML = `
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                <span class="atc-title" style="color:#4ade80; font-weight:600;">🚀 訓練進行中</span>
                <button type="button" class="atc-stop" style="background:#ef4444; color:#fff; border:none; border-radius:6px; padding:3px 10px; font-size:0.72rem; cursor:pointer;">終止訓練</button>
            </div>
            <div style="background:#1e293b; border-radius:6px; height:8px; overflow:hidden; margin-bottom:6px;">
                <div class="atc-bar" style="background:#4ade80; height:100%; width:0%; transition:width .3s;"></div>
            </div>
            <div class="atc-text" style="color:#94a3b8;">準備中…</div>
            <div class="atc-metric" style="color:#64748b; margin-top:4px;"></div>`;
        const stopBtn = wrapper.querySelector('.atc-stop');
        stopBtn.addEventListener('click', async () => {
            stopBtn.disabled = true;
            stopBtn.textContent = '終止中…';
            try { await fetch('/api/train/stop', { method: 'POST' }); } catch (e) {}
        });
        container.appendChild(wrapper);
        agentTrainCardRefs = {
            wrapper,
            title: wrapper.querySelector('.atc-title'),
            stop: stopBtn,
            bar: wrapper.querySelector('.atc-bar'),
            text: wrapper.querySelector('.atc-text'),
            metric: wrapper.querySelector('.atc-metric'),
        };
    }
    const refs = agentTrainCardRefs;
    const cur = d.current_epoch || 0;
    const total = d.total_epochs || 0;
    // 優先用後端 percent（epoch 內就會動），沒有才退回 epoch 比例——這是修掉「整個 epoch
    // 卡在 0% 像當掉」的關鍵。
    const pct = (d.percent != null) ? Math.round(d.percent) : (total ? Math.round((cur / total) * 100) : 0);
    refs.bar.style.width = `${pct}%`;
    refs.text.textContent = `第 ${cur} / ${total} 輪（${pct}%）`;
    const last = (d.metrics && d.metrics.length) ? d.metrics[d.metrics.length - 1] : null;
    refs.metric.textContent = last ? `最新：mAP50 ${last.map}% ・ loss ${last.loss}` : '';
    container.scrollTop = container.scrollHeight;
}

function finishAgentTrainingCard(d) {
    const refs = agentTrainCardRefs;
    if (!refs) return;
    const [label, color] = ({
        completed: ['✅ 訓練完成', '#4ade80'],
        stopped: ['🛑 訓練已終止', '#f87171'],
        error: ['⚠️ 訓練發生錯誤', '#f87171'],
    })[d.status] || ['訓練結束', '#94a3b8'];
    refs.title.textContent = label;
    refs.title.style.color = color;
    if (refs.stop) refs.stop.remove();
    if (d.status === 'completed') refs.bar.style.width = '100%';
    if (d.message) refs.metric.textContent = d.message;
}

async function sendManagerAgentMessage(text, loadingId) {
    const modelType = document.getElementById('manager-agent-engine-select').value;
    const threadId = getOrCreateThreadId();
    const requestBody = { message: text, model_type: modelType, thread_id: threadId };

    startAgentTrainingWatch();
    try {
        const res = await fetch('/api/agent/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });
        if (!res.ok) {
            const errorData = await res.json().catch(() => ({}));
            throw new Error(errorData.detail || 'Agent 任務無法啟動');
        }

        updateManagerAgentLoadingMessage(loadingId, 'Manager 正在連線並準備執行任務…');
        const result = await consumeAgentStream(res, null, loadingId);
        removeMessage(loadingId);
        await finishOrResumeAgentTurn(result, threadId, modelType);
    } catch (e) {
        removeMessage(loadingId);
        appendMessage(e.message || "連線失敗", 'error');
    } finally {
        stopAgentTrainingWatch();
    }
}

// Deep Agent 模式：跟 sendManagerAgentMessage 是兩條各自獨立的路徑，只共用
// consumeAgentStream／finishOrResumeAgentTurn 這些「解析 SSE、彈確認視窗」的顯示層
// ——後端刻意讓 /api/agent/deep/chat 產生跟 /api/agent/chat 完全相同的事件形狀
// （status／step／interrupt／error／final），所以這一層不用為第三個模式再寫一套。
// thread_id 用 getOrCreateDeepThreadId()，理由見 DEEP_AGENT_THREAD_KEY 的註解。
async function sendDeepAgentMessage(text, loadingId) {
    const modelType = document.getElementById('manager-agent-engine-select').value;
    const threadId = getOrCreateDeepThreadId();
    const requestBody = { message: text, model_type: modelType, thread_id: threadId };

    startAgentTrainingWatch();
    try {
        const res = await fetch('/api/agent/deep/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });
        if (!res.ok) {
            const errorData = await res.json().catch(() => ({}));
            throw new Error(errorData.detail || 'Deep Agent 任務無法啟動');
        }

        updateManagerAgentLoadingMessage(loadingId, 'Deep Agent 正在連線並準備執行任務…');
        const result = await consumeAgentStream(res, null, loadingId);
        removeMessage(loadingId);
        await finishOrResumeAgentTurn(result, threadId, modelType, '/api/agent/deep/resume');
    } catch (e) {
        removeMessage(loadingId);
        appendMessage(e.message || "連線失敗", 'error');
    } finally {
        stopAgentTrainingWatch();
    }
}

// Evaluation Agent 評估完就把「模型效能診斷」頁的三張圖直接畫在對話裡（混淆矩陣／
// F1／P-R 曲線），使用者不用再切到診斷頁自己選模型。圖表路徑來自 skill_eval 的結構化
// 回傳，不是 LLM 轉述的，所以不會有破圖。
function appendEvalChartsCard(payload) {
    const charts = payload.charts || [];
    if (!charts.length) return;
    const container = document.getElementById('ai-chat-history');
    const card = document.createElement('div');
    card.style.cssText = "align-self: flex-start; background: #0f172a; border: 1px solid #3b82f6; padding: 14px; border-radius: 12px; width: 92%; margin-top: 10px;";
    const title = payload.model_name ? `模型效能診斷圖表：${payload.model_name}` : '模型效能診斷圖表';
    card.innerHTML = `
        <div style="font-size: 0.8rem; color: #93c5fd; font-weight: 700; margin-bottom: 10px;">📊 ${title}</div>
        <div style="display: flex; flex-direction: column; gap: 12px;">
            ${charts.map(c => `
                <div>
                    <div style="font-size: 0.7rem; color: #64748b; margin-bottom: 4px;">${c.name}</div>
                    <img src="${c.url}" alt="${c.name}" loading="lazy"
                        style="width: 100%; border-radius: 8px; background: #fff; cursor: zoom-in;"
                        onclick="window.open(this.src, '_blank')">
                </div>
            `).join('')}
        </div>
        <div style="font-size: 0.65rem; color: #64748b; margin-top: 8px;">點圖可開新視窗放大檢視。</div>
        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,.08);">
            <div style="font-size: 0.72rem; color: #cbd5e1; margin-bottom: 6px;">想用實際圖片驗證這個模型嗎？</div>
            <button onclick="document.getElementById('ai-chat-file-input').click()"
                style="background: rgba(59,130,246,.18); border: 1px solid rgba(59,130,246,.45); color: #93c5fd; border-radius: 8px; padding: 5px 10px; font-size: 0.72rem; cursor: pointer;">📎 上傳圖片做推論</button>
            <span style="font-size: 0.65rem; color: #64748b; margin-left: 6px;">也可以直接把圖片拖進對話視窗</span>
        </div>
    `;
    container.appendChild(card);
    container.scrollTop = container.scrollHeight;
}

function appendHealthCard(data) {
    const container = document.getElementById('ai-chat-history');
    const card = document.createElement('div');
    card.style.cssText = "align-self: flex-start; background: #0f172a; border: 1px solid #3b82f6; padding: 16px; border-radius: 12px; width: 80%; margin-top: 10px;";
    
    // 同上：同一毫秒內畫兩張健康卡會拿到相同的 canvas id，圖表就會畫錯位置
    const chartId = `chat-chart-${Date.now()}-${++messageSeq}`;
    card.innerHTML = `
        <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 8px; display:flex; align-items:center; gap:5px;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2.5"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
            AI 診斷報告: ${data.coverage}% 標記率
        </div>
        <div style="display: flex; align-items: center; gap: 15px;">
            <div style="width: 100px; height: 100px;">
                <canvas id="${chartId}"></canvas>
            </div>
            <div style="flex: 1;">
                <div style="font-size: 1.2rem; font-weight: 800; color: #3b82f6;">Score: ${data.score}</div>
                <div style="font-size: 0.7rem; color: #94a3b8;">平衡度: ${data.balance}%</div>
                <div style="font-size: 0.7rem; color: #94a3b8;">總數: ${data.total_images} 張</div>
            </div>
        </div>
    `;
    
    container.appendChild(card);
    container.scrollTop = container.scrollHeight;
    
    // 延遲一點點渲染圖表，確保 DOM 已經掛載
    setTimeout(() => {
        const ctx = document.getElementById(chartId).getContext('2d');
        const labels = Object.values(data.class_distribution).map(v => v.name);
        const counts = Object.values(data.class_distribution).map(v => v.count);
        const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4'];
        
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: counts,
                    backgroundColor: labels.map((_, i) => colors[i % colors.length]),
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } }
            }
        });
    }, 100);
}

const MANAGER_AGENT_NODE_LABELS = {
    planner: '🗺️ Planner（規劃）',
    manager: '🧭 Manager（決策）',
    observer: '🔍 Observer（觀測）',
    data_agent: '📁 Data Agent',
    model_agent: '🤖 Model Agent',
    evaluation_agent: '📊 Evaluation Agent'
};

function escapeHtml(value) {
    const div = document.createElement('div');
    div.textContent = value === null || value === undefined ? '' : String(value);
    return div.innerHTML;
}

// 把 /api/agent/chat 串流回來的單個 trace step（Planner 的計畫、Manager 的
// 路由決策、Observer 的判讀、子 Agent 呼叫的工具）轉成一段 HTML，給
// appendTraceStep 逐步塞進卡片——不然使用者只看得到最終一句回覆，看不出
// 三個 Agent 實際怎麼協作。
function renderTraceStepHtml(step, index) {
    const label = MANAGER_AGENT_NODE_LABELS[step.node] || step.node;
    let inner = `<div style="margin-top:8px; padding-top:8px; border-top:1px solid rgba(148,163,184,0.15);">`;
    inner += `<div style="color:#e2e8f0; font-weight:600;">#${index} ${escapeHtml(label)}</div>`;

    if (step.decision) {
        inner += `<div>→ 交給：<b>${escapeHtml(step.decision)}</b></div>`;
    }
    if (step.plan && step.plan.goal) {
        inner += `<div>🎯 目標：${escapeHtml(step.plan.goal)}</div>`;
        if (step.plan.steps && step.plan.steps.length) {
            inner += '<ul style="margin:4px 0 0 16px; padding:0;">' +
                step.plan.steps.map(s => `<li>[${escapeHtml(s.status)}] ${escapeHtml(s.description)}（${escapeHtml(s.agent)}）</li>`).join('') +
                '</ul>';
        }
    }
    if (step.tool_calls && step.tool_calls.length) {
        inner += step.tool_calls.map(tc => {
            if (tc.tool) {
                if (tc.tool === 'prepare_roboflow_augmentation') {
                    return `<div style="color:#7dd3fc;">🔎 Data Agent 正在搜尋 Roboflow 候選並下載到暫存區評估…</div>`;
                }
                if (tc.tool === 'search_roboflow_universe') {
                    return `<div style="color:#7dd3fc;">🌐 正在查詢 Roboflow Universe：${escapeHtml(tc.args.query || '')}</div>`;
                }
                return `<div>🔧 呼叫工具：<b>${escapeHtml(tc.tool)}</b>（${escapeHtml(JSON.stringify(tc.args))}）</div>`;
            }
            if (tc.tool_result === 'prepare_roboflow_augmentation') {
                let result = tc.output;
                if (typeof result === 'string') {
                    try { result = JSON.parse(result); } catch (e) { result = null; }
                }
                if (result && result.status === 'proposal_ready') {
                    const candidate = result.candidate || {};
                    return `<div style="margin-top:5px; padding:7px; border-radius:6px; background:rgba(14,116,144,.13); color:#bae6fd;">` +
                        `✅ 暫存評估完成：比較 ${escapeHtml(result.candidate_count || 0)} 個候選，建議 ` +
                        `<b>${escapeHtml(candidate.name || candidate.project)}</b> v${escapeHtml(candidate.version)}；` +
                        `可安全加入 ${escapeHtml(result.usable_images)} 張。等待最後確認，不會先修改正式資料集。</div>`;
                }
            }
            return `<div>📥 工具回傳（${escapeHtml(tc.tool_result)}）：${escapeHtml(typeof tc.output === 'string' ? tc.output : JSON.stringify(tc.output))}</div>`;
        }).join('');
    }
    if (step.report) {
        inner += `<div>📝 回報：status=<b>${escapeHtml(step.report.status)}</b>` +
            (step.report.reason_code ? ` reason_code=${escapeHtml(step.report.reason_code)}` : '') +
            `<br>理由：${escapeHtml(step.report.reason)}` +
            (step.report.suggestion ? `<br>建議：${escapeHtml(step.report.suggestion)}` : '') +
            `</div>`;
    }
    if (step.evaluation) {
        inner += `<div>🔍 判讀：verdict=<b>${escapeHtml(step.evaluation.verdict)}</b>` +
            `，有進展=${step.evaluation.progress_since_last_round ? '是' : '否'}` +
            (step.evaluation.contradiction ? '，⚠️ 偵測到矛盾' : '') +
            `</div>`;
    }
    if (step.text) {
        inner += `<div>💬 ${escapeHtml(step.text)}</div>`;
    }
    // 模型回空訊息這種「provider 回 200 但實際上失敗了」的狀況，只有這裡看得到。
    if (step.warning) {
        inner += `<div style="color:#fbbf24; word-break:break-all;">⚠️ ${escapeHtml(step.warning)}</div>`;
    }
    if (step.usage) {
        inner += `<div style="color:#64748b;">🔢 tokens：input ${step.usage.input_tokens} / output ${step.usage.output_tokens}</div>`;
    }
    inner += `</div>`;
    return inner;
}

// 建立空的決策過程卡片，跑到哪一步就用 appendTraceStep 補進去，不用等全部
// 跑完才一次性渲染——預設展開（open），因為需求就是「跑到哪就看到哪」。
function createTraceCard() {
    const container = document.getElementById('ai-chat-history');

    const wrapper = document.createElement('details');
    wrapper.open = true;
    wrapper.style.cssText = "align-self: flex-start; width: 92%; margin-top: 6px; background: #0f172a; border: 1px solid rgba(59,130,246,0.3); border-radius: 10px; padding: 8px 12px; font-size: 0.72rem; color: #94a3b8; overflow-wrap: anywhere; word-break: break-word;";

    const summary = document.createElement('summary');
    summary.style.cssText = "cursor: pointer; color: #60a5fa; font-weight: 600; outline: none;";
    summary.textContent = '🔬 決策過程（執行中…）';
    wrapper.appendChild(summary);

    const stepsContainer = document.createElement('div');
    wrapper.appendChild(stepsContainer);

    container.appendChild(wrapper);
    container.scrollTop = container.scrollHeight;

    return { wrapper, summary, stepsContainer, stepCount: 0 };
}

function appendTraceStep(refs, step) {
    refs.stepCount += 1;
    const stepDiv = document.createElement('div');
    stepDiv.innerHTML = renderTraceStepHtml(step, refs.stepCount);
    refs.stepsContainer.appendChild(stepDiv);
    refs.summary.textContent = `🔬 決策過程（共 ${refs.stepCount} 步，點開查看）`;

    const container = document.getElementById('ai-chat-history');
    container.scrollTop = container.scrollHeight;
}

function appendTraceError(refs, message) {
    const errDiv = document.createElement('div');
    errDiv.style.cssText = "margin-top:8px; padding-top:8px; border-top:1px solid rgba(239,68,68,0.3); color:#f87171;";
    errDiv.textContent = `⚠️ 執行中發生錯誤：${message}`;
    refs.stepsContainer.appendChild(errDiv);
    refs.summary.textContent = `🔬 決策過程（共 ${refs.stepCount} 步，中途失敗）`;
}

// 跑完後在卡片底部補一行總計，讓使用者不用自己把每一步的 token 加起來。
// usage/llmCalls 可能是 undefined（模型沒填 usage_metadata，例如本機 Ollama）。
function finishTraceCard(refs, usage, llmCalls) {
    if (!usage || !usage.total_tokens) return;
    const summaryDiv = document.createElement('div');
    summaryDiv.style.cssText = "margin-top:8px; padding-top:8px; border-top:1px solid rgba(148,163,184,0.15); color:#64748b;";
    summaryDiv.textContent = `📊 共 ${llmCalls || 0} 次 LLM 呼叫，${usage.total_tokens} tokens（input ${usage.input_tokens} / output ${usage.output_tokens}）`;
    refs.stepsContainer.appendChild(summaryDiv);
}

// 對話泡泡的 id 以前是 'msg-' + Date.now()，但 sendChatMessage 會在同一個同步流程裡
// 連續建立「使用者訊息」跟「載入中訊息」——同一毫秒內就會拿到相同的 id。
// getElementById 只回傳第一個相符的元素，於是 removeMessage/更新載入文字都會打到
// 使用者那顆泡泡：訊息憑空消失、或被改寫成「Manager 正在擬訂計畫」。加一個遞增序號
// 保證唯一。
let messageSeq = 0;

function appendMessage(text, sender, isLoading = false) {
    const container = document.getElementById('ai-chat-history');
    const msgDiv = document.createElement('div');
    const id = `msg-${Date.now()}-${++messageSeq}`;
    msgDiv.id = id;
    
    msgDiv.style.padding = '10px 14px';
    msgDiv.style.borderRadius = '12px';
    msgDiv.style.maxWidth = '85%';
    msgDiv.style.lineHeight = '1.4';
    
    if (sender === 'user') {
        msgDiv.style.alignSelf = 'flex-end';
        msgDiv.style.background = '#3b82f6';
        msgDiv.style.color = 'white';
    } else if (sender === 'error') {
        msgDiv.style.alignSelf = 'flex-start';
        msgDiv.style.background = 'rgba(239, 68, 68, 0.2)';
        msgDiv.style.border = '1px solid rgba(239, 68, 68, 0.5)';
        msgDiv.style.color = '#fca5a5';
    } else {
        msgDiv.style.alignSelf = 'flex-start';
        msgDiv.style.background = 'rgba(59, 130, 246, 0.1)';
        msgDiv.style.border = '1px solid rgba(59, 130, 246, 0.2)';
        msgDiv.style.color = '#e2e8f0';
        if (isLoading) msgDiv.style.opacity = '0.5';
    }
    
    msgDiv.innerText = text;
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function quickPrompt(text) {
    document.getElementById('ai-chat-input').value = text;
    sendChatMessage();
}

// --- 整合驗證中心邏輯 ---
let unifiedFiles = [];

function handleVerifyUpload(input, type) {
    const files = Array.from(input.files);
    const imageExtensions = /\.(jpg|jpeg|png|bmp|webp)$/i;
    unifiedFiles = files.filter(f => f.name.match(imageExtensions));
    
    // 清理另一個輸入框的選擇
    if (type === 'folder') {
        document.getElementById('file-input-verify-files').value = '';
    } else {
        document.getElementById('file-input-verify-folder').value = '';
    }
    
    const statusText = document.getElementById('verify-status-text');
    if (unifiedFiles.length > 0) {
        if (type === 'folder') {
            statusText.innerText = `📁 已選資料夾，含 ${unifiedFiles.length} 張圖片`;
        } else {
            statusText.innerText = `📸 已選 ${unifiedFiles.length} 張圖片`;
        }
        statusText.style.color = '#3b82f6';
    } else {
        statusText.innerText = '未選擇任何有效影像';
        statusText.style.color = '#ef4444';
    }
}

async function startUnifiedVerification() {
    const model = document.getElementById('verify-model-select').value;
    const imgsz = document.getElementById('verify-imgsz-select').value;
    const conf = document.getElementById('verify-conf-range').value;
    const iou = document.getElementById('verify-iou-range').value;
    const tta = document.getElementById('verify-tta-toggle').checked;
    const container = document.getElementById('verify-results-container');

    if (unifiedFiles.length === 0) {
        showAlert("請先選擇圖片或資料夾", "error");
        return;
    }

    container.innerHTML = '<div style="color:#3b82f6; grid-column: 1/-1; text-align:center; padding:20px;">AI 正在分析中...</div>';

    try {
        const formData = new FormData();
        unifiedFiles.forEach(f => formData.append('files', f));
        formData.append('model', model);
        formData.append('imgsz', imgsz);
        formData.append('conf', conf);
        formData.append('iou', iou);
        formData.append('augment', tta);

        const res = await fetch('/api/verify/files', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.status === 'error') throw new Error(data.message);
        
        renderVerifyResults(data.results);
    } catch (err) {
        showAlert("檢測失敗: " + err.message, "error");
        container.innerHTML = '';
    }
}

function renderVerifyResults(results) {
    const container = document.getElementById('verify-results-container');
    if (results.length === 0) {
        container.innerHTML = '<div style="color:#94a3b8; grid-column: 1/-1; text-align:center; padding:20px;">未偵測到任何結果</div>';
        return;
    }

    container.innerHTML = results.map(r => `
        <div class="card" style="padding:0; overflow:hidden; background: rgba(30, 41, 59, 0.5);">
            <img src="${r.url}?t=${Date.now()}" style="width:100%; height:150px; object-fit:cover; cursor:pointer;" onclick="window.open(this.src)">
            <div style="padding:10px; font-size:0.75rem;">
                <div style="color:#fff; font-weight:700; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${r.name}</div>
                <div style="color:#64748b; margin-top:4px;">目標數: ${r.objects ? r.objects.length : 0}</div>
                ${r.objects && r.objects.length > 0 ? `
                    <div style="margin-top:8px; border-top:1px solid rgba(255,255,255,0.05); padding-top:6px; max-height:100px; overflow-y:auto; display:flex; flex-direction:column; gap:4px;">
                        ${r.objects.map(obj => {
                            const [x1, y1, x2, y2] = obj.bbox.map(coord => Math.round(coord));
                            return `
                                <div style="display:flex; justify-content:space-between; color:#cbd5e1; font-size:0.65rem;">
                                    <span style="color:#60a5fa; font-weight:600;">${obj.name}</span>
                                    <span>${(obj.confidence * 100).toFixed(0)}% [${x1},${y1},${x2},${y2}]</span>
                                </div>
                            `;
                        }).join('')}
                    </div>
                ` : ''}
            </div>
        </div>
    `).join('');
}
async function loadModels() {
    const select = document.getElementById('verify-model-select');
    if (!select) return;  // 助手在別的頁面時沒有這個下拉，直接跳過，避免存取 null 出錯
    try {
        const res = await fetch('/api/models');
        const data = await res.json();
        const diagSelect = document.getElementById('diagnosis-model-select');

        const current = select.value;
        const currentDiag = diagSelect ? diagSelect.value : '';
        
        select.innerHTML = '';
        if (diagSelect) diagSelect.innerHTML = '<option value="">-- 請選擇要診斷的模型 --</option>';
        
        if (data.models && data.models.length > 0) {
            data.models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m;
                opt.textContent = m;
                select.appendChild(opt);
                
                if (diagSelect) {
                    const optDiag = document.createElement('option');
                    optDiag.value = m;
                    optDiag.textContent = m;
                    diagSelect.appendChild(optDiag);
                }
            });
            if (current && data.models.includes(current)) {
                select.value = current;
            } else if (data.models.includes('yolov9m.pt')) {
                select.value = 'yolov9m.pt';
            }
            
            if (diagSelect && currentDiag && data.models.includes(currentDiag)) {
                diagSelect.value = currentDiag;
            }
        } else {
            select.innerHTML = '<option value="">無可用模型</option>';
        }
    } catch (e) {
        console.error('Load models error:', e);
    }
}

// 刪除驗證中心目前選取的訓練模型。前端先擋一次非 *_best.pt，後端也會再驗證，
// 預訓練權重（yolov9s.pt 這類）兩邊都不給刪。
async function deleteSelectedModel(selectId = 'verify-model-select') {
    const select = document.getElementById(selectId);
    const name = select ? select.value : '';
    if (!name) { showAlert("請先選擇要刪除的模型", "error"); return; }
    if (!name.endsWith('_best.pt')) { showAlert("只能刪除訓練產出的模型（*_best.pt），預訓練權重不可刪除", "error"); return; }
    if (!confirm(`確定刪除模型「${name}」？此動作無法復原。`)) return;
    try {
        const formData = new FormData();
        formData.append('model_name', name);
        const res = await fetch('/api/models/delete', { method: 'POST', body: formData });
        const data = await res.json();
        showAlert(data.message, data.status === 'success' ? 'success' : 'error');
        if (data.status === 'success') loadModels();
    } catch (e) {
        showAlert("刪除失敗: " + e.message, "error");
    }
}

// 頁面載入時初始化模型選單
document.addEventListener('DOMContentLoaded', () => {
    loadModels();
});

async function loadDiagnosisCharts() {
    // 從診斷頁面獲取當前選用的模型，並推導出訓練資料夾名稱
    const selectedModel = document.getElementById('diagnosis-model-select').value;
    if (!selectedModel) {
        showAlert("請先選擇要診斷的模型", "error");
        return;
    }
    
    // 假設檔名格式為: my_model_best.pt，則資料夾名稱為 my_model
    let folderName = selectedModel.replace('_best.pt', '').replace('.pt', '');

    const charts = [
        { id: 'chart-confusion', name: 'confusion_matrix.png' },
        { id: 'chart-f1', name: 'BoxF1_curve.png' },
        { id: 'chart-pr', name: 'BoxPR_curve.png' }
    ];

    for (const chart of charts) {
        const el = document.getElementById(chart.id);
        if (el) {
            // 帶上 model 參數
            el.src = `/api/model/evaluation/charts/${chart.name}?model=${encodeURIComponent(folderName)}&t=${Date.now()}`;
            el.onerror = () => {
                console.warn(`Chart ${chart.name} not found.`);
                el.alt = "尚未生成此報表 (請先完成模型訓練)";
            };
        }
    }
    showAlert(`正在載入 ${folderName} 的診斷報告...`, "success");
}
